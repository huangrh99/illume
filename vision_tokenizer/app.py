import argparse
import datetime
import json
import os
import time
import torch
import gradio as gr
from PIL import Image
from tokenizer.sdxl_decoder_pipe import StableDiffusionXLDecoderPipeline
from torchvision import transforms
import logging
from utils.registry_utils import Config
from tokenizer.builder import build_vq_model
from dataset.multi_ratio_dataset import get_image_size, assign_ratio


def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config


def build_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = build_logger("gradio_web_server", "gradio_web_server.log")

vq_model = None
is_ema_model = False
diffusion_pipeline = None
lazy_load = False

# diffusion decoder hyperparameters.
resolution_list = [
    (1024, 1024), (768, 1024), (1024, 768),
    (512, 2048), (2048, 512), (640, 1920),
    (1920, 640), (768, 1536),
    (1536, 768), (768, 1152), (1152, 768)
]

cfg_range = (1, 10.0)
step_range = (1, 100)


def resize_to_shortest_edge(img, shortest_edge_resolution):
    width, height = img.size

    if width < height:
        new_width = shortest_edge_resolution
        new_height = int(height * (new_width / width))
    elif height < width:
        new_height = shortest_edge_resolution
        new_width = int(width * (new_height / height))
    else:
        new_width = shortest_edge_resolution
        new_height = shortest_edge_resolution

    resized_img = img.resize((new_width, new_height))
    return resized_img


from PIL import Image


def resize_to_square_with_long_edge(image: Image.Image, size: int = 512):
    """Resize image so that its *long* side equals `size`, short side scaled proportionally."""
    width, height = image.size
    if width > height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    return image.resize((new_width, new_height), Image.LANCZOS)


def pad_to_square(image: Image.Image, target_size: int = 512, color=(255, 255, 255)):
    image = resize_to_square_with_long_edge(image, target_size)
    new_img = Image.new("RGB", (target_size, target_size), color)
    offset_x = (target_size - image.width) // 2
    offset_y = (target_size - image.height) // 2
    new_img.paste(image, (offset_x, offset_y))
    return new_img


def load_vqgan_model(args, model_dtype='fp16', use_ema=False, ):
    global vq_model
    vq_model = build_vq_model(args.vq_model)

    if model_dtype == 'fp16':
        vq_model = vq_model.to(torch.float16)
        logger.info("Convert the model dtype to float16")
    elif model_dtype == 'bf16':
        vq_model = vq_model.to(torch.bfloat16)
        logger.info("Convert the model dtype to bfloat16")

    vq_model.to('cuda')
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")

    if "ema" in checkpoint:
        ema_state_dict = checkpoint["ema"]
    else:
        ema_state_dict = None

    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    else:
        model_state_dict = checkpoint

    if use_ema:
        vq_model.load_state_dict(ema_state_dict, strict=True)
    else:
        vq_model.load_state_dict(model_state_dict, strict=True)
    return vq_model


def load_diffusion_decoder(args):
    global diffusion_pipeline
    diffusion_pipeline = StableDiffusionXLDecoderPipeline.from_pretrained(
        args.sdxl_decoder_path,
        add_watermarker=False,
        vq_config=args,
        vq_model=vq_model,
    )
    diffusion_pipeline.to(vq_model.device)


def vqgan_diffusion_decoder_reconstruct(input_image, diffusion_upsample, cfg_values, steps):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(vq_model.device)

    org_width, org_height = input_image.size
    if diffusion_upsample:
        width, height = org_width * 2, org_height * 2
    else:
        width, height = org_width, org_height

    print(diffusion_upsample, org_width, org_height, width, height)
    group_index = assign_ratio(height, width, resolution_list)
    select_h, select_w = resolution_list[group_index]

    diffusion_outputs = diffusion_pipeline(
        images=input_tensor,
        height=select_h,
        width=select_w,
        guidance_scale=cfg_values,
        num_inference_steps=steps
    )
    sample = diffusion_outputs.images[0]
    sample.resize((width, height))
    return sample, f"📐 **Output Resolution**: {width}x{height}"


@torch.no_grad()
def vqgan_reconstruct(input_image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    org_width, org_height = input_image.size

    width = org_width // 16 * 16
    height = org_height // 16 * 16

    input_image = input_image.resize((width, height))
    input_tensor = transform(input_image).unsqueeze(0).to(vq_model.device)

    with torch.no_grad():
        inputs = vq_model.get_input(dict(image=input_tensor))
        (quant_semantic, _, _, _), \
        (quant_detail, _, _) = vq_model.encode(**inputs)
        reconstructed_image = vq_model.decode(quant_semantic, quant_detail)

    reconstructed_image = torch.clamp(127.5 * reconstructed_image + 128.0, 0, 255)
    reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')

    output_image = Image.fromarray(reconstructed_image)
    output_image.resize((org_width, org_height))
    return output_image, f"📐 **Output Resolution**: {org_width}x{org_height}"


title_markdown = '''# DualViTok Demo
The DualViTok is a dual-branch vision tokenizer designed to capture both deep semantics and fine-grained textures. Implementation details can be found in ILLUME+[[ArXiv](https://arxiv.org/abs/2504.01934)].  
'''

usage_markdown = """
<details>
<summary><strong>📘 Usage Instructions (click to expand)</strong></summary>

1. Upload an image and click the <strong>Reconstruct</strong> button.  
2. Set <code>Max Shortest Side</code> to limit the image resolution.  
3. Click <code>Force Upscale to Max Shortest Side to enable <strong>Force Upscale</strong> to resize the shortest side of the image to the <code>Max Shortest Side</code>.  
4. <em>(Optional)</em> Check <code>Use EMA model</code> to use the EMA checkpoint for reconstruction.  
5. <em>(Optional)</em> Click <code>Load Diffusion Decoder</code> to enable Diffusion Model decoding.  
   You can also enable <code>2x Upsample</code> to apply super-resolution to the uploaded image.

</details>
"""


def build_gradio_interface(args):
    if not lazy_load:
        load_vqgan_model(args, model_dtype=args.model_dtype)

    with gr.Blocks() as demo:
        gr.Markdown(title_markdown)
        gr.Markdown(usage_markdown)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🖼️ Input Image")
                input_image = gr.Image(type="pil", label="Upload Image", width=384, height=384)
                input_resolution_display = gr.Markdown("")
                gr.Examples(
                    examples=[
                        ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/1.png",],
                        ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/2.png",],
                        ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/3.png",],
                    ],
                    inputs=input_image,
                    label="Example Images",
                )

            with gr.Column():
                gr.Markdown("## 🔄 Reconstructed Image")
                output_image_recon = gr.Image(type="pil", label="Reconstruction", width=384, height=384)
                output_resolution_display = gr.Markdown("")

            with gr.Column():
                gr.Markdown("## ⚙ Hyperparameters")
                # with gr.Row():
                short_resolution_dropdown = gr.Dropdown(
                    choices=[None, 256, 384, 512, 1024],
                    value=1024,
                    label="Max Shortest Side"
                )
                force_upscale_checkbox = gr.Checkbox(label="Force Upscale to Max Shortest Side", value=False)
                use_ema_checkbox = gr.Checkbox(label="Use EMA Model", value=False)

                with gr.Accordion("🔽 Use Diffusion Decoder", open=False):
                    use_diffusion_checkbox = gr.Checkbox(label="Load Diffusion Decoder", value=False)
                    diffusion_upsample_checkbox = gr.Checkbox(label="Enable 2x Upsample", value=False)
                    cfg_slider = gr.Slider(
                        minimum=cfg_range[0], maximum=cfg_range[1],
                        step=0.5, value=1.5,
                        label="CFG Value"
                    )
                    step_slider = gr.Slider(
                        minimum=step_range[0], maximum=step_range[1],
                        step=1, value=20,
                        label="Inference Steps"
                    )
                reconstruct_btn = gr.Button("🚀 Reconstruct", variant="primary")

        def handle_input_image(image):
            if image is not None:
                image = image.convert("RGB")
                w, h = image.size
                return image, f"📐 **Input Resolution**: {w}x{h}"
            return None, ""

        input_image.change(
            handle_input_image,
            inputs=input_image,
            outputs=[input_image, input_resolution_display]
        )

        def reconstruct_fn(image, use_ema_flag, short_edge_resolution, force_upscale,
                           use_diffusion_flag, diffusion_upsample, cfg_value, num_steps):

            if short_edge_resolution is not None:
                if force_upscale or min(image.size) > short_edge_resolution:
                    image = resize_to_shortest_edge(image, int(short_edge_resolution))

            global vq_model
            if lazy_load and vq_model is None:
                load_vqgan_model(args, model_dtype=args.model_dtype)

            if use_ema_flag:
                if not is_ema_model:
                    load_vqgan_model(args, model_dtype=args.model_dtype, use_ema=True)
                    logger.info("Switched to EMA checkpoint")
            else:
                if is_ema_model:
                    load_vqgan_model(args, model_dtype=args.model_dtype, use_ema=False)
                    logger.info("Switched to non-EMA checkpoint")

            if use_diffusion_flag:
                if diffusion_pipeline is None:
                    load_diffusion_decoder(args)
                recon_image, resolution_str = vqgan_diffusion_decoder_reconstruct(image, diffusion_upsample, cfg_value,
                                                                                  num_steps)
            else:
                recon_image, resolution_str = vqgan_reconstruct(image)

            return pad_to_square(recon_image, target_size=384), resolution_str

        reconstruct_btn.click(
            reconstruct_fn,
            inputs=[input_image, use_ema_checkbox, short_resolution_dropdown, force_upscale_checkbox,
                    use_diffusion_checkbox, diffusion_upsample_checkbox, cfg_slider, step_slider],
            outputs=[output_image_recon, output_resolution_display])

    demo.launch(server_name='0.0.0.0')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--vq-ckpt", type=str, help="ckpt path for vq model")
    parser.add_argument("--torch-dtype", type=str, default='fp32')
    parser.add_argument("--model-dtype", type=str, default='fp32')
    parser.add_argument("--sdxl-decoder-path", type=str, default=None)
    parser.add_argument("--verbose", action='store_true')

    args = parser.parse_args()

    config = read_config(args.config)
    config.vq_ckpt = args.vq_ckpt
    config.torch_dtype = args.torch_dtype
    config.model_dtype = args.model_dtype
    config.verbose = args.verbose
    config.sdxl_decoder_path = args.sdxl_decoder_path

    build_gradio_interface(config)


if __name__ == "__main__":
    main()