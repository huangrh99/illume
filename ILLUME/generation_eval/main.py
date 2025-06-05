import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import itertools

from illume.data.data_utils import write_to_jsonl, unpad_and_resize_back

from generation_eval.generation_dataset.builder import build_eval_dataset
from generation_eval.models.builder import build_eval_model

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception as e:
    print("import torch_npu failed.")


def rank0_print(content):
    if int(os.getenv('WORLD_SIZE', '1')) > 1 and torch.distributed.get_rank() == 0:
        print(content)
    elif int(os.getenv('WORLD_SIZE', '1')) == 1:
        print(content)


def save_output_images(samples, batch_data, output_dir):
    for sample, info in zip(samples, batch_data):
        output_file = os.path.join(output_dir, info["out_image_path"])
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        img = Image.fromarray(sample.astype(np.uint8))

        if "original_sizes" in info:  # for editing task, unpad and resize back to its original image size
            original_size = info["original_sizes"]
            img = unpad_and_resize_back(img, original_size[0], original_size[1])
        img.save(output_file)


def main_inference(args):
    # set torch dist
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if world_size > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank,
        )

    torch.cuda.set_device(local_rank)
    print("world_size", world_size)
    print("rank", rank)
    print("local_rank", local_rank)

    chosen_datasets = args.chosen_datasets.split(',')
    rank0_print(f"chosen_datasets: {chosen_datasets}")

    # eval_model is built first
    eval_model_cfg = dict(
        type=args.model_name,
        config=args.mllm_config,
        tokenizer_config=args.tokenizer_config,
        diffusion_decoder_path=args.diffusion_decoder_path,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        torch_dtype=args.torch_dtype,
        seed=args.seed
    )
    eval_model = build_eval_model(eval_model_cfg)

    # Then, inference_config is created using the model's method
    inference_config = eval_model.prepare_inference_config(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        llm_cfg_scale=args.llm_cfg_scale,
        diffusion_cfg_scale=args.diffusion_cfg_scale,
    )

    rank0_print(f"temperature: {inference_config.temperature}")
    rank0_print(f"top_k: {inference_config.top_k}")
    rank0_print(f"top_p: {inference_config.top_p}")
    rank0_print(f"llm_cfg_scale: {inference_config.llm_cfg_scale}")
    rank0_print(f"diffusion_cfg_scale: {inference_config.diffusion_cfg_scale}")
    rank0_print(f"image_semantic_temperature: {inference_config.image_semantic_temperature}")
    rank0_print(f"image_semantic_top_k: {inference_config.image_semantic_top_k}")
    rank0_print(f"image_semantic_top_p: {inference_config.image_semantic_top_p}")
    rank0_print(f"image_pixel_temperature: {inference_config.image_pixel_temperature}")
    rank0_print(f"image_pixel_top_k: {inference_config.image_pixel_top_k}")
    rank0_print(f"image_pixel_top_p: {inference_config.image_pixel_top_p}")

    for dataset_name in chosen_datasets:
        # update dataset configs
        update_configs = {}
        if args.resolution_type == "fixed":
            update_configs["ratios"] = [(args.resolution, args.resolution)]

        val_dataset = build_eval_dataset(dataset_name, update_configs)
        rank0_print(
            f"--------------start {dataset_name}, dataset length {len(val_dataset)}, role: {val_dataset.get_role()}--------------")

        for resolution in val_dataset.get_ratios():
            rank0_print(f"--------------resolution: {resolution}--------------")
            inference_config.dataset_name = dataset_name
            inference_config.resolution = resolution
            inference_config.unconditional_prompt = val_dataset.get_unconditional_prompt()

            # set output filename
            ratio_name = val_dataset.get_ratio_name_from_ratio(resolution)
            name = f"{inference_config.dataset_name}_temperature{inference_config.temperature}_topk{inference_config.top_k}_topp{inference_config.top_p}_cfg{inference_config.llm_cfg_scale}"
            result_jsonl_file = f"{name}_{ratio_name}.jsonl"
            result_image_dir = os.path.join(f"{name}_dualvitok", ratio_name)
            result_image_diffusion_dir = os.path.join(f"{name}_diffusion_cfg{inference_config.diffusion_cfg_scale}",
                                                      ratio_name)
            rank0_print(f"result_jsonl_file, {result_jsonl_file}")
            rank0_print(f"result_image_dir, {result_image_dir}")
            rank0_print(f"result_image_diffusion_dir, {result_image_diffusion_dir}")

            select_data_index = [i for i in range(len(val_dataset))][rank::world_size]
            batch_index_list = [select_data_index[i:i + args.batch_size] for i in
                                range(0, len(select_data_index), args.batch_size)]

            llm_outputs = []
            for batch_index in tqdm(batch_index_list, disable=(local_rank != 0)):
                batch_data = [val_dataset.__getitem__(idx) for idx in batch_index]

                output = eval_model.get_one_batch_results(batch_data, inference_config)
                llm_outputs.extend(output["batch_llm_output"])

                # save output images
                save_output_images(output["out_images_tokenizer"], output["batch_llm_output"],
                                   os.path.join(eval_model.output_dir, "generation_eval", result_image_dir))
                                   
                save_output_images(output["out_images_diffusion"], output["batch_llm_output"],
                                   os.path.join(eval_model.output_dir, "generation_eval", result_image_diffusion_dir))

            if world_size > 1:
                torch.distributed.barrier()

            # save llm outputs
            if world_size > 1:
                merged_outputs = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(merged_outputs, llm_outputs)
                merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
                torch.distributed.barrier()
            else:
                merged_outputs = llm_outputs
            if local_rank == 0:
                print(f"VQ decoded images saved in "
                      f"{os.path.abspath(os.path.join(eval_model.output_dir, 'generation_eval', result_image_dir))}")

                print(f"Diffusion decoded images saved in "
                      f"{os.path.abspath(os.path.join(eval_model.output_dir, 'generation_eval', result_image_diffusion_dir))}")

                write_to_jsonl(merged_outputs,
                               os.path.join(eval_model.output_dir, "generation_eval", result_jsonl_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ILLUME")
    parser.add_argument("--mllm_config", type=str, default=None)
    parser.add_argument("--tokenizer_config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--torch_dtype", type=str, default='fp32')
    parser.add_argument("--seed", type=int, default=42)
    #
    parser.add_argument("--diffusion_decoder_path", type=str, default="<diffusion_decoder_model_path>")
    parser.add_argument("--tokenizer_checkpoint", type=str, default="<tokenzier_checkpoint>")
    #
    parser.add_argument("--chosen_datasets", type=str, default=None)  # None
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--llm_cfg_scale", type=float, default=2.0)
    parser.add_argument("--diffusion_cfg_scale", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--resolution_type", type=str, default="fixed_anchors")  # fixed, fixed_anchors
    args = parser.parse_args()

    main_inference(args)
