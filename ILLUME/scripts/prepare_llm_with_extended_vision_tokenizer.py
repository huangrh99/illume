import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tokenizers


def _add_tokens(tokenizer, added_list):
    tokenizer.add_tokens([tokenizers.AddedToken(item, single_word=True) for item in added_list])


def extend_qwen2_5(model_path, output_model_path, add_token_num_per_levels):
    levels = len(add_token_num_per_levels)
    add_token_name = "<|image_level{}_{}|>"

    # save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left", use_fast=True)

    # show original vocab and embedding table
    print(f"original tokenizer.vocab_size: {tokenizer.vocab_size}\n")
    print(f"original len(tokenizer): {len(tokenizer)}\n")

    ############################
    added_tokens_num = tokenizer.add_tokens(["<start_of_image>", "<end_of_image>", "<end_of_line>"])
    for level in range(levels):
        added_tokens_num += tokenizer.add_tokens(["<start_of_level{}>".format(level), "<end_of_level{}>".format(level)])

    # add height/width indicator, <height_i> means height equals to (i * anyres_indicator_base)
    anyres_indicator_base = 64
    anyres_max_resolution = 1024
    aspect_ratio_max_num = anyres_max_resolution // anyres_indicator_base
    added_tokens_num += tokenizer.add_tokens(
        ["<height_{}>".format((i + 1) * anyres_indicator_base) for i in range(aspect_ratio_max_num)])
    added_tokens_num += tokenizer.add_tokens(
        ["<width_{}>".format((i + 1) * anyres_indicator_base) for i in range(aspect_ratio_max_num)])

    for level, add_token_num_per_level in enumerate(add_token_num_per_levels):
        added_tokens_num += tokenizer.add_tokens(
            [add_token_name.format(level, i) for i in range(add_token_num_per_level)])

    print(f"modified tokenizer.vocab_size: {tokenizer.vocab_size}\n")
    print(f"modified len(tokenizer): {len(tokenizer)}\n")

    text_example = "This is an image:" + "<height_256><width_384><start_of_image><start_of_level0>" + "".join(
        [add_token_name.format(0, i) for i in range(3)]) + "<end_of_level1><end_of_image>"
    encoded_text = tokenizer.encode(text_example)

    print(f"Original text: {text_example}")
    print(f"Encoded text: {encoded_text}")
    decoded_text = tokenizer.decode(encoded_text)
    print(f"Decoded text: {decoded_text}")

    print("added_tokens_num", added_tokens_num)

    tokenizer.save_pretrained(output_model_path)
    print(f"save tokenizer to {output_model_path}")
    
    # save model
    dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    print(f"original embedding table shape:{model.get_input_embeddings().weight.size(0)}\n")
    print(f"original model.get_output_embeddings().weight.shape: {model.get_output_embeddings().weight.shape}")

    text_vocab_size, text_dim = model.get_input_embeddings().weight.shape

    new_num_tokens = added_tokens_num + text_vocab_size
    print("new_num_tokens", new_num_tokens)
    model.resize_token_embeddings(new_num_tokens)

    print(f"modified embedding table shape:{model.get_input_embeddings().weight.size(0)}\n")
    print(f"modified model.get_output_embeddings().weight.shape: {model.get_output_embeddings().weight.shape}")

    print("save model")
    model.save_pretrained(output_model_path)
    print(f"save model to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--semantic_codebook_size", type=int, default=32768)
    parser.add_argument("--pixel_codebook_size", type=int, default=32768 * 3)
    parser.add_argument("--output_model_path", type=str,
                        default="checkpoints/Qwen2.5-0.5B-Instruct-with-vision-tokenizer-32k-96k-level2")
    args = parser.parse_args()

    add_token_num_per_levels = [args.semantic_codebook_size,
                                args.pixel_codebook_size]  # semantic and pixel codebook size of dualvitok
    extend_qwen2_5(args.model_path, args.output_model_path, add_token_num_per_levels)

# python ILLUME_plus/ILLUME/scripts/prepare_llm_with_extended_vision_tokenizer.py --model_path ./Qwen2.5-0.5B-Instruct/ --output_model_path ./Qwen2.5-0.5B-Instruct-with-vision-tokenizer-32k-96k-level2
# python ILLUME_plus/ILLUME/scripts/prepare_llm_with_extended_vision_tokenizer.py --model_path ./Qwen2.5-7B-Instruct/ --output_model_path ./Qwen2.5-7B-Instruct-with-vision-tokenizer-32k-96k-level2
