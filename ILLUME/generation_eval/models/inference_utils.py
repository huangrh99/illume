import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Optional, List, Dict, Tuple, Union, Set

import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_interleaved_text_image(
        full_output_text: str,
        num_levels: int = 2,
        image_placeholder: str = "<image>",
        start_tag: str = "<start_of_image>",
        end_tag: str = "<end_of_image>"
) -> Tuple[str, List[List[List[int]]]]:
    """
    Parses text containing interleaved image token blocks.

    Identifies blocks enclosed by start_tag and end_tag, extracts image tokens
    (<|image_levelX_Y|>) within them, and replaces the blocks with a placeholder
    in the output text.

    Args:
        full_output_text: The raw input string containing text and image blocks.
        num_levels: The expected number of levels for image tokens (e.g., 2).
        image_placeholder: The string to replace image blocks with in the output text.
        start_tag: The exact string marking the beginning of an image block.
        end_tag: The exact string marking the end of an image block.
        eos_token: If provided, this token will be removed from the final text.

    Returns:
        A tuple containing:
        - generated_text (str): The text with image blocks replaced by placeholders.
        - all_image_indices (List[List[List[int]]]): A list where each element
          represents one image. Each image element is a list containing lists
          of token indices for each level.
          Example for 2 images, 2 levels:
          [
            [[level0_indices_img1], [level1_indices_img1]], # Image 1
            [[level0_indices_img2], [level1_indices_img2]]  # Image 2
          ]
    """
    all_image_indices: List[List[List[int]]] = []
    processed_text_parts: List[str] = []
    list_image_token_parts: List[str] = []
    last_end: int = 0

    # Escape start/end tags for regex safety if they contain special characters
    escaped_start_tag = re.escape(start_tag)
    escaped_end_tag = re.escape(end_tag)

    # Pattern to find image blocks: start_tag ... end_tag (non-greedy)
    image_block_pattern = rf'{escaped_start_tag}(.*?){escaped_end_tag}'
    # Pattern to find individual image tokens within a block
    token_pattern = r'<\|image_level(\d+)_(\d+)\|>'

    # Find all image blocks (re.DOTALL allows '.' to match newlines)
    for match in re.finditer(image_block_pattern, full_output_text, re.DOTALL):
        # 1. Add text preceding this image block
        processed_text_parts.append(full_output_text[last_end:match.start()])

        # collect the image token ids.
        list_image_token_parts.append(full_output_text[match.start(): match.end()])

        # 2. Add the placeholder for the image
        processed_text_parts.append(image_placeholder)

        # 3. Process the content *within* the current image block
        image_token_content = match.group(1)  # Content between tags
        parsed_level_indices = {}  # {level: [indices]} for *this* image

        # Find all image tokens within this block
        for token_match in re.finditer(token_pattern, image_token_content):
            try:
                level = int(token_match.group(1))
                index = int(token_match.group(2))
                if level >= num_levels:
                    logging.warning(f"Parsed token level {level} >= num_levels {num_levels}. Ignoring token.")
                    continue
                if level not in parsed_level_indices:
                    parsed_level_indices[level] = []
                parsed_level_indices[level].append(index)
            except (ValueError, IndexError):
                logging.warning(f"Could not parse token: {token_match.group(0)}")
                continue  # Skip malformed tokens

        # Structure the indices for the current image based on expected levels
        current_image_indices = []
        logging.debug(f"Processing Image Block. Found levels: {parsed_level_indices.keys()}")
        for level in range(num_levels):
            # Get indices for the level, default to empty list if level not found
            indices = parsed_level_indices.get(level, [])
            # Optional: Sort indices if order isn't guaranteed (usually is by finditer)
            # indices.sort()
            current_image_indices.append(indices)
            logging.debug(f"  Level {level} indices count: {len(indices)}")

        all_image_indices.append(current_image_indices)
        logging.info(f"Parsed Image {len(all_image_indices)}: Found indices for {len(current_image_indices)} levels.")

        # 4. Update position for the next iteration
        last_end = match.end()

    # Add any remaining text after the last image block
    processed_text_parts.append(full_output_text[last_end:])

    # Join the text parts to form the final generated text
    generated_text = "".join(processed_text_parts)

    return generated_text, all_image_indices, list_image_token_parts


def calculate_image_token_num(h, w, downsample_rate_per_level=[28, 16]):
    # Assuming RESOLUTION_MAPPING is accessible or hardcoded if needed
    # For simplicity, let's assume direct calculation based on downsampling
    # Replace with actual RESOLUTION_MAPPING logic if necessary
    # Example: w1, h1 = RESOLUTION_MAPPING.get((w, h), (w, h)) # Get from mapping
    w1, h1 = w, h  # Placeholder if mapping not available/needed here
    w1, h1 = w1 // downsample_rate_per_level[0], h1 // downsample_rate_per_level[0]
    semantic_token_num = w1 * h1

    w2, h2 = w // downsample_rate_per_level[1], h // downsample_rate_per_level[1]
    pixel_token_num = w2 * h2
    logging.info(f"Calculated token nums: semantic={semantic_token_num}, pixel={pixel_token_num} for target ({h},{w})")
    # Estimate max_token_length (adjust based on special tokens in your format)
    max_token_length = (h1 * (w1 + 1) + 2) + (h2 * (w2 + 1) + 2) + 2 + 2 + 1 + 1 + 50  # Add buffer
    return [semantic_token_num, pixel_token_num], max_token_length, h1, w1, h2, w2


def check_image_token_num(image_embed_inds, token_nums=[81, 256], identifier=""):
    image_embed_inds_out = []
    if len(image_embed_inds) != len(token_nums):
        logging.error(
            f"{identifier} Mismatch between number of image token levels ({len(image_embed_inds)}) and expected token_nums ({len(token_nums)})")
        # Handle error appropriately - maybe return None or raise exception
        return None

    for level, (embed_inds, token_num) in enumerate(zip(image_embed_inds, token_nums)):
        if not len(embed_inds) == token_num:
            logging.warning(
                f"{identifier} Level {level} embed_inds length {len(embed_inds)} not equal to expected {token_num}! Padding/truncating.")
            if len(embed_inds) > token_num:
                embed_inds = embed_inds[:token_num]
            elif len(embed_inds) == 0:
                # Handle empty case - perhaps fill with a default token?
                logging.warning(f"{identifier} Level {level} embed_inds is empty. Filling with zeros.")
                embed_inds = [0] * token_num  # Or a placeholder token ID
            else:
                # Pad with the last token ID
                embed_inds.extend([embed_inds[-1]] * (token_num - len(embed_inds)))
        image_embed_inds_out.append(embed_inds)
    return image_embed_inds_out


class CFGLogits(LogitsProcessor):
    ## refer to https://github.com/huggingface/transformers/issues/24536

    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `uncond` branch. Finally, according to CFG Rescale, the reweighted logits are interpolated back with weight
    `rescale_factor` the conditional ones to smooth the effect and increase output quality.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        uncond (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
    """

    def __init__(self, guidance_scale, uncond, model, images=None, image_sizes=None, rescale_factor=1.0):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.images = images
        self.image_sizes = image_sizes
        self.model = model
        self.out = None
        self.rescale_factor = rescale_factor

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        if self.out is None:
            self.out = self.model(self.uncond.to(self.model.device), images=self.images, image_sizes=self.image_sizes,
                                  use_cache=True)
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return out


class DualVQImageTokenProcessor(LogitsProcessor):
    def __init__(self, level0_range, level1_range, num_level0_rows, num_level0_tokens,
                 num_level1_rows, num_level1_tokens, special_tokens):
        self.level0_range = level0_range  # (min_id, max_id) for level0 tokens
        self.level1_range = level1_range  # (min_id, max_id) for level1 tokens
        self.num_level0_rows = num_level0_rows  # 9 rows for level0
        self.num_level0_tokens = num_level0_tokens  # 9 tokens per row
        self.num_level1_rows = num_level1_rows  # 16 rows for level1
        self.num_level1_tokens = num_level1_tokens  # 16 tokens per row
        self.special_tokens = special_tokens  # Dictionary of special tokens

        self.current_level = None  # "level0" or "level1"
        self.tokens_in_row = 0  # Count of tokens in the current row
        self.rows_in_level = 0  # Count of rows in the current level
        self.generating_image = False  # True if inside <start_of_image> ... <end_of_image>

    def __call__(self, input_ids, scores):
        try:
            last_token = input_ids[0, -1].item()
        except:
            # print("last_token error", input_ids.shape)
            # TODO FIXME
            # Ensure the first token is <start_of_image> for image generation tasks
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.special_tokens["start_of_image"]] = 0  # 只允许生成 <start_of_image>
            scores += mask
            return scores

        # --- State transition logic based on last token ---

        # Just finished generating <start_of_image>
        if last_token == self.special_tokens["start_of_image"]:
            self.generating_image = True
            self.current_level = None  # Reset level state
            self.tokens_in_row = 0
            self.rows_in_level = 0
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.special_tokens["start_of_level0"]] = 0  # 只允许生成 <start_of_level0>
            scores += mask
            return scores

        # Just finished generating <start_of_level0> (Only if 0 in selected_levels)
        if last_token == self.special_tokens["start_of_level0"]:
            self.current_level = "level0"
            self.tokens_in_row = 0
            self.rows_in_level = 0
            # Allow level 0 tokens next (handled below in level 0 processing logic)
            # No immediate return needed here, fall through to level 0 processing.

        # Just finished generating <end_of_level0>
        if last_token == self.special_tokens["end_of_level0"]:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.special_tokens["start_of_level1"]] = 0  # 只允许生成 <start_of_level1>
            scores += mask
            return scores

        # Just finished generating <start_of_level1> (Only if 1 in selected_levels)
        if last_token == self.special_tokens["start_of_level1"]:
            self.current_level = "level1"
            self.tokens_in_row = 0
            self.rows_in_level = 0
            # Allow level 1 tokens next (handled below in level 1 processing logic)
            # No immediate return needed here, fall through to level 1 processing.

        # Just finished generating <end_of_level1>
        if last_token == self.special_tokens["end_of_level1"]:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.special_tokens["end_of_image"]] = 0  # Force <end_of_image>
            scores += mask
            return scores

        # Just finished generating <end_of_image>
        if last_token == self.special_tokens["end_of_image"]:
            self.generating_image = False
            self.current_level = None
            self.tokens_in_row = 0
            self.rows_in_level = 0
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.special_tokens["end_of_text"]] = 0
            scores += mask
            return scores

        # processing Level 0 token limit
        if self.current_level == "level0":
            if self.tokens_in_row == self.num_level0_tokens:  # 9 tokens reached, enforce <end_of_line>
                mask = torch.full_like(scores, float("-inf"))
                mask[:, self.special_tokens["end_of_line"]] = 0
                scores += mask
                self.tokens_in_row = 0
                self.rows_in_level += 1
                return scores

            if self.rows_in_level == self.num_level0_rows:  # 9 rows reached, enforce <end_of_level0>
                mask = torch.full_like(scores, float("-inf"))
                mask[:, self.special_tokens["end_of_level0"]] = 0
                scores += mask
                return scores

            # Only allow generating level0 tokens
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.level0_range[0]:self.level0_range[1]] = 0
            scores += mask
            self.tokens_in_row += 1
            return scores

        # processing Level 1 token limit
        if self.current_level == "level1":
            if self.tokens_in_row == self.num_level1_tokens:  # 16 tokens reached, enforce <end_of_line>
                mask = torch.full_like(scores, float("-inf"))
                mask[:, self.special_tokens["end_of_line"]] = 0
                scores += mask
                self.tokens_in_row = 0
                self.rows_in_level += 1
                return scores

            if self.rows_in_level == self.num_level1_rows:  # 16 rows reached, enforce <end_of_level1>
                mask = torch.full_like(scores, float("-inf"))
                mask[:, self.special_tokens["end_of_level1"]] = 0
                scores += mask
                return scores

            # Only allow generating level1 tokens
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.level1_range[0]:self.level1_range[1]] = 0
            scores += mask
            self.tokens_in_row += 1
            return scores

        # If not generating an image or between states not handled above, return original scores
        # (e.g., generating text before <start_of_image> or after <end_of_text>)
        return scores


class DynamicSamplingProcessor(LogitsProcessor):
    def __init__(self,
                 special_tokens,
                 default_temp=1.0, level0_temp=1.0, level1_temp=2.0,
                 default_top_k=2048, level0_top_k=2048, level1_top_k=2048 * 3,
                 default_top_p=0.8, level0_top_p=0.8, level1_top_p=1.0,
                 ):
        """
        Custom LogitsProcessor to dynamically adjust temperature, top_k, top_p based on the current generated token ID.
        """

        self.start_of_level0_token_id = special_tokens["start_of_level0"]
        self.end_of_level0_token_id = special_tokens["end_of_level0"]
        self.start_of_level1_token_id = special_tokens["start_of_level1"]
        self.end_of_level1_token_id = special_tokens["end_of_level1"]

        self.default_temp = default_temp
        self.default_top_k = default_top_k
        self.default_top_p = default_top_p

        self.level0_temp = level0_temp
        self.level0_top_k = level0_top_k
        self.level0_top_p = level0_top_p

        self.level1_temp = level1_temp
        self.level1_top_k = level1_top_k
        self.level1_top_p = level1_top_p

        self.in_level0_mode = False  # Whether in Level 0
        self.in_level1_mode = False  # Whether in Level 1

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Dynamically adjust top_k based on input_ids.
        :param input_ids: Shape [batch_size, seq_len], current generated token sequence
        :param scores: Shape [batch_size, vocab_size], current logits scores
        :return: Processed logits scores
        """
        batch_size = input_ids.shape[0]  # Although code uses input_ids[0, -1], get batch_size for potential future use

        try:
            # Assuming batch_size is 1 for state tracking simplicity here.
            # If batch_size > 1, this state logic needs to be batched.
            last_token = input_ids[0, -1].item()  # Get the last token of the current batch (assuming bs=1)

            # Update state
            if last_token == self.start_of_level0_token_id:
                self.in_level0_mode = True
                self.in_level1_mode = False
            elif last_token == self.end_of_level0_token_id:
                self.in_level0_mode = False
                # Don't change in_level1_mode here, wait for start_of_level1

            if last_token == self.start_of_level1_token_id:
                self.in_level0_mode = False
                self.in_level1_mode = True
            elif last_token == self.end_of_level1_token_id:
                self.in_level1_mode = False
                # Don't change in_level0_mode here

        except IndexError:
            # multimodal setting when predicting the first token, there is no last token, input_ids shape might be [bs, 0] or similar
            # Or during initial prompt processing. Reset modes.
            self.in_level0_mode = False
            self.in_level1_mode = False

        # Apply sampling based on the current mode
        if self.in_level0_mode:
            scores = self._apply_sampling(scores, self.level0_temp, self.level0_top_k, self.level0_top_p)
        elif self.in_level1_mode:
            scores = self._apply_sampling(scores, self.level1_temp, self.level1_top_k, self.level1_top_p)
        else:
            # Apply default sampling when not inside level0 or level1 image generation
            scores = self._apply_sampling(scores, self.default_temp, self.default_top_k, self.default_top_p)

        return scores

    def _apply_sampling(self, scores, temp, top_k, top_p):
        """ Apply top-k, top-p, and temperature """
        # Apply temperature
        if temp > 0.0:  # Avoid division by 0
            scores = scores / temp

        # Top-K filtering
        if top_k > 0:
            # Ensure top_k is not larger than vocab size
            _top_k = min(top_k, scores.size(-1))
            # Remove all tokens with a probability less than the k-th token's probability
            top_k_values, _ = torch.topk(scores, _top_k)
            # Get the score of the k-th token (last one in the top_k_values tensor)
            kth_score = top_k_values[:, -1].unsqueeze(-1)
            # Set scores below the k-th score to -inf
            scores[scores < kth_score] = -float("Inf")

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (nucleus)
            # We keep tokens whose cumulative probability is <= top_p
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the mask to the right to keep the first token above the threshold
            # If all tokens are removed (e.g., due to numerical instability), keep at least the top token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False  # Always keep the highest probability token

            # Scatter the boolean mask back to the original indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            # Apply the mask by setting scores to -inf
            scores[indices_to_remove] = -float("Inf")

        return scores


class InterleavedLogitsProcessor(LogitsProcessor):
    """
    Combines CFG, Dual VQ Image Token Structure Enforcement, and Dynamic Sampling
    for interleaved text and image generation.
    """

    def __init__(self,
                 # CFG parameters
                 guidance_scale, uncond, model,
                 # DualVQ parameters
                 level0_range, level1_range, num_level0_rows, num_level0_tokens,
                 num_level1_rows, num_level1_tokens, special_tokens,
                 *,
                 # Dynamic Sampling parameters
                 default_temp=1.0, level0_temp=1.0, level1_temp=2.0,
                 default_top_k=2048, level0_top_k=2048, level1_top_k=2048 * 3,
                 default_top_p=0.8, level0_top_p=0.8, level1_top_p=1.0,
                 # General
                 images=None, image_sizes=None
                 ):

        # --- CFG ---
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.images = images
        self.image_sizes = image_sizes
        self.model = model
        self.out = None

        # --- DualVQ ---
        self.level0_range = level0_range
        self.level1_range = level1_range
        self.num_level0_rows = num_level0_rows
        self.num_level0_tokens = num_level0_tokens
        self.num_level1_rows = num_level1_rows
        self.num_level1_tokens = num_level1_tokens
        self.special_tokens = special_tokens

        # DualVQ State
        self.generating_image = False
        self.current_level = None
        self.tokens_in_row = 0
        self.rows_in_level = 0

        # --- Dynamic Sampling ---
        self.start_of_level0_token_id = special_tokens["start_of_level0"]
        self.end_of_level0_token_id = special_tokens["end_of_level0"]
        self.start_of_level1_token_id = special_tokens["start_of_level1"]
        self.end_of_level1_token_id = special_tokens["end_of_level1"]
        self.start_of_image_token_id = special_tokens["start_of_image"]
        self.end_of_image_token_id = special_tokens["end_of_image"]

        self.default_temp = default_temp
        self.default_top_k = default_top_k
        self.default_top_p = default_top_p
        self.level0_temp = level0_temp
        self.level0_top_k = level0_top_k
        self.level0_top_p = level0_top_p
        self.level1_temp = level1_temp
        self.level1_top_k = level1_top_k
        self.level1_top_p = level1_top_p

        # Dynamic Sampling State
        self.in_level0_mode = False
        self.in_level1_mode = False

        # --- Validation ---
        if not self.special_tokens:
            raise ValueError("special_tokens dictionary cannot be empty.")
        # *** Updated required keys ***
        required_keys = ["start_of_image", "end_of_image", "start_of_level0",
                         "end_of_level0", "start_of_level1", "end_of_level1",
                         "end_of_line", "end_of_text"]
        for key in required_keys:
            if key not in self.special_tokens:
                raise ValueError(f"Missing required key in special_tokens: {key}")

    def _apply_cfg(self, input_ids, scores):
        """Applies Classifier-Free Guidance."""
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        if self.out is None:
            self.out = self.model(torch.cat([self.uncond, input_ids[:, -1:]], dim=-1).to(self.model.device), 
                                  images=self.images, image_sizes=self.image_sizes, use_cache=True)
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return out

    def _apply_sampling(self, scores, temp, top_k, top_p):
        """ Apply top-k, top-p, and temperature """
        scores = scores / temp

        # Top-K filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(scores, min(top_k, scores.size(-1)))
            scores[scores < top_k_values[:, -1].unsqueeze(-1)] = -float("Inf")

        # Top-P filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Only keep tokens with cumulative probability <= top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores[indices_to_remove] = -float("Inf")

        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # --- Step 0: Get last token and Vocab Size ---
        last_token = None
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()  # Assuming batch size 1

        # --- Step 1: Update State & Apply Constraints ---
        # State updates based on the *last generated* token
        if last_token == self.start_of_image_token_id:
            self.generating_image = True
            self.current_level = None
            self.tokens_in_row = 0
            self.rows_in_level = 0
            self.in_level0_mode = False
            self.in_level1_mode = False
        elif last_token == self.start_of_level0_token_id:
            self.current_level = "level0"
            self.tokens_in_row = 0
            self.rows_in_level = 0
            self.in_level0_mode = True
            self.in_level1_mode = False
        elif last_token == self.end_of_level0_token_id:
            self.current_level = None
            self.in_level0_mode = False
        elif last_token == self.start_of_level1_token_id:
            self.current_level = "level1"
            self.tokens_in_row = 0
            self.rows_in_level = 0
            self.in_level0_mode = False
            self.in_level1_mode = True
        elif last_token == self.end_of_level1_token_id:
            self.current_level = None
            self.in_level1_mode = False
        elif last_token == self.end_of_image_token_id:
            self.generating_image = False
            self.current_level = None
            self.tokens_in_row = 0
            self.rows_in_level = 0
            self.in_level0_mode = False
            self.in_level1_mode = False
        elif last_token == self.special_tokens["end_of_line"] and self.generating_image:
            self.tokens_in_row = 0
            self.rows_in_level += 1
        elif self.generating_image and self.current_level is not None:
            if (self.current_level == "level0" and self.level0_range[0] <= last_token < self.level0_range[1]) or \
                    (self.current_level == "level1" and self.level1_range[0] <= last_token < self.level1_range[1]):
                self.tokens_in_row += 1

        # --- Step 2: Apply CFG ---
        if self.generating_image:
            scores = self._apply_cfg(input_ids, scores)
        else:
            if self.out:
                self.out = None

        # Apply constraints based on the *current* state (determining the *next* token)
        mask = torch.zeros_like(scores, dtype=torch.bool)  # True means ALLOWED

        if self.generating_image:
            # --- Image Generation Masking ---
            if self.current_level == "level0":
                if self.rows_in_level == self.num_level0_rows:
                    mask[:, self.special_tokens["end_of_level0"]] = True
                elif self.tokens_in_row == self.num_level0_tokens:
                    mask[:, self.special_tokens["end_of_line"]] = True
                else:
                    mask[:, self.level0_range[0]:self.level0_range[1]] = True
            elif self.current_level == "level1":
                if self.rows_in_level == self.num_level1_rows:
                    mask[:, self.special_tokens["end_of_level1"]] = True
                elif self.tokens_in_row == self.num_level1_tokens:
                    mask[:, self.special_tokens["end_of_line"]] = True
                else:
                    mask[:, self.level1_range[0]:self.level1_range[1]] = True
            else:  # Between structure tokens
                if last_token == self.start_of_image_token_id:
                    mask[:, self.special_tokens["start_of_level0"]] = True
                elif last_token == self.end_of_level0_token_id:
                    mask[:, self.special_tokens["start_of_level1"]] = True
                elif last_token == self.end_of_level1_token_id:
                    mask[:, self.special_tokens["end_of_image"]] = True
                elif last_token is None and input_ids.shape[1] == 0:  # Very first token is image?
                    mask[:, self.start_of_image_token_id] = True
                else:  # Allow relevant structural tokens if needed
                    mask[:, self.special_tokens["start_of_level0"]] = True
                    mask[:, self.special_tokens["start_of_level1"]] = True
                    mask[:, self.special_tokens["end_of_image"]] = True

        else:
            # Allow *all* tokens by default...
            mask[:, :] = True
            # ...then specifically *disallow* image content and intermediate structure tokens
            mask[:, self.level0_range[0]:self.level0_range[1]] = False
            mask[:, self.level1_range[0]:self.level1_range[1]] = False
            mask[:, self.special_tokens["start_of_level0"]] = False
            mask[:, self.special_tokens["end_of_level0"]] = False
            mask[:, self.special_tokens["start_of_level1"]] = False
            mask[:, self.special_tokens["end_of_level1"]] = False
            mask[:, self.special_tokens["end_of_line"]] = False  # EOL only allowed within image context

            # Ensure the specific allowed tokens for text phase are indeed allowed
            # (This overrides any potential disallowing above if IDs overlap, e.g., if EOS was in image range)
            mask[:, self.special_tokens["end_of_text"]] = True
            mask[:, self.special_tokens["start_of_image"]] = True

        # Apply the mask
        scores[~mask] = -float("Inf")

        # Handle edge case: If all tokens are masked
        if not torch.any(scores > -float("Inf"), dim=-1).all():
            print("WARN: All tokens masked, allowing EOS.")
            # Allow EOS and potentially other safe tokens if needed
            scores[:] = -float("Inf")  # Reset all to -inf first
            scores[:, self.special_tokens["end_of_text"]] = 0

        # --- Step 3: Apply Dynamic Sampling ---
        current_temp, current_top_k, current_top_p = self.default_temp, self.default_top_k, self.default_top_p
        if self.in_level0_mode:
            current_temp, current_top_k, current_top_p = self.level0_temp, self.level0_top_k, self.level0_top_p
        elif self.in_level1_mode:
            current_temp, current_top_k, current_top_p = self.level1_temp, self.level1_top_k, self.level1_top_p

        scores = self._apply_sampling(scores, current_temp, current_top_k, current_top_p)

        return scores


def replace_placeholder_with_list(
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,  # Allow ints or floats in the list
        placeholder_value: Union[int, float]  # Placeholder can be int or float
) -> torch.Tensor:
    if tensor_a.dim() != 1:
        raise ValueError("Input tensor_a must be 1-dimensional.")

    indices = torch.where(tensor_a == placeholder_value)[0]

    if len(indices) == 0:
        # Placeholder not found, return the original tensor
        print(
            f"Warning: Placeholder value {placeholder_value} not found in the tensor. Returning original tensor.")
        return tensor_a

    # Get the index of the *first* occurrence
    idx = indices[0].item()

    result_tensor = torch.cat((tensor_a[:idx], tensor_b.to(tensor_a), tensor_a[idx + 1:]), dim=0)
    return result_tensor
