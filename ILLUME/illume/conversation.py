import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image
import re


def replace_last(original_string, target, replacement):
    last_index = original_string.rfind(target)

    if last_index == -1:
        return original_string

    before = original_string[:last_index]
    after = original_string[last_index + len(target):]

    return before + replacement + after


def replace_sequentially_regex(original_string, target, content_list):
    # 创建 content_list 的迭代器
    replacements_iter = iter(content_list)

    def get_replacement(match):
        try:
            return str(next(replacements_iter))  # 确保替换内容是字符串
        except StopIteration:
            return match.group(0)

    new_string = re.sub(re.escape(target), get_replacement, original_string)
    return new_string

def replace_ratio_tags_in_text(text):
    # The pattern remains the same, capturing the two numbers
    pattern = r"<height_(\d+)><width_(\d+)>"

    # Define a helper function that will be called for each match
    # This function receives the match object as input
    def _replacer_function(match, resolution_indicator=64):
        try:
            # Extract captured groups (height and width numbers as strings)
            h_tag_str = match.group(1)
            w_tag_str = match.group(2)

            # Convert to integers
            h_tag = int(h_tag_str)
            w_tag = int(w_tag_str)

            # Calculate final dimensions
            h = h_tag * resolution_indicator
            w = w_tag * resolution_indicator

            # Return the replacement string for this specific match
            return f'{h}x{w}'
        except (ValueError, IndexError):
            # In case of unexpected errors (e.g., regex issue, conversion error)
            # return the original matched text to avoid breaking the string
            # match.group(0) returns the entire substring that matched the pattern
            print(f"Warning: Could not process tag '{match.group(0)}'. Keeping original.")
            return match.group(0)

    # Use re.sub() to find all matches of the pattern in the text
    # and replace each match using the result of _replacer_function
    processed_text = re.sub(pattern, _replacer_function, text)

    return processed_text


def process_think_answer_tag_for_gradio(input_text):
    # Replace <think></think> and <answer></answer> with collapsible sections
    output_text = input_text.replace(
        "<think>",
        '<div class="collapsible"><button class="collapsible-btn">Think</button><div class="content">').replace(
        "</think>", "</div></div>"
    )
    output_text = output_text.replace(
        "<answer>",
        '<div class="collapsible"><button class="collapsible-btn">Answer</button><div class="content">').replace(
        "</answer>", "</div></div>"
    )
    return output_text

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    GLM4 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            # init_role, init_msg = messages[0].copy()
            # init_msg = init_msg[0].replace("<image>", "").strip()
            # if 'mmtag' in self.version:
            #     messages[0] = (init_role, init_msg)
            #     messages.insert(0, (self.roles[0], "<Image><image></Image>"))
            #     messages.insert(1, (self.roles[1], "Received."))
            # else:
            #     messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if role == self.roles[0]:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += role + message + self.sep
                    else:
                        if type(message) is tuple:
                            # process the discrete image token in the output.
                            message, _, image_token_lists = message
                            message = replace_sequentially_regex(message, '<image>', image_token_lists)
                        ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        elif self.sep_style == SeparatorStyle.GLM4:
            role = ("<|user|>", "<|assistant|>")
            ret = self.system + role[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += self.sep + message + role[(i + 1) % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        if isinstance(self.messages, tuple):
            self.messages += ([role, message],)
        else:
            self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()

                msg = replace_ratio_tags_in_text(msg)
                ret.append([msg, None])
            else:
                if type(msg) is tuple:
                    msg, image, _ = msg
                    if not isinstance(image, list):
                        image = [image]

                    image_str_list = []
                    for img_idx, img in enumerate(image):
                        img_b64_str = self.process_image(
                            img, "Default", return_pil=False,
                            image_format='JPEG')
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="generated image {img_idx}" />'
                        image_str_list.append(img_str)
                    msg = replace_sequentially_regex(msg, '<image>', image_str_list)

                if msg and 'think' in self.version:
                    msg = process_think_answer_tag_for_gradio(msg)
                ret[-1][-1] = msg

        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
         "Renewable energy sources are those that can be replenished naturally in a relatively "
         "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
         "Non-renewable energy sources, on the other hand, are finite and will eventually be "
         "depleted, such as coal, oil, and natural gas. Here are some key differences between "
         "renewable and non-renewable energy sources:\n"
         "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
         "energy sources are finite and will eventually run out.\n"
         "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
         "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
         "and other negative effects.\n"
         "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
         "have lower operational costs than non-renewable sources.\n"
         "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
         "locations than non-renewable sources.\n"
         "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
         "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
         "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
         "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llama3 = Conversation(
    system="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

conv_llama3_without_system = Conversation(
    system="",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3_without_system",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

conv_llama3_base = Conversation(
    system="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3_base",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

conv_llama3_expand = Conversation(
    system="""[BOS]SYSTEM\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("[BOS]USER:\n", "[BOS]ASSISTANT:\n"),
    version="llama3_expand",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="[EOT]",
)

conv_llama3_expandv2 = Conversation(
    system="""[BOS]SYSTEM\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("[BOS]USER:\n", "[BOS]ASSISTANT:\n"),
    version="llama3_expand",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="[unused0]",  # use different eos token.
)

conv_llama3_expandV2 = Conversation(
    system="""[BOS]SYSTEM\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("[BOS]USER:\n", "[BOS]ASSISTANT:\n"),
    version="llama3_expand",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="[unused0]",
)

conv_qwen2 = Conversation(
    system='<|im_start|>system\nYou are a helpful assistant.',
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>\n",
)

conv_qwen2_think = Conversation(
    system='<|im_start|>system\nYou are a helpful assistant.'
           "You will first thinks about the reasoning process in the mind and then provides the user with the answer. "
           "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
           "<think> reasoning process here </think><answer> answer here </answer>",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen2_think",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>\n",
)

qwen2_image_gen_with_think = Conversation(
    system='<|im_start|>system\nYou are a helpful assistant.'
           "You will first thinks about the reasoning process in the mind and then provides the user with the answer. "
           "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
           "<think> reasoning process here </think><answer> answer here </answer>. "
           "If an image needs to be generated inside <think>, generate it at resolution <height_4><width_4>. "
           "In <answer>, generate any requested images at the resolution specified by the user.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen2_think",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>\n",
)

conv_glm4 = Conversation(
    system='[gMASK]<sop>',
    roles=("<|user|>\n", "<|assistant|>"),
    version="glm4",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.GLM4,
    sep="\n",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llama3": conv_llama3,
    "llama3_without_system": conv_llama3_without_system,
    "llama3_expand": conv_llama3_expand,
    "llama3_expand_v2": conv_llama3_expandv2,
    "llama3_base": conv_llama3_base,

    "mpt": conv_mpt,
    "qwen2": conv_qwen2,
    "qwen2_think": conv_qwen2_think,
    "qwen2_image_gen_with_think": conv_qwen2_think,
    "glm4": conv_glm4,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
