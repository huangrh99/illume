import os
import json
from PIL import Image, ImageOps


ROLE_TEMPLATES = {
    "text2image": [
        'Generate an image of {resolution_tag}, the content of the image is <prompt>\n'
    ],
    "editing": [
        "<image>\nPlease edit the image according to the instruction: <prompt>.\n"
    ],
    "random2image": [
        'Generate a random image of {resolution_tag}\n'
    ],
    "image_reconstruction": [
        "<image>\nReconstruct the image according to the given image\n"
    ]
}


def read_data_file(file, load_with_bytes=True):
    if not os.path.exists(file):
        print(f"{file} not exist!!!")
        return []
    if file.endswith('.json'):
        with open(file, 'r') as f:
            list_data_dict = json.load(f)
    elif file.endswith('.jsonl'):
        if load_with_bytes:
            with open(file, "rb") as fr:
                return [item for item in fr]
        else:
            with open(file, 'r') as f:
                list_data_dict = [json.loads(l.strip('\n')) for l in f.readlines()]
    else:
        raise RuntimeError(f"Unrecoginized file: {file}")
    return list_data_dict


def read_from_jsonl(file):
    with open(file, 'r') as f:
        infos = [json.loads(l.strip('\n')) for l in f.readlines()]
    return infos


def write_to_jsonl(results, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w+', encoding='utf-8') as f_w:
        for tmp in results:
            f_w.write(json.dumps(tmp, ensure_ascii=False) + '\n')


def return_all_files_in_dir(dir):
    filelist = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            filelist.append(os.path.join(root, file))
    return filelist


def count_lines_in_jsonl_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def encode_image_token_into_code(image_embed_inds, add_token_name="<|image_level{}_{}|>"):
    '''
    Args:
        image_embed_inds: 3D list, vision token ids for each tokenizer level
        add_token_name: tag name for vision tokens
    Returns:
        image_token_return: str
    '''

    image_token_name_list = []
    for level, image_embed_ind in enumerate(image_embed_inds):
        image_token_name = []
        for row in image_embed_ind:
            image_token_name.append([add_token_name.format(level, ind) for ind in row])

        image_token_name_list.append("<start_of_level{}>".format(level))
        for row in image_token_name:
            row.append("<end_of_line>")

        for row in image_token_name:
            image_token_name_list.extend(row)

        image_token_name_list.append("<end_of_level{}>".format(level))

    image_token_return = "".join(image_token_name_list)
    image_token_return = "<start_of_image>" + image_token_return + "<end_of_image>"
    return image_token_return


def center_crop_and_resize(img, output_size=(256, 256)):
    target_h, target_w = output_size
    img_w, img_h = img.size

    scale_w, scale_h = img_w / target_w, img_h / target_h
    if scale_h > scale_w:
        new_w, new_h = target_w, int(target_w / img_w * img_h)
    else:
        new_w, new_h = int(target_h / img_h * img_w), target_h

    # Resize 图片，保持长宽比
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 计算中心裁剪区域
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # 裁剪多余部分
    img = img.crop((left, top, right, bottom))

    return img


def resize_with_padding(img, output_size=(256, 256), fill_color=(255, 255, 255)):
    target_height, target_width = output_size

    # Step 1: Resize with aspect ratio preserved
    original_width, original_height = img.size
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    resized_image = img.resize(new_size, Image.LANCZOS)

    # Step 2: Add padding to reach target size
    delta_w = target_width - new_size[0]
    delta_h = target_height - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    padded_image = ImageOps.expand(resized_image, padding, fill=fill_color)

    return padded_image


def unpad_and_resize_back(padded_image, original_width, original_height):
    """
    Revert the padded+resized image back to original size.

    Args:
        padded_image (PIL.Image): Image after padding.
        original_width (int): Original image width before resize & pad.
        original_height (int): Original image height before resize & pad.

    Returns:
        PIL.Image: Image resized back to original resolution.
    """
    # Compute the scale factor used during the first resize
    target_width, target_height = padded_image.size
    ratio = min(target_width / original_width, target_height / original_height)
    resized_w = int(original_width * ratio)
    resized_h = int(original_height * ratio)

    # Compute cropping box on padded image
    left = (target_width - resized_w) // 2
    upper = (target_height - resized_h) // 2
    right = left + resized_w
    lower = upper + resized_h

    # Crop out the resized region (before padding)
    cropped_image = padded_image.crop((left, upper, right, lower))

    # Resize back to original resolution
    recovered_image = cropped_image.resize((original_width, original_height), Image.LANCZOS)
    return recovered_image


