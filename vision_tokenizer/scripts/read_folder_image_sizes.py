import os
import argparse
import json
import time

from PIL import Image
import imagesize
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_image_size(image_path):
    """
    Resize image so that the shorter side is equal to target_size while maintaining the aspect ratio.
    Returns a tuple (image_path, new_width, new_height).
    If the image area is below target_size^2, no resizing is done and the original dimensions are returned.
    """
    # with Image.open(image_path) as img:
    #     width, height = img.sizel

    try:
        width, height = imagesize.get(image_path)
    except Exception as e:
        print(e)
        width = height = 0
    return image_path, width, height


def traverse_folder(folder_path, num_threads=8):
    image_paths = []
    # Walk through the folder and collect image file paths.
    walk_fn = os.walk

    for root, dirs, files in walk_fn(folder_path):
        for file in files:
            if not file.startswith('.') and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    print(f"{len(image_paths)} images in {folder_path}")
    return image_paths


def read_size(image_paths, num_threads=8):
    print(f"{len(image_paths)} images")

    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(read_image_size, image_path) for image_path in image_paths]
        # Wait for all threads to complete.
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

            if len(results) % 2000 == 1999:
                print(len(results), (time.time() - start_time) / len(results))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize images in a folder so that the shorter side is set to a target size, and record their dimensions."
    )
    parser.add_argument("--input_folder", type=str, required=True, default=None,
                        help="Path to the folder containing images.")
    parser.add_argument("--threads", type=int, default=256, help="Number of threads to use for processing.")
    parser.add_argument("--output_json", type=str, required=True, default=None,
                        help="Output JSON file path to store image sizes.")
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    input_folder = args.input_folder
    num_threads = args.threads
    output_json = args.output_json

    if not os.path.isdir(input_folder):
        print("Invalid folder path.")
        exit(1)

    # If output_json is not specified, default to 'image_sizes.json' inside the input folder.
    if not os.path.exists(os.path.split(output_json)[0]):
        os.makedirs(os.path.split(output_json)[0], exist_ok=True)

    print(f"Starting to process images in {input_folder}...")
    # Process images and obtain a list of (path, width, height).
    image_paths = traverse_folder(input_folder, num_threads)

    if args.debug:
        image_paths = image_paths[:1000]

    results = read_size(image_paths, num_threads)
    print("All images have been processed successfully.")

    # Build a dictionary mapping image paths to their dimensions.
    sizes_dict = [dict(image=os.path.relpath(path, input_folder), width=width, height=height) for path, width, height in
                  results if width]

    # Write the dictionary to the specified JSON file.
    with open(output_json, "w") as f:
        json.dump(sizes_dict, f, indent=4)
    print(f"Image sizes saved to {output_json}")
