# ILLUME

## MLLM Data PREPARATION

#### STEP 1: Prepare dataset
The overall datasets used to train the mllm are summarized in metadata config, see examples in [examples_meta_data_config.json](../../configs/data_configs/train_data_examples/examples_meta_data_config.json).
For each dataset, the dataset config is in following format:

```json
    "dataset_name": {
        "root": "",  # directory of dataset, used to save jsonl file after tokenizer inference
        "image_dir": "", # image directory of dataset
        "annotation_dir": "", # raw annotation directory of dataset
        "tokenizer_inference": true/false, # whether need to inference vision tokenizer: for text-only & understanding task, set as false; for generation task, set as true
    },
```

For each dataset, the annotation files should be transferred into jsonl file. See examples in [train_data_examples](../../configs/data_configs/train_data_examples).

- An example of understanding task is:


```json
    {
        "images": ["0.jpg"], 
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe the image."}, 
            {"from": "gpt", "value": "The image describes a cute frog dressed up like a cowboy with a western theme."}
        ]
    }
```

- An example of text-to-image generation task is:

```json
    {
        "images": ["0.png"], 
        "conversations": [
            {"from": "human", "value": "Generate an image of {resolution_tag}, the content of image is a cute frog dressed up like a cowboy with a western theme.\n"}, 
            {"from": "gpt", "value": "<image>"}
        ]
    }
```
**{resolution_tag}** refers to the image resolution, we will fill it in the dataset prrocessing. For example, **<height_256><width_1024>** indicates that the image resolution is 256 * 1024.

- An example of editing task is:
  
```json
    {
        "images": ["0.png", "1.png"], 
        "conversations": [
            {"from": "human", "value": "<image>\nPlease edit the image according to the instruction: Make the boots a vibrant red.\n"}, 
            {"from": "gpt", "value": "<image>"}
        ]
    }
```

#### STEP 2: Tokenizer codebook inference
For generation/editing dataset, we inference the vision tokenzier in advance to obtain the image resolution distribution statistics of the data and the visual token ID corresponding to the image. See output jsonl examples in [0000.jsonl](../../configs/data_configs/train_data_examples/examples_editing/jsonl_after_tokenizer/dualvitok/fixed_256/0000.jsonl).

```shell
# codebook inference for stage 1&2 data: using fixed 256 resolution
cd vision_tokenizer
export PYTHONPATH=$PYTHONPATH:$(pwd)

TOKENIZER_CONFIG=../configs/example/dualvitok/dualvitok_anyres_max512.py
TOKENIZER_CKPT=../checkpoints/dualvitok/pytorch_model.bin

resolution=256
resolution_type=fixed
DATA_CONFIG=../configs/data_configs/train_data_examples/examples_meta_data_config.json

torchrun --nproc_per_node=8 tokenizer/tokenizer_codebook_inference.py \
--model_config $TOKENIZER_CONFIG \
--tokenizer_checkpoint $TOKENIZER_CKPT \
--data_config $DATA_CONFIG \
--resolution_type $resolution_type \
--resolution $resolution \
--torch_dtype fp32


# codebook inference for stage 2.5 & 3 data: using predefined resolution aspect ratio with pixel numbers close to 512*512
cd vision_tokenizer
TOKENIZER_CONFIG=../configs/example/dualvitok/dualvitok_anyres_max512.py
TOKENIZER_CKPT=../checkpoints/dualvitok/pytorch_model.bin

resolution=512
resolution_type=fixed_anchors
DATA_CONFIG=../configs/data_configs/train_data_examples/examples_meta_data_config.json

torchrun --nproc_per_node=8 tokenizer/tokenizer_codebook_inference.py \
--model_config $TOKENIZER_CONFIG \
--tokenizer_checkpoint $TOKENIZER_CKPT \
--data_config $DATA_CONFIG \
--resolution_type $resolution_type \
--resolution $resolution \
--torch_dtype fp32
```
    