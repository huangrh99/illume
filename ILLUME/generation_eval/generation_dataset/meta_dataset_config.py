from illume.data.aspect_ratio_utils import RATIOS

# -------------------- text2image Generation ------------------------
Text2ImageExampleDataset = dict(
    type="DefaultDataset",
    annotation_path="../configs/data_configs/test_data_examples/Text2ImageExample/t2i_test_examples.jsonl",
    role="text2image",
    unconditional_role="random2image",
    ratios=RATIOS
)


# -------------------- editing task ------------------------
EditingSingleTurnExampleDataset = dict(
    type="EditingSingleTurnDataset",
    annotation_path="../configs/data_configs/test_data_examples/EditingSingleTurnExample/edit_test_examples.jsonl",
    image_dir="../configs/data_configs/test_data_examples/EditingSingleTurnExample/images",
    role="editing",
    unconditional_role="image_reconstruction",
    ratios=[(-1, -1)],
    sample_num=-1
)
