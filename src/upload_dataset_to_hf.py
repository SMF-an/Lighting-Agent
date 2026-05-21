import argparse
import json
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi, get_full_repo_name


load_dotenv()


def build_dataset_card(repo_id):
    return f"""---
language: en
license: mit
tags:
- lighting
- image-captioning
- dataset
- vision-language
---

This dataset contains light-effect images and paired English captions.

## Contents

- `train`: prepared split with embedded image-caption pairs

## Usage
You can load this dataset using the Hugging Face Datasets library:

```python
from datasets import load_dataset
dataset = load_dataset("{repo_id}", split="train")
```
"""


def read_caption_records(source_dir):
    captions_path = source_dir / "light_effect_captions.jsonl"
    if not captions_path.exists():
        raise FileNotFoundError(f"Caption file not found: {captions_path}")

    records = []
    with captions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            image_value = record.get("image")
            caption = record.get("caption")

            if image_value is None:
                raise KeyError("Each caption record must contain an 'image' field.")
            if caption is None:
                raise KeyError("Each caption record must contain a 'caption' field.")

            image_path = source_dir / image_value
            if not image_path.exists():
                raise FileNotFoundError(f"Image referenced in captions file not found: {image_path}")

            records.append(
                {
                    "image": str(image_path),
                    "caption": caption,
                }
            )

    if not records:
        raise ValueError(f"No caption records found in {captions_path}")

    return records


def build_dataset_dict(source_dir):
    features = Features(
        {
            "image": Image(),
            "caption": Value("string"),
        }
    )
    dataset = Dataset.from_list(read_caption_records(source_dir)).cast(features)
    return DatasetDict({"train": dataset})


def upload_dataset_folder(dataset_name, source_dir):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

    repo_id = get_full_repo_name(dataset_name)

    dataset_dict = build_dataset_dict(source_dir)
    dataset_dict.push_to_hub(repo_id, token=token)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=BytesIO(build_dataset_card(repo_id).encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the lighting dataset to Hugging Face Datasets.")
    parser.add_argument(
        "--dataset-name",
        default="light-effect-dataset",
        help="Name of the dataset repository to create on Hugging Face (e.g., 'my-username/my-light-effect-dataset').",
    )
    parser.add_argument(
        "--source-dir",
        default="../data",
        help="Local dataset folder to upload (contains images/ and light_effect_captions.jsonl).",
    )
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {source_dir}")

    upload_dataset_folder(
        dataset_name=args.dataset_name,
        source_dir=source_dir,
    )
    print(f"Uploaded {source_dir} to Hugging Face Datasets as '{args.dataset_name}'.")


if __name__ == "__main__":
    main()
