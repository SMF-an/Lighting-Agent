import json
from io import BytesIO
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset


def resolve_image_value(image_value, image_root=None):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict):
        if image_value.get("path"):
            return resolve_image_value(image_value["path"], image_root=image_root)
        if image_value.get("bytes"):
            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")

    if isinstance(image_value, (str, Path)):
        image_path = Path(image_value)
        if not image_path.is_absolute() and image_root is not None:
            image_path = Path(image_root) / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    raise TypeError("image must be a PIL image or a valid image path.")


class LightEffectDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_column="image",
        caption_column="caption",
        image_root=None,
        size=768
    ):
        self.dataset = dataset
        self.image_column = image_column
        self.caption_column = caption_column
        self.image_root = Path(image_root) if image_root is not None else None
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def _get_sample(self, index):
        sample = self.dataset[index]
        if isinstance(sample, dict):
            return sample
        raise TypeError("Each dataset item must be a dict with image and caption fields.")

    def _resolve_image(self, image_value):
        return resolve_image_value(image_value, image_root=self.image_root)

    def _get_caption(self, sample):
        caption = sample.get(self.caption_column)
        if caption is None:
            raise KeyError(
                f"Missing caption field '{self.caption_column}' and no instance_prompt was provided."
            )
        return caption

    def __getitem__(self, index):
        sample = self._get_sample(index)
        image = self._resolve_image(sample[self.image_column])
        caption = self._get_caption(sample)

        pixel_values = self.transforms(image)

        return {
            "pixel_values": pixel_values,
            "caption": caption,
        }


def load_local_dataset(jsonl_path):
    records = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_hf_dataset(repo_id, split="train", token=None):
    return load_dataset(repo_id, split=split, token=token)


def build_train_dataset(
    use_hf_dataset=False,
    hf_repo_id=None,
    jsonl_path=None,
    image_root=None,
    split="train",
    image_column="image",
    caption_column="caption",
    size=768,
    token=None,
):
    if use_hf_dataset:
        if hf_repo_id is None:
            raise ValueError("hf_repo_id is required when use_hf_dataset=True.")
        dataset = load_hf_dataset(hf_repo_id, split=split, token=token)
        image_root = None
    else:
        if jsonl_path is None:
            raise ValueError("jsonl_path is required when use_hf_dataset=False.")
        dataset = load_local_dataset(jsonl_path)

    return LightEffectDataset(
        dataset=dataset,
        image_column=image_column,
        caption_column=caption_column,
        image_root=image_root,
        size=size,
    )


def build_collate_fn():
    def collate_fn(examples):
        pixel_values = [example["pixel_values"] for example in examples]
        captions = [example["caption"] for example in examples]

        batch = {
            "pixel_values": torch.stack(pixel_values),
            "captions": captions
        }
        return batch

    return collate_fn


def build_train_dataloader(
    use_hf_dataset=False,
    hf_repo_id=None,
    jsonl_path=None,
    image_root=None,
    size=768,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    split="train",
    image_column="image",
    caption_column="caption",
    token=None,
):
    dataset = build_train_dataset(
        use_hf_dataset=use_hf_dataset,
        hf_repo_id=hf_repo_id,
        jsonl_path=jsonl_path,
        image_root=image_root,
        split=split,
        image_column=image_column,
        caption_column=caption_column,
        size=size,
        token=token,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=build_collate_fn(),
    )