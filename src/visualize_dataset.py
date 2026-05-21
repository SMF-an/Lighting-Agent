import argparse
import math
import os
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from data.dataset import build_train_dataset, resolve_image_value


def sample_records(records, num_samples, seed):
    if not records:
        raise ValueError("No records found to visualize.")

    rng = random.Random(seed)
    if num_samples >= len(records):
        return list(records)
    return rng.sample(records, num_samples)


def build_grid_figure(records, image_root, output_path, cols=3, caption_width=42):
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6.2 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    flat_axes = [ax for row in axes for ax in row]

    for ax in flat_axes[len(records):]:
        ax.axis("off")

    # Reserve extra vertical space between image and caption to avoid overlap
    fig.subplots_adjust(hspace=0.6, wspace=0.3, top=0.92)

    for idx, (ax, record) in enumerate(zip(flat_axes, records), start=1):
        image_value = record.get("image")
        if image_value is None:
            raise KeyError("Each record must contain an 'image' field.")

        resolved_image = resolve_image_value(image_value, image_root=image_root)
        caption = record.get("caption", "")
        if isinstance(resolved_image, Image.Image):
            image = resolved_image.convert("RGB")
        else:
            image = Image.open(resolved_image).convert("RGB")

        ax.imshow(image)
        ax.axis("off")

        wrapped_caption = textwrap.fill(caption, width=caption_width)
        # Increase title padding so it doesn't overlap the image area
        ax.set_title(f"{idx}. {wrapped_caption}", fontsize=10, pad=14)

    fig.suptitle("Dataset Preview", fontsize=16, y=0.995)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the light effect dataset as an image-caption grid.")
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        help="Load the dataset from a Hugging Face dataset repository instead of local files.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="theavenger/light-effect-dataset",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--jsonl-path",
        default="../data/light_effect_captions.jsonl",
        help="Local JSONL file containing image-caption pairs.",
    )
    parser.add_argument(
        "--image-root",
        default="../data",
        help="Local image root used to resolve relative image paths.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of samples to display.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns in the visualization grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--output",
        default="../result/dataset_preview.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--caption-width",
        type=int,
        default=42,
        help="Text wrapping width for captions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    dataset = build_train_dataset(
        use_hf_dataset=args.use_hf_dataset,
        hf_repo_id=args.hf_repo_id,
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        split="train",
        token=token,
    )

    records = dataset.dataset
    image_root = dataset.image_root

    selected_records = sample_records(records, args.num_samples, args.seed)
    build_grid_figure(
        records=selected_records,
        image_root=image_root,
        output_path=Path(args.output),
        cols=args.cols,
        caption_width=args.caption_width,
    )
    print(f"Saved preview to {args.output}")


if __name__ == "__main__":
    main()
