import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from simulate_graph_one import generate_image
from simulate_graph_more import generate_gradient_image
from generate_light_caption import export_captions


def build_dataset(
    root_dir: Path,
    num_images_one: int = 100,
    num_images_more: int = 100,
    width: int = 768,
    height: int = 768,
    seed_one: int = 42,
    seed_more: int = 43,
) -> int:
    images_root = root_dir / "images"
    captions_path = root_dir / "light_effect_captions.jsonl"
    images_root.mkdir(parents=True, exist_ok=True)
    
    print("===== Step 1: Generating images =====")

    total_images = 0
    rng_one = np.random.default_rng(seed_one)
    for _ in tqdm(range(num_images_one), desc="simulate_graph_one", unit="img"):
        img, _, _ = generate_image(width, height, rng_one)
        total_images += 1
        save_path = images_root / f"{total_images:04d}.png"
        Image.fromarray(img).save(save_path)

    rng_more = np.random.default_rng(seed_more)
    for _ in tqdm(range(num_images_more), desc="simulate_graph_more", unit="img"):
        img, _, _ = generate_gradient_image(width, height, rng_more)
        total_images += 1
        save_path = images_root / f"{total_images:04d}.png"
        Image.fromarray(img).save(save_path)

    print("===== Step 2: Exporting captions =====")
    caption_total = export_captions(images_root, captions_path)

    print(f"Done. Generated {total_images} images and wrote {caption_total} caption records to {captions_path}")
    return caption_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the light-effect dataset pipeline.")
    parser.add_argument("--root-dir", type=Path, default=Path("../../data"), help="Dataset output root directory.")
    parser.add_argument("--num-images-one", type=int, default=1000, help="Number of images to generate for simulate_graph_one.")
    parser.add_argument("--num-images-more", type=int, default=1000, help="Number of images to generate for simulate_graph_more.")
    parser.add_argument("--width", type=int, default=768, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=768, help="Image height in pixels.")
    parser.add_argument("--seed-one", type=int, default=42, help="Random seed for simulate_graph_one.")
    parser.add_argument("--seed-more", type=int, default=43, help="Random seed for simulate_graph_more.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        root_dir=args.root_dir,
        num_images_one=args.num_images_one,
        num_images_more=args.num_images_more,
        width=args.width,
        height=args.height,
        seed_one=args.seed_one,
        seed_more=args.seed_more,
    )


if __name__ == "__main__":
    main()