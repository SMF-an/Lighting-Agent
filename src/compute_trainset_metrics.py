import json
from pathlib import Path
from PIL import Image

from evaluation import compute_saturation_metrics, high_freq_energy


def compute_dataset_metrics(image_dir: Path) -> dict:
    images = list(image_dir.glob("*.png"))
    
    saturation_values = []
    high_freq_values = []

    for img in images:
        image = Image.open(img).convert("RGB")

        saturation_values.append(float(compute_saturation_metrics(image)))
        high_freq_values.append(float(high_freq_energy(image)))

    saturation_mean = sum(saturation_values) / len(saturation_values)
    high_freq_mean = sum(high_freq_values) / len(high_freq_values)

    return {
        "num_images": len(saturation_values),
        "mean_saturation": saturation_mean,
        "mean_high_freq_energy": high_freq_mean,
    }


if __name__ == "__main__":
    
    image_dir = Path("../data/images")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    metrics = compute_dataset_metrics(image_dir)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
