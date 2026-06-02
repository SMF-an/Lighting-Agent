"""
Evaluation script for light-effect images generation.
"""
from pathlib import Path
import argparse
import csv
import json
import string
import random
import tempfile
import shutil

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import os
from transformers import CLIPProcessor, CLIPModel
from torch_fidelity import calculate_metrics

# Prevent tokenizers parallelism deadlock when forking processes later
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from infer.inference import load_pipeline, generate_image
from data.generate_random_prompt import sample_prompt, VOCABULARY


EPS = 1e-12
SCENE_LEAK_PROBES = [
    "a living room with a sofa and table",
    "an indoor room with furniture",
    "a bedroom with a bed and chair",
    "a kitchen with cabinets and appliances",
    "an office room with desk and computer",
    "a hallway or interior room scene",
]


def slugify(text: str) -> str:
    keep = "-_" + string.ascii_letters + string.digits
    return "".join(c if c in keep else ("_" if c.isspace() else "") for c in text)[:120]


def compute_hue_hist(image: Image.Image, bins: int = 32) -> np.ndarray:
    """Compute normalized hue histogram (HSV H channel from PIL.convert('HSV'))."""
    hsv = image.convert("HSV")
    arr = np.array(hsv)
    h = arr[:, :, 0].ravel()  # 0..255
    hist, _ = np.histogram(h, bins=bins, range=(0, 255))
    hist = hist.astype(float)
    s = hist.sum()
    if s <= 0:
        return np.ones(bins) / bins
    return hist / (s + EPS)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float) + EPS
    q = q.astype(float) + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * np.log(a / b))
    return 0.5 * (kl(p, m) + kl(q, m))


def _safe_float(value):
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_prompts_from_json(prompt_path: Path) -> list[str]:
    """Load prompts from a JSONL file with one object per line."""
    if not prompt_path.exists():
        return []

    prompts = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt = item.get("prompt") if isinstance(item, dict) else None
        if isinstance(prompt, str):
            prompt = prompt.strip()
            if prompt:
                prompts.append(prompt)

    return prompts


def select_prompts(prompt_pool: list[str], num_prompts: int, rng: random.Random) -> list[str]:
    if not prompt_pool:
        return []
    if num_prompts <= len(prompt_pool):
        return rng.sample(prompt_pool, num_prompts)
    selected = list(prompt_pool)
    while len(selected) < num_prompts:
        selected.append(prompt_pool[rng.randrange(len(prompt_pool))])
    rng.shuffle(selected)
    return selected[:num_prompts]


def summarize_metrics(csv_path: Path) -> dict:
    """Aggregate per-sample metrics into checkpoint-level summary statistics."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    metric_names = [
        "clip_score",
        "scene_leak_score",
        "hue_js",
        "high_freq",
        "prompt_consistency",
        "mean_saturation"
    ]
    summary = {
        "csv_path": str(csv_path),
        "num_rows": len(rows),
        "metrics": {},
    }

    for metric_name in metric_names:
        values = np.array([_safe_float(row.get(metric_name)) for row in rows], dtype=float)
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            summary["metrics"][metric_name] = {
                "mean": None,
                "std": None
            }
            continue

        summary["metrics"][metric_name] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid))
        }

    return summary


def compute_reference_hist_from_dir(image_dir: Path, bins: int = 32, max_images: int = 200) -> np.ndarray:
    images = list(image_dir.glob("*.png"))
    if not images:
        return np.ones(bins) / bins
    imgs = images[:max_images]
    acc = np.zeros(bins, dtype=float)
    for p in imgs:
        try:
            im = Image.open(p).convert("RGB")
            acc += compute_hue_hist(im, bins=bins)
        except Exception:
            continue
    s = acc.sum()
    if s <= 0:
        return np.ones(bins) / bins
    return acc / (s + EPS)


def high_freq_energy(image: Image.Image, cutoff_ratio: float = 0.15) -> float:
    """Match the training-time high-frequency loss logic on an RGB image.

    This uses the same luminance conversion, 128x128 resize, rFFT, and cross-shaped
    frequency mask as `compute_highfreq_loss` in `full_finetune.py`.
    """
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    luminance = torch.from_numpy(luminance).unsqueeze(0).unsqueeze(0)
    luminance = torch.nn.functional.interpolate(
        luminance,
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    Fk = torch.fft.rfftn(luminance, dim=(0, 1))
    mag2 = (Fk.real ** 2 + Fk.imag ** 2)

    H, Wp = mag2.shape[0], mag2.shape[1]
    fy = torch.fft.fftfreq(H).abs().unsqueeze(1)
    fx = torch.fft.rfftfreq(luminance.shape[1]).unsqueeze(0)

    row_band = fy <= float(cutoff_ratio)
    col_band = fx <= float(cutoff_ratio)
    mask = torch.logical_or(row_band, col_band).float()

    outside_mask = 1.0 - mask
    hf_energy = (mag2 * outside_mask).sum()
    total_energy = mag2.sum()
    if total_energy <= 0:
        return 0.0
    return float((hf_energy / (total_energy + 1e-8)).item())


def _collect_image_paths_from_dir_or_list(paths_or_dir):
    if isinstance(paths_or_dir, (list, tuple)):
        return [Path(p) for p in paths_or_dir]
    return list(Path(paths_or_dir).rglob("*.png"))


def compute_kid_with_torch_fidelity(gen_paths, ref_dir: Path, device: str = "cpu", max_images: int = 500) -> float:
    """Compute KID using torch-fidelity's calculate_metrics API.

    Accepts either a directory or a list of image paths for `gen_paths`.
    """
    gen_list = _collect_image_paths_from_dir_or_list(gen_paths)
    ref_list = list(Path(ref_dir).rglob("*.png"))
    if not gen_list or not ref_list:
        raise ValueError("No generated or reference images available for KID computation")

    # Limit number of images by copying first N into a temp dir (torch-fidelity expects dirs)
    with tempfile.TemporaryDirectory() as tmpd:
        tmpd_path = Path(tmpd)
        count = 0
        for p in gen_list:
            if count >= max_images:
                break
            try:
                shutil.copy(str(p), str(tmpd_path / f"{count:06d}.png"))
                count += 1
            except Exception:
                continue

        # call torch-fidelity with an appropriate kid_subset_size
        cuda = True if (isinstance(device, str) and device.startswith("cuda")) else False
        kid_subset_size = int(min(count, len(ref_list), 1000))
        metrics = calculate_metrics(input1=str(tmpd_path), input2=str(ref_dir), cuda=cuda, kid=True, kid_subset_size=kid_subset_size, verbose=False)

        return float(metrics["kernel_inception_distance_mean"]), float(metrics["kernel_inception_distance_std"])


def compute_saturation_metrics(image: Image.Image) -> dict:
    """Compute simple saturation statistics from RGB image."""
    im = image.convert("RGB")
    arr = np.array(im).astype(float) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    # avoid division by zero; saturation = (max-min)/max
    sat = np.where(mx > 1e-8, (mx - mn) / (mx + 1e-8), 0.0)
    mean_sat = float(sat.mean())
    return mean_sat


def compute_clip_score(image: Image.Image, text: str, model: CLIPModel, processor: CLIPProcessor, device: str) -> float:
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    pixel_values = inputs.get("pixel_values")
    input_ids = inputs.get("input_ids")
    if pixel_values is None or input_ids is None:
        # Fallback: use separate encoding
        inputs_img = processor(images=image, return_tensors="pt")
        inputs_txt = processor(text=[text], return_tensors="pt", padding=True)
        pixel_values = inputs_img["pixel_values"]
        input_ids = inputs_txt["input_ids"]

    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        image_emb = model.get_image_features(pixel_values=pixel_values)
        text_emb = model.get_text_features(input_ids=input_ids)
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        # cosine similarity between first image and first text
        score = (image_emb * text_emb).sum(dim=-1).cpu().item()
    return float(score)


def compute_clip_image_embedding(image: Image.Image, model: CLIPModel, processor: CLIPProcessor, device: str) -> torch.Tensor:
    """Compute a normalized CLIP image embedding for image-to-image comparison."""
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(pixel_values=pixel_values)
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
    return image_emb.squeeze(0).detach().cpu()


def compute_prompt_consistency(image_embeddings: list[torch.Tensor]) -> float:
    """Average pairwise cosine similarity among images generated for the same prompt."""
    if len(image_embeddings) < 2:
        return float("nan")
    emb = torch.stack(image_embeddings, dim=0)
    sim = emb @ emb.T
    n = sim.shape[0]
    tri = torch.triu_indices(n, n, offset=1)
    if tri.shape[1] == 0:
        return float("nan")
    return float(sim[tri[0], tri[1]].mean().item())


def compute_scene_leakage_score(image: Image.Image, model: CLIPModel, processor: CLIPProcessor, device: str) -> float:
    """Estimate how strongly the image resembles a scene or furniture-heavy indoor photo."""
    inputs = processor(text=SCENE_LEAK_PROBES, images=image, return_tensors="pt", padding=True)
    pixel_values = inputs.get("pixel_values")
    input_ids = inputs.get("input_ids")
    if pixel_values is None or input_ids is None:
        inputs_img = processor(images=image, return_tensors="pt")
        inputs_txt = processor(text=SCENE_LEAK_PROBES, return_tensors="pt", padding=True)
        pixel_values = inputs_img["pixel_values"]
        input_ids = inputs_txt["input_ids"]

    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        image_emb = model.get_image_features(pixel_values=pixel_values)
        text_emb = model.get_text_features(input_ids=input_ids)
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        score = (image_emb @ text_emb.T).max(dim=-1).values.cpu().item()
    return float(score)


def main():
    parser = argparse.ArgumentParser(description="Evaluate light-effect images from finetuned checkpoints")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5", help="Base model ID.")
    parser.add_argument("--checkpoint-dir", default="../runs/lora/checkpoints/final", help="Directory with checkpoint to evaluate")
    parser.add_argument("--checkpoint-type", choices=["pretrain", "full", "lora"], default="pretrain", help="Checkpoint type to load.")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=16, help="Number of prompts to sample")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--output-dir", default="../result/eval/pretrain", help="Directory to save generated samples and metrics CSV")
    parser.add_argument("--dataset-dir", default="../data/images", help="Directory with reference images for histogram comparison")
    parser.add_argument("--bins", type=int, default=32, help="Number of hue bins for histogram")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-file", default="../result/eval/prompts.json", help="JSON/JSONL file containing prompts to sample from.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_type=args.checkpoint_type,
        device=device,
    )
    print(f"Using checkpoint type: {args.checkpoint_type}")

    # Load CLIP
    print("Loading CLIP model for scoring...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

    # Reference histogram
    ref_hist = compute_reference_hist_from_dir(Path(args.dataset_dir), bins=args.bins)

    # Prepare prompts
    rng = random.Random(args.seed if args.seed is not None else 42)
    prompt_file = Path(args.prompt_file) if args.prompt_file else None
    prompts = []
    if prompt_file is not None:
        prompts = load_prompts_from_json(prompt_file)
        if prompts:
            prompts = select_prompts(prompts, args.num_prompts, rng)
            print(f"Loaded {len(prompts)} prompt(s) from {prompt_file}")

    if not prompts:
        prompts = [sample_prompt(VOCABULARY, rng) for _ in range(args.num_prompts)]
        print("Using random prompts from the built-in vocabulary")

    # CSV setup
    csv_path = output_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "prompt",
        "sample_index",
        "clip_score",
        "scene_leak_score",
        "hue_js",
        "high_freq",
        "prompt_consistency",
        "mean_saturation",
        "image_path",
    ])


    total = len(prompts) * args.samples_per_prompt
    with tqdm(total=total, desc="Sampling", unit="sample") as progress_bar:
        for p_idx, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {p_idx+1}/{len(prompts)}")
            prompt_slug = slugify(prompt)
            prompt_dir = samples_dir / prompt_slug
            prompt_dir.mkdir(parents=True, exist_ok=True)

            prompt_rows = []
            prompt_embeddings = []

            for s in range(args.samples_per_prompt):
                seed = None if args.seed is None else (args.seed + s)
                images = generate_image(
                    pipeline=pipeline,
                    prompts=prompt,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    seed=seed,
                )
                if not images:
                    progress_bar.update(1)
                    continue
                img = images[0]
                img_path = prompt_dir / f"sample_{s:03d}.png"
                img.save(img_path)

                # Metrics
                try:
                    clip_score = compute_clip_score(img, prompt, clip_model, clip_processor, device)
                except Exception:
                    clip_score = float("nan")

                try:
                    clip_image_emb = compute_clip_image_embedding(img, clip_model, clip_processor, device)
                    prompt_embeddings.append(clip_image_emb)
                except Exception:
                    clip_image_emb = None

                try:
                    scene_leak_score = compute_scene_leakage_score(img, clip_model, clip_processor, device)
                except Exception:
                    scene_leak_score = float("nan")

                try:
                    hue_hist = compute_hue_hist(img, bins=args.bins)
                    hue_js = js_divergence(hue_hist, ref_hist)
                except Exception:
                    hue_js = float("nan")

                try:
                    hf = high_freq_energy(img)
                except Exception:
                    hf = float("nan")

                try:
                    mean_sat = compute_saturation_metrics(img)
                except Exception:
                    mean_sat = float("nan")

                prompt_rows.append([
                    prompt,
                    s,
                    clip_score,
                    scene_leak_score,
                    hue_js,
                    hf,
                    mean_sat,
                    str(img_path),
                ])
                progress_bar.update(1)

            prompt_consistency = compute_prompt_consistency(prompt_embeddings)
            for row in prompt_rows:
                row.insert(6, prompt_consistency)
                csv_writer.writerow(row)
            csv_file.flush()

    csv_file.close()

    summary = summarize_metrics(csv_path)

    # Collect generated images and compute KID with torch-fidelity
    gen_images = _collect_image_paths_from_dir_or_list(samples_dir)
    ref_dir = Path(args.dataset_dir)
    if gen_images and ref_dir.exists():
        print("Computing KID using torch-fidelity between generated samples and reference dataset...")
        try:
            kid_mean, kid_std = compute_kid_with_torch_fidelity(gen_images, ref_dir, device=device, max_images=500)
            summary["metrics"]["kid"] = {"mean": float(kid_mean), "std": float(kid_std)}
            print(f"KID: {kid_mean:.6f} ± {kid_std:.6f}")
        except Exception as e:
            print(f"Failed to compute KID with torch-fidelity: {e}")
    else:
        print("Skipping KID: no generated or reference images found")

    # write final summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluation complete. Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
