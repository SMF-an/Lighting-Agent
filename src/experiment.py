import argparse
import csv
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from infer.inference import load_pipeline, generate_image


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("num-steps-list cannot be empty")
    parsed = [int(item) for item in values]
    if any(value <= 0 for value in parsed):
        raise ValueError("all num-steps values must be > 0")
    return parsed


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("guidance-list cannot be empty")
    parsed = [float(item) for item in values]
    if any(value < 0 for value in parsed):
        raise ValueError("all guidance values must be >= 0")
    return parsed


def format_variant_label(variant_name: str, variant_value: int | float | str | bool) -> str:
    if isinstance(variant_value, bool):
        value_text = "on" if variant_value else "off"
    elif isinstance(variant_value, float):
        value_text = f"{variant_value:g}"
    else:
        value_text = str(variant_value)
    return f"{variant_name}={value_text}"


def load_display_font(size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text_to_width(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            candidate_bbox = draw.textbbox((0, 0), candidate, font=font)
            candidate_width = candidate_bbox[2] - candidate_bbox[0]
            if candidate_width <= max_width:
                current_line = candidate
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return "\n".join(lines)


def save_contact_sheet(
    *,
    prompt: str,
    variant_images: list[tuple[Image.Image, str]],
    output_path: Path,
    columns: int = 4,
) -> None:
    if not variant_images:
        raise ValueError("variant_images must be non-empty")

    columns = max(1, min(columns, len(variant_images)))
    rows = math.ceil(len(variant_images) / columns)

    sample_image = variant_images[0][0].convert("RGB")
    tile_width, tile_height = sample_image.size
    label_height = 44
    padding = 16

    measurement_canvas = Image.new("RGB", (1, 1), "white")
    measurement_draw = ImageDraw.Draw(measurement_canvas)
    sheet_width = padding * 2 + columns * tile_width + (columns - 1) * padding

    title_font = load_display_font(24)
    label_font = load_display_font(18)

    wrapped_prompt = wrap_text_to_width(prompt, title_font, sheet_width - padding * 2, measurement_draw) or prompt
    title_bbox = measurement_draw.multiline_textbbox((0, 0), wrapped_prompt, font=title_font, spacing=8)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]

    sheet_height = padding * 3 + title_height + rows * (tile_height + label_height) + (rows - 1) * padding

    sheet = Image.new("RGB", (sheet_width, sheet_height), (248, 247, 243))
    draw = ImageDraw.Draw(sheet)
    title_x = max(padding, (sheet_width - title_width) // 2)
    draw.multiline_text(
        (title_x, padding),
        wrapped_prompt,
        fill=(28, 28, 28),
        font=title_font,
        spacing=8,
        align="center",
    )

    start_y = padding * 2 + title_height
    for index, (image, label) in enumerate(variant_images):
        row = index // columns
        col = index % columns

        x = padding + col * (tile_width + padding)
        y = start_y + row * (tile_height + label_height + padding)

        tile = image.convert("RGB")
        if tile.size != (tile_width, tile_height):
            tile = tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS)

        sheet.paste(tile, (x, y))
        draw.rectangle(
            [x, y + tile_height, x + tile_width, y + tile_height + label_height],
            fill=(255, 255, 255),
        )
        draw.rectangle([x, y, x + tile_width - 1, y + tile_height + label_height - 1], outline=(190, 190, 190))

        label_bbox = measurement_draw.textbbox((0, 0), label, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height_px = label_bbox[3] - label_bbox[1]
        label_x = x + max(0, (tile_width - label_width) // 2)
        label_y = y + tile_height + max(0, (label_height - label_height_px) // 2) - 1
        draw.text((label_x, label_y), label, fill=(24, 24, 24), font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def run_variant_group(
    *,
    pipeline,
    prompts: list[str],
    output_dir: Path,
    variant_name: str,
    variant_values: list[int | float | bool],
    base_num_steps: int,
    base_guidance_scale: float,
    negative_prompt: str | None,
    seed: int | None,
) -> list[dict[str, str]]:
    group_dir = output_dir / variant_name
    group_dir.mkdir(parents=True, exist_ok=True)

    prompt_variant_images: list[list[tuple[Image.Image, str]]] = [[] for _ in prompts]
    variant_labels: list[str] = []

    for variant_value in tqdm(variant_values, desc=variant_name, unit="setting"):
        if variant_name == "num_steps":
            num_steps = int(variant_value)
            guidance_scale = base_guidance_scale
            current_negative_prompt = negative_prompt
        elif variant_name == "guidance_scale":
            num_steps = base_num_steps
            guidance_scale = float(variant_value)
            current_negative_prompt = negative_prompt
        elif variant_name == "negative_prompt":
            num_steps = base_num_steps
            guidance_scale = base_guidance_scale
            current_negative_prompt = negative_prompt if bool(variant_value) else None
        else:
            raise ValueError(f"Unsupported variant_name: {variant_name}")

        images = generate_image(
            pipeline=pipeline,
            prompts=prompts,
            negative_prompt=current_negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            return_process=False,
        )
        if not images:
            raise RuntimeError("generate_image returned no images")
        if len(images) != len(prompts):
            raise RuntimeError("generate_image returned a mismatched number of images")

        label = format_variant_label(variant_name, variant_value)
        variant_labels.append(label)
        for prompt_index, image in enumerate(images):
            prompt_variant_images[prompt_index].append((image, label))

    rows: list[dict[str, str]] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        image_path = group_dir / f"prompt_{prompt_index:02d}.png"
        save_contact_sheet(
            prompt=prompt,
            variant_images=prompt_variant_images[prompt_index - 1],
            output_path=image_path,
        )

        rows.append(
            {
                "prompt_text": prompt,
                "variant_name": variant_name,
                "variant_values": " | ".join(variant_labels),
                "image_path": str(image_path),
            }
        )

    return rows


def write_rows_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "prompt_text",
                "group",
                "variant_name",
                "variant_values",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_steps_list = parse_int_list(args.num_steps_list)
    guidance_list = parse_float_list(args.guidance_list)

    prompts = [
        "Abstract light effect, uniform from purple to bright yellow-green, energetic atmosphere, smooth texture, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft blue to lime green, calm atmosphere, diffuse light, no room, no furniture, no objects.",
        "Abstract light effect, uniform from bright teal to cool blue, balanced atmosphere, diffuse light, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft teal to gentle cyan, balanced atmosphere, smooth texture, no room, no furniture, no objects.",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_type=args.checkpoint_type,
        device=device,
    )

    all_rows: list[dict[str, str]] = []

    # 1) Vary num_steps.
    steps_rows = run_variant_group(
        pipeline=pipeline,
        prompts=prompts,
        output_dir=output_dir,
        variant_name="num_steps",
        variant_values=num_steps_list,
        base_num_steps=args.base_num_steps,
        base_guidance_scale=args.base_guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    all_rows.extend(steps_rows)

    # 2) Vary guidance_scale only.
    guidance_rows = run_variant_group(
        pipeline=pipeline,
        prompts=prompts,
        output_dir=output_dir,
        variant_name="guidance_scale",
        variant_values=guidance_list,
        base_num_steps=args.base_num_steps,
        base_guidance_scale=args.base_guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    all_rows.extend(guidance_rows)

    # 3) Vary negative prompt usage only.
    negative_rows = run_variant_group(
        pipeline=pipeline,
        prompts=prompts,
        output_dir=output_dir,
        variant_name="negative_prompt",
        variant_values=[True, False],
        base_num_steps=args.base_num_steps,
        base_guidance_scale=args.base_guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    all_rows.extend(negative_rows)

    summary_csv_path = output_dir / "experiment_summary.csv"
    write_rows_csv(summary_csv_path, all_rows)
    return summary_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=("Hyperparameter experiment for Stable Diffusion generation settings. "))
    parser.add_argument(
        "--base-model",
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model ID.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="../runs/full_color_high_freq_text/checkpoints/final",
        help="Path to the fine-tuned checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["pretrain", "full", "lora"],
        default="full",
        help="Checkpoint type to load.",
    )
    parser.add_argument(
        "--num-steps-list",
        default="5,10,30,50",
        help="Comma-separated inference steps list.",
    )
    parser.add_argument(
        "--guidance-list",
        default="2.5,5.0,7.5,10.0",
        help="Comma-separated guidance scale list.",
    )
    parser.add_argument(
        "--base-num-steps",
        type=int,
        default=30,
        help="Fixed num_steps used when experimenting with other variables.",
    )
    parser.add_argument(
        "--base-guidance-scale",
        type=float,
        default=7.5,
        help="Fixed guidance_scale used when experimenting with other variables.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="furniture, sofa, chair, table, desk, room, indoor, objects, scene",
        help="Negative prompt text used when negative prompt is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility. Set to None by removing this argument.",
    )
    parser.add_argument(
        "--output-dir",
        default="../result/experiment",
        help="Directory to save generated images and per-variable CSV results.",
    )
    args = parser.parse_args()

    if args.base_num_steps <= 0:
        raise ValueError("base-num-steps must be > 0")
    if args.base_guidance_scale < 0:
        raise ValueError("base-guidance-scale must be >= 0")

    csv_path = run_experiment(args)
    print(f"\nExperiment complete. Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
