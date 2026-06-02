import argparse
import csv
from pathlib import Path

import torch

from inference import load_pipeline, generate_image


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


def run_single_case(
    *,
    pipeline,
    prompts: list[str],
    output_dir: Path,
    group_name: str,
    run_index: int,
    num_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    seed: int | None
) -> list[dict[str, str]]:
    images = generate_image(
        pipeline=pipeline,
        prompts=prompts,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        return_process=False,
    )
    if not images:
        raise RuntimeError("generate_image returned no images")
    if len(images) != len(prompts):
        raise RuntimeError("generate_image returned a mismatched number of images")

    rows: list[dict[str, str]] = []
    for prompt_index, (prompt, image) in enumerate(zip(prompts, images), start=1):
        filename = (
            f"{prompt_index}_{group_name}_{run_index:03d}_steps_{num_steps}"
            f"_guidance_{guidance_scale:g}"
            f"_neg_{int(bool(negative_prompt))}.png"
        )
        image_path = output_dir / filename
        image.save(image_path)

        row = {
            "prompt_text": prompt,
            "group": group_name,
            "run_index": str(run_index),
            "num_steps": str(num_steps),
            "guidance_scale": f"{guidance_scale:g}",
            "use_negative_prompt": bool(negative_prompt),
            "image_path": str(image_path),
        }
        rows.append(row)

    return rows


def write_rows_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "prompt_text",
                "group",
                "run_index",
                "num_steps",
                "guidance_scale",
                "use_negative_prompt",
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
            "Soft gradient lighting transitions from vibrant pink to bright green, creating a dynamic and energetic atmosphere with warm undertones.",
            "Soft gradient lighting transitions from pale yellow to gentle purple, creating a dreamy and ethereal atmosphere with subtle glowing highlights.",
            "Soft gradient lighting transitions from lavender to warm pink with a subtle coral glow, creating a dreamy and serene atmosphere.",
            "Soft gradient lighting transitions from warm peach to gentle mint, creating a serene and inviting atmosphere.",
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
    steps_dir = output_dir / "num_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    steps_rows: list[dict[str, str]] = []
    for index, num_steps in enumerate(num_steps_list, start=1):
        rows = run_single_case(
            pipeline=pipeline,
            prompts=prompts,
            output_dir=steps_dir,
            group_name="num_steps",
            run_index=index,
            num_steps=num_steps,
            guidance_scale=args.base_guidance_scale,
            negative_prompt="furniture, sofa, chair, table, desk, room, indoor, objects, scene",
            seed=args.seed,
        )
        steps_rows.extend(rows)
    write_rows_csv(steps_dir / "results.csv", steps_rows)
    all_rows.extend(steps_rows)

    # 2) Vary guidance_scale only.
    guidance_dir = output_dir / "vary_guidance_scale"
    guidance_dir.mkdir(parents=True, exist_ok=True)
    guidance_rows: list[dict[str, str]] = []
    for index, guidance_scale in enumerate(guidance_list, start=1):
        rows = run_single_case(
            pipeline=pipeline,
            prompts=prompts,
            output_dir=guidance_dir,
            group_name="vary_guidance_scale",
            run_index=index,
            num_steps=args.base_num_steps,
            guidance_scale=guidance_scale,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
        )
        guidance_rows.extend(rows)
    write_rows_csv(guidance_dir / "results.csv", guidance_rows)
    all_rows.extend(guidance_rows)

    # 3) Vary negative prompt usage only.
    negative_dir = output_dir / "vary_negative_prompt"
    negative_dir.mkdir(parents=True, exist_ok=True)
    negative_rows: list[dict[str, str]] = []
    rows = run_single_case(
        pipeline=pipeline,
        prompts=prompts,
        output_dir=negative_dir,
        group_name="vary_negative_prompt",
        run_index=index,
        num_steps=args.base_num_steps,
        guidance_scale=args.base_guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    negative_rows.extend(rows)
    rows = run_single_case(
        pipeline=pipeline,
        prompts=prompts,
        output_dir=negative_dir,
        group_name="vary_negative_prompt",
        run_index=index,
        num_steps=args.base_num_steps,
        guidance_scale=args.base_guidance_scale,
        negative_prompt=None,
        seed=args.seed,
    )
    negative_rows.extend(rows)
    write_rows_csv(negative_dir / "results.csv", negative_rows)
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
        default="../../runs/full_20260524_124658/checkpoints/final",
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
        default="10,30,50,80",
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
        default=50,
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
        default="../../result/experiment",
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