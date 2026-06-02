import argparse
import base64
import json
import mimetypes
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from inference import load_pipeline, generate_image, save_process_images
from prompt_translator import call_llm_for_effect
from sdl_converter import convert_image_to_sdl_with_dither, parse_sdl_file, srgb_to_lab


load_dotenv()


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_rgb_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def compute_colorfulness(rgb_array: np.ndarray) -> float:
    pixels = rgb_array.astype(np.float32)
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    return float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def compute_objective_metrics(image_path: Path, gamut_colors: np.ndarray | None = None) -> dict[str, object]:
    rgb_array = load_rgb_array(image_path)
    pixels = rgb_array.reshape(-1, 3).astype(np.float32)

    luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    max_channel = pixels.max(axis=1)
    min_channel = pixels.min(axis=1)
    saturation = np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )

    metrics: dict[str, object] = {
        "mean_rgb": [round(float(value), 2) for value in pixels.mean(axis=0)],
        "mean_luminance": round(float(luminance.mean()), 2),
        "luminance_std": round(float(luminance.std()), 2),
        "luminance_p10_p90": [
            round(float(np.percentile(luminance, 10)), 2),
            round(float(np.percentile(luminance, 90)), 2),
        ],
        "mean_saturation": round(float(saturation.mean()), 4),
        "colorfulness": round(compute_colorfulness(rgb_array), 2),
        "unique_color_count": int(np.unique(rgb_array.reshape(-1, 3), axis=0).shape[0]),
    }

    if gamut_colors is not None and len(gamut_colors) > 0:
        pixels_uint8 = rgb_array.reshape(-1, 3)
        unique_pixels, counts = np.unique(pixels_uint8, axis=0, return_counts=True)
        gamut_set = {tuple(color.tolist()) for color in gamut_colors}
        exact_mask = np.array([tuple(color.tolist()) in gamut_set for color in unique_pixels], dtype=bool)
        exact_pixel_count = int(counts[exact_mask].sum())
        total_pixel_count = int(counts.sum())

        sample_size = min(len(unique_pixels), 4000)
        if sample_size > 0:
            sample_indices = np.linspace(0, len(unique_pixels) - 1, sample_size, dtype=int)
            sample_rgb = unique_pixels[sample_indices]
            sample_lab = srgb_to_lab(sample_rgb)
            gamut_lab = srgb_to_lab(gamut_colors)

            nearest_distances = []
            batch_size = 512
            for start in range(0, len(sample_lab), batch_size):
                end = min(start + batch_size, len(sample_lab))
                batch_lab = sample_lab[start:end]
                diff = batch_lab[:, np.newaxis, :] - gamut_lab[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff * diff, axis=2))
                nearest_distances.append(distances.min(axis=1))

            nearest_distances_array = np.concatenate(nearest_distances)
            mean_min_delta_e = float(nearest_distances_array.mean())
            p95_min_delta_e = float(np.percentile(nearest_distances_array, 95))
        else:
            mean_min_delta_e = 0.0
            p95_min_delta_e = 0.0

        metrics["gamut"] = {
            "exact_pixel_ratio": round(exact_pixel_count / max(total_pixel_count, 1), 6),
            "exact_unique_ratio": round(float(exact_mask.mean()) if len(exact_mask) else 0.0, 6),
            "mean_nearest_delta_e": round(mean_min_delta_e, 4),
            "p95_nearest_delta_e": round(p95_min_delta_e, 4),
            "exact_pixel_count": exact_pixel_count,
            "total_pixel_count": total_pixel_count,
        }

    return metrics


def build_evaluation_payload(
    scene_text: str,
    prompt: str,
    raw_image_path: Path,
    final_image_path: Path,
    gamut_colors: np.ndarray,
) -> dict[str, object]:
    raw_metrics = compute_objective_metrics(raw_image_path, gamut_colors=gamut_colors)
    final_metrics = compute_objective_metrics(final_image_path, gamut_colors=gamut_colors)

    return {
        "scene_text": scene_text,
        "prompt": prompt,
        "raw_image": raw_metrics,
        "final_image": final_metrics,
        "derived": {
            "mean_luminance_delta": round(
                float(final_metrics["mean_luminance"]) - float(raw_metrics["mean_luminance"]),
                2,
            ),
            "mean_saturation_delta": round(
                float(final_metrics["mean_saturation"]) - float(raw_metrics["mean_saturation"]),
                4,
            ),
        },
    }


def call_vlm_to_select_best_candidate(
    image_paths: list[Path],
    scene_text: str,
    prompt: str,
) -> tuple[int, str]:
    if not image_paths:
        raise ValueError("image_paths must be a non-empty list")
    
    model_id = os.getenv("VLM_MODEL_ID", "qwen-vl-max-latest")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    user_content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "You are selecting the single best candidate image for a lighting-effect generation task.\n"
                f"Scene text: {scene_text}\n"
                f"Target prompt: {prompt}\n"
                f"There are {len(image_paths)} candidate images.\n"
                "Return JSON only in the form {\"selected_index\": <0-based index>, \"reason\": \"short reason\"}.\n"
                "Choose the image that best matches the scene semantics, lighting atmosphere, and prompt details."
            ),
        }
    ]

    for index, image_path in enumerate(image_paths):
        user_content.append(
            {
                "type": "text",
                "text": f"Candidate {index}: {image_path.name}",
            }
        )
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image_path)},
            }
        )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are a careful visual selector. Reply with JSON only.",
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "selected_index" in parsed:
            selected_index = int(parsed["selected_index"])
            if selected_index < 0 or selected_index >= len(image_paths):
                raise ValueError("selected_index is out of range")
            reason = str(parsed.get("reason", ""))
            return selected_index, reason
    except Exception:
        pass

    for token in raw_text.replace("\n", " ").split():
        if token.isdigit():
            selected_index = int(token)
            if 0 <= selected_index < len(image_paths):
                return selected_index, raw_text

    raise RuntimeError(f"VLM selection response could not be parsed: {raw_text}")


def call_vlm_to_score_quality(
    raw_image_path: Path,
    final_image_path: Path,
    scene_text: str,
    prompt: str,
    objective_payload: dict[str, object],
) -> dict[str, object]:
    model_id = os.getenv("VLM_MODEL_ID", "qwen-vl-max-latest")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    user_content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "You are evaluating a lighting-effect image generation pipeline.\n"
                f"Scene text: {scene_text}\n"
                f"Target prompt: {prompt}\n\n"
                "Objective metrics are provided below. Use them together with the images; do not ignore the metrics.\n"
                f"{json.dumps(objective_payload, ensure_ascii=False, indent=2)}\n\n"
                "Score the result in JSON only with these keys: lighting_plausibility_score, gamut_adherence_score, overall_score, lighting_reason, gamut_reason, verdict.\n"
                "Use scores from 1 to 10.\n"
                "Lighting plausibility should reflect whether the image looks like a coherent light-effect scene.\n"
                "Gamut adherence should reflect whether the final image strictly obeys the SDL gamut constraint and whether the color transition stays consistent after conversion.\n"
                "Overall score should balance both."
            ),
        },
        {
            "type": "text",
            "text": "Raw image before SDL conversion:",
        },
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(raw_image_path)},
        },
        {
            "type": "text",
            "text": "Final image after SDL conversion:",
        },
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(final_image_path)},
        },
    ]

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are a meticulous visual quality evaluator. Reply with JSON only.",
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "lighting_plausibility_score": None,
        "gamut_adherence_score": None,
        "overall_score": None,
        "lighting_reason": "failed to parse VLM response",
        "gamut_reason": raw_text,
        "verdict": "unparsed",
    }


def run_pipeline(args: argparse.Namespace) -> tuple[Path, Path, str, dict[str, object] | None]:
    scene_text = " ".join(args.scene).strip()
    if not scene_text:
        raise ValueError("scene description cannot be empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Translating scene description into a controlled prompt...")
    prompt = call_llm_for_effect(scene_text)
    print(f"Prompt: {prompt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/3] Loading model on {device}...")
    pipeline = load_pipeline(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_type=args.checkpoint_type,
        device=device,
        sampler=args.sampler,
    )
    print(f"Using checkpoint type: {args.checkpoint_type}")

    num_candidates = max(1, args.num_candidates)
    candidate_images: list[Path] = []

    if args.use_candidate_selection:
        print(f"[2/3] Generating {num_candidates} candidate lighting images...")
        candidate_dir = output_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        if args.seed is not None:
            seed_values = [args.seed + index for index in range(num_candidates)]
        else:
            seed_values = [random.randint(0, 2**31 - 1) for _ in range(num_candidates)]

        for index, seed_value in enumerate(seed_values, start=1):
            candidate_result = generate_image(
                pipeline=pipeline,
                prompts=prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                seed=seed_value,
                return_process=args.return_process,
                num_process_images=args.num_process_images,
            )

            if args.return_process:
                candidate_final_images, process_images = candidate_result
                if not candidate_final_images:
                    raise ValueError("image generation returned no results")
                candidate_image = candidate_final_images[0]
                candidate_process_dir = candidate_dir / f"candidate_{index:02d}"
                candidate_process_dir.mkdir(parents=True, exist_ok=True)
                save_process_images(process_images, candidate_process_dir)
            else:
                candidate_images_list = candidate_result
                if not candidate_images_list:
                    raise ValueError("image generation returned no results")
                candidate_image = candidate_images_list[0]

            candidate_path = candidate_dir / f"candidate_{index:02d}.png"
            candidate_image.save(candidate_path)
            candidate_images.append(candidate_path)

        selected_index, reason = call_vlm_to_select_best_candidate(
            candidate_images,
            scene_text=scene_text,
            prompt=prompt
        )
        raw_path = candidate_images[selected_index]
        print(f"Selected candidate #{selected_index + 1}: {raw_path.name}")
        if reason:
            print(f"VLM reason: {reason}")
        # Reload the selected image from disk to keep the downstream flow identical.
        raw_image = Image.open(raw_path).convert("RGB")
    else:
        print("[2/3] Generating raw lighting image...")
        generation_result = generate_image(
            pipeline=pipeline,
            prompts=prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            return_process=args.return_process,
            num_process_images=args.num_process_images,
        )

        if args.return_process:
            final_images, process_images = generation_result
            if not final_images:
                raise ValueError("image generation returned no results")
            raw_image = final_images[0]
            save_process_images(process_images, output_dir)
        else:
            raw_image = generation_result[0]

    raw_path = output_dir / "raw_image.png"
    raw_image.save(raw_path)
    print(f"Raw image saved to: {raw_path}")

    print("[3/3] Applying SDL gamut conversion...")
    gamut_colors = parse_sdl_file(Path(args.sdl_file))
    final_image = convert_image_to_sdl_with_dither(raw_path, gamut_colors)

    final_path = output_dir / "sdl_converted.png"
    final_image.save(final_path)
    print(f"Final image saved to: {final_path}")

    evaluation_report: dict[str, object] | None = None
    if args.enable_vlm_eval:
        print("[4/4] Scoring image quality with multimodal evaluation...")
        objective_payload = build_evaluation_payload(
            scene_text=scene_text,
            prompt=prompt,
            raw_image_path=raw_path,
            final_image_path=final_path,
            gamut_colors=gamut_colors,
        )
        eval_path = output_dir / "evaluation_report.json"
        try:
            evaluation_report = call_vlm_to_score_quality(
                raw_image_path=raw_path,
                final_image_path=final_path,
                scene_text=scene_text,
                prompt=prompt,
                objective_payload=objective_payload,
            )
            payload_to_save = {
                "objective": objective_payload,
                "vlm": evaluation_report,
            }
            eval_path.write_text(json.dumps(payload_to_save, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Evaluation report saved to: {eval_path}")
            print(f"Lighting score: {evaluation_report.get('lighting_plausibility_score')}")
            print(f"Gamut score: {evaluation_report.get('gamut_adherence_score')}")
            print(f"Overall score: {evaluation_report.get('overall_score')}")
        except Exception as exc:
            print(f"VLM evaluation skipped or failed: {exc}")
            evaluation_report = None

    return raw_path, final_path, prompt, evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated pipeline: scene description -> prompt -> raw image -> SDL-constrained image",
    )
    parser.add_argument("scene", nargs="+", help="Scene description text")
    parser.add_argument(
        "--base-model",
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model ID.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        # default="../../runs/lora/checkpoints/final",
        default="../../runs/full_color_high_freq_text/checkpoints/final",
        help="Path to the fine-tuned checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["full", "lora"],
        default="full",
        help="Checkpoint type to load.",
    )
    parser.add_argument(
        "--sampler",
        choices=["ddim", "dpm", "dpmpp", "euler", "heun", "unipc"],
        default="dpm",
        help="Sampler / scheduler to use during generation.",
    )
    parser.add_argument(
        "--sdl-file",
        default="../../data/SDL2_0.txt",
        help="Path to the SDL gamut boundary file.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../result",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Number of denoising steps used by the image generator.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--return-process",
        action="store_true",
        help="Save intermediate denoising process images.",
    )
    parser.add_argument(
        "--num-process-images",
        type=int,
        default=8,
        help="Number of intermediate process images to save when --return-process is enabled.",
    )
    parser.add_argument(
        "--use-candidate-selection",
        action="store_true",
        help="Generate multiple candidate images and let a VLM select the best one.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="Number of candidate images to generate when --use-candidate-selection is enabled.",
    )
    parser.add_argument(
        "--enable-vlm-eval",
        action="store_true",
        help="Run a multimodal evaluation step after generation and SDL conversion.",
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()