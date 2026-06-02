import importlib
import tempfile
from functools import lru_cache
import html
from pathlib import Path
import argparse
import random
import gradio as gr
import torch
from PIL import Image

from pipeline_infer import (
    call_vlm_to_select_best_candidate,
    build_evaluation_payload,
    call_vlm_to_score_quality,
)


@lru_cache(maxsize=1)
def _inference_module():
    return importlib.import_module("inference")


@lru_cache(maxsize=1)
def _prompt_translator_module():
    return importlib.import_module("prompt_translator")


@lru_cache(maxsize=1)
def _sdl_converter_module():
    return importlib.import_module("sdl_converter")


def _flatten_process_images(process_images):
    if not process_images:
        return []

    flattened = []
    for prompt_steps in process_images:
        for step_entry in prompt_steps or []:
            if isinstance(step_entry, tuple) and len(step_entry) == 2:
                _, images = step_entry
            else:
                images = step_entry

            if isinstance(images, list):
                flattened.extend(images)
            else:
                flattened.append(images)
    return flattened


# Note: configuration values are passed into build_demo() and then into
# run_web_pipeline() via a small wrapper to avoid module-level globals.


@lru_cache(maxsize=8)
def _get_pipeline(base_model: str, checkpoint_dir: str, checkpoint_type: str, device: str, sampler: str):
    return _inference_module().load_pipeline(
        base_model=base_model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_type=checkpoint_type,
        device=device,
        sampler=sampler,
    )


def run_web_pipeline(
    scene_text: str,
    num_steps: int,
    guidance_scale: float,
    sampler: str,
    show_process: bool,
    use_candidate_selection: bool,
    num_candidates: int,
    base_model: str,
    checkpoint_dir: str,
    checkpoint_type: str,
    sdl_file: str,
    output_dir: str,
    enable_vlm_eval: bool = False,
):
    scene_text = (scene_text or "").strip()
    if not scene_text:
        raise gr.Error("请输入场景描述。")

    sdl_path = Path(sdl_file).expanduser()
    if not sdl_path.exists():
        raise gr.Error(f"SDL 文件不存在: {sdl_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory(prefix="light-effect-web-") as tmp_dir:
        output_dir = Path(tmp_dir)

        prompt = _prompt_translator_module().call_llm_for_effect(scene_text)

        pipeline = _get_pipeline(base_model, checkpoint_dir, checkpoint_type, device, sampler)

        process_gallery = []
        candidate_gallery = []
        if use_candidate_selection:
            candidate_count = max(1, num_candidates)
            candidate_dir = output_dir / "candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            seed_values = [random.randint(0, 2**31 - 1) for _ in range(candidate_count)]
            candidate_paths: list[Path] = []
            candidate_images: list[Image.Image] = []
            candidate_process_galleries: list[list[object]] = []

            for index, seed_value in enumerate(seed_values, start=1):
                candidate_result = _inference_module().generate_image(
                    pipeline=pipeline,
                    prompts=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=seed_value,
                    return_process=show_process,
                )

                if show_process:
                    candidate_final_images, process_images = candidate_result
                    if not candidate_final_images:
                        raise gr.Error("模型没有返回生成结果。")
                    candidate_image = candidate_final_images[0]
                    candidate_process_galleries.append(_flatten_process_images(process_images))
                else:
                    candidate_final_images = candidate_result
                    if not candidate_final_images:
                        raise gr.Error("模型没有返回生成结果。")
                    candidate_image = candidate_final_images[0]
                    candidate_process_galleries.append([])

                candidate_path = candidate_dir / f"candidate_{index:02d}.png"
                candidate_image.save(candidate_path)
                candidate_paths.append(candidate_path)
                candidate_images.append(candidate_image)

            selected_index, reason = call_vlm_to_select_best_candidate(
                candidate_paths,
                scene_text=scene_text,
                prompt=prompt,
            )
            raw_path = candidate_paths[selected_index]
            raw_image = Image.open(raw_path).convert("RGB")
            if reason:
                print(f"VLM reason: {reason}")
            print(f"Selected candidate #{selected_index + 1}: {raw_path.name}")
            process_gallery = candidate_process_galleries[selected_index]
            candidate_gallery = candidate_images
        else:
            generation_result = _inference_module().generate_image(
                pipeline=pipeline,
                prompts=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=None,
                return_process=show_process,
            )

            if show_process:
                final_images, process_images = generation_result
                if not final_images:
                    raise gr.Error("模型没有返回生成结果。")
                raw_image = final_images[0]
                process_gallery = _flatten_process_images(process_images)
            else:
                final_images = generation_result
                if not final_images:
                    raise gr.Error("模型没有返回生成结果。")
                raw_image = final_images[0]

        raw_path = output_dir / "raw_image.png"
        raw_image.save(raw_path)

        gamut_colors = _sdl_converter_module().parse_sdl_file(sdl_path)
        final_image = _sdl_converter_module().convert_image_to_sdl_with_dither(raw_path, gamut_colors)
        final_path = output_dir / "sdl_converted.png"
        final_image.save(final_path)

        evaluation_report = None
        if enable_vlm_eval:
            try:
                gamut_colors = _sdl_converter_module().parse_sdl_file(sdl_path)
                objective_payload = build_evaluation_payload(
                    scene_text=scene_text,
                    prompt=prompt,
                    raw_image_path=raw_path,
                    final_image_path=final_path,
                    gamut_colors=gamut_colors,
                )
                evaluation_report = call_vlm_to_score_quality(
                    raw_image_path=raw_path,
                    final_image_path=final_path,
                    scene_text=scene_text,
                    prompt=prompt,
                    objective_payload=objective_payload,
                )
            except Exception as exc:
                evaluation_report = {"error": str(exc)}

        return prompt, raw_image, final_image, candidate_gallery, process_gallery, evaluation_report


def build_demo(base_model, checkpoint_dir, sdl_file, checkpoint_type, output_dir):
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #7c2d12 100%);
    }
    .main-panel {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 20px;
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(14px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.28);
    }
    .title-block h1 {
        font-size: 2.2rem !important;
        line-height: 1.1;
        margin-bottom: 0.25rem !important;
        color: #f8fafc;
        text-shadow: 0 2px 10px rgba(0,0,0,0.45);
    }
    .title-block p {
        color: rgba(241,245,249,0.92);
        font-size: 1rem;
        max-width: 58rem;
        text-shadow: 0 1px 6px rgba(0,0,0,0.35);
    }
    """

    with gr.Blocks(css=css, title="Light-Effect Generation Pipeline Demo") as demo:
        gr.Markdown(
            """
            <div class="title-block">
            <h1>Light-Effect Generation Pipeline Demo</h1>
            <p>输入中文场景描述，系统会先翻译成控制 prompt，再生成原始图像并做 SDL 色域转换。</p>
            </div>
            """
        )

        with gr.Row(elem_classes=["main-panel"]):
            # Left: inputs
            with gr.Column(scale=5):
                scene_text = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：酒店套房客厅，浪漫氛围。",
                    lines=6,
                )

                # Sampling controls stacked for clearer layout
                num_steps = gr.Slider(label="采样步数", minimum=1, maximum=100, value=30, step=1)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
                sampler = gr.Dropdown(
                    label="采样器",
                    choices=["ddim", "dpm", "dpmpp", "euler", "heun", "unipc"],
                    value="dpm",
                )
                show_process = gr.Checkbox(label="显示中间去噪过程", value=False)
                use_candidate_selection = gr.Checkbox(label="候选图选择模式", value=False)
                enable_vlm_eval = gr.Checkbox(label="启用大模型评估", value=False)
                num_candidates = gr.Slider(label="候选图数量", minimum=1, maximum=8, value=4, step=1)

                # Primary action
                run_btn = gr.Button("生成", variant="primary")

            # Right: outputs
            with gr.Column(scale=4):
                translated_prompt = gr.Textbox(label="翻译后的 Prompt", lines=4)
                with gr.Row():
                    raw_image = gr.Image(label="Raw Image", type="pil")
                    final_image = gr.Image(label="SDL Converted", type="pil")
                eval_card = gr.HTML("<div style='color:#e6eef8'>评估结果将在此显示。</div>")

        # Full-width Process Images gallery at the bottom
        with gr.Row():
            with gr.Column():
                candidate_gallery = gr.Gallery(label="Candidate Images", columns=4, height=320)

        # Full-width Process Images gallery at the bottom
        with gr.Row():
            with gr.Column():
                process_gallery = gr.Gallery(label="Process Images", columns=8, height=360)

        # Wrap run_web_pipeline to bind the configured model/checkpoint/SDL/output
        def format_eval_html(report: dict | None) -> str:
            if not report:
                return "<div style='color:#cbd5e1'>评估未开启或无结果。</div>"
            if isinstance(report, dict) and report.get("error"):
                return f"<div style='color:#f87171'><strong>评估出错：</strong>{html.escape(str(report.get('error')))}</div>"

            # Extract scores and reasons safely
            lighting = report.get("lighting_plausibility_score")
            gamut = report.get("gamut_adherence_score")
            overall = report.get("overall_score")
            lighting_reason = report.get("lighting_reason") or report.get("lighting_reason", "")
            gamut_reason = report.get("gamut_reason") or report.get("gamut_reason", "")
            verdict = report.get("verdict") or ""

            def score_to_percent(s):
                try:
                    return max(0, min(100, float(s) * 10))
                except Exception:
                    return 0

            lighting_pct = score_to_percent(lighting)
            gamut_pct = score_to_percent(gamut)
            overall_pct = score_to_percent(overall)

            def bar_html(label, score, pct, color):
                return (
                    f"<div style='margin-bottom:8px'><div style='font-weight:700;color:#f8fafc'>{label}: "
                    f"{html.escape(str(score)) if score is not None else 'N/A'}</div>"
                    f"<div style='background:#020617;border-radius:6px;padding:3px;margin-top:4px;border:1px solid rgba(148,163,184,0.35)'>"
                    f"<div style='width:{pct}%;background:{color};height:12px;border-radius:4px'></div>"
                    f"</div></div>"
                )

            html_chunks = [
                "<div style='padding:12px;border-radius:12px;background:rgba(2,6,23,0.88);border:1px solid rgba(148,163,184,0.45);color:#f8fafc;box-shadow:0 8px 20px rgba(0,0,0,0.35)'>",
                f"<div style='font-size:1.1rem;font-weight:800;margin-bottom:6px;color:#f8fafc'>综合评分：{html.escape(str(verdict))}</div>",
                bar_html('光效合理性', lighting, lighting_pct, '#60a5fa'),
                bar_html('色域遵从度', gamut, gamut_pct, '#34d399'),
                bar_html('总体评分', overall, overall_pct, '#fbbf24'),
                "<div style='margin-top:10px;color:#e2e8f0'>",
                f"<div style='font-weight:700;color:#f8fafc'>光效理由</div><div style='margin-bottom:6px;color:#e2e8f0'>{html.escape(str(lighting_reason))}</div>",
                f"<div style='font-weight:700;color:#f8fafc'>色域理由</div><div style='color:#e2e8f0'>{html.escape(str(gamut_reason))}</div>",
                "</div></div>",
            ]

            return "".join(html_chunks)


        def _run(
            scene_text_in,
            num_steps_in,
            guidance_scale_in,
            sampler_in,
            show_process_in,
            use_candidate_selection_in,
            num_candidates_in,
            enable_vlm_eval_in,
        ):
            result = run_web_pipeline(
                scene_text_in,
                num_steps_in,
                guidance_scale_in,
                sampler_in,
                show_process_in,
                use_candidate_selection_in,
                int(num_candidates_in),
                base_model,
                checkpoint_dir,
                checkpoint_type,
                sdl_file,
                output_dir,
                enable_vlm_eval=enable_vlm_eval_in,
            )

            # Unpack and format evaluation
            try:
                prompt_out, raw_img_out, final_img_out, cand_gallery_out, proc_gallery_out, eval_report_out = result
            except Exception:
                # Fallback if older tuple shape
                prompt_out, raw_img_out, final_img_out, cand_gallery_out, proc_gallery_out = result
                eval_report_out = None

            eval_html = format_eval_html(eval_report_out)
            return prompt_out, raw_img_out, final_img_out, cand_gallery_out, proc_gallery_out, eval_html

        run_btn.click(
            fn=_run,
            inputs=[
                scene_text,
                num_steps,
                guidance_scale,
                sampler,
                show_process,
                use_candidate_selection,
                num_candidates,
                enable_vlm_eval,
            ],
            outputs=[translated_prompt, raw_image, final_image, candidate_gallery, process_gallery, eval_card],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Web demo for light-effect generation pipeline.")
    parser.add_argument(
        "--base-model",
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion v1.5 model ID.",
    )
    parser.add_argument(
        "--checkpoint-type",
        default="full",
        choices=["pretrain", "full", "lora"],
        help="Type of the checkpoint to load.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="../../runs/full_color_high_freq_text/checkpoints/final",
        help="Path to a checkpoint directory.",
    )
    parser.add_argument(
        "--sdl-file",
        default="../../data/SDL2_0.txt",
        help="Path to the SDL gamut boundary file.",
    )
    parser.add_argument(
        "--output",
        default="../../result",
        help="Output path for generated image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_demo(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        sdl_file=args.sdl_file,
        checkpoint_type=args.checkpoint_type,
        output_dir=args.output,
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()