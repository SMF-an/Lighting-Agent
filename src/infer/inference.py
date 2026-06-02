import argparse
from pathlib import Path
import torch
import os
import textwrap
import matplotlib.pyplot as plt
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
)
from peft import PeftModel
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


load_dotenv()


def download_hf_checkpoint(repo_id: str, token: str | None = None) -> Path:
    """Download a model repo from Hugging Face and return local path.

    This uses `snapshot_download` to pull the model repo into the HF cache
    and returns the path to the downloaded repository root.
    """
    try:
        local_path = snapshot_download(repo_id, repo_type="model", token=token)
        return Path(local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download HF repo {repo_id}: {e}")


def load_full_checkpoint(base_model_id, checkpoint_path, device="cuda"):
    """
    Load SD1.5 with a full fine-tuned UNet checkpoint.

    Args:
        base_model_id: Base SD1.5 model ID
        checkpoint_path: Path to the checkpoint directory
        device: Device to load model on

    Returns:
        Loaded pipeline ready for inference
    """
    print(f"Loading base model: {base_model_id}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        variant="fp16" if device == "cuda" else None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    checkpoint_dir = Path(checkpoint_path)
    model_path = checkpoint_dir / "model" / "unet_state_dict.pt"

    print(f"Loading checkpoint from: {checkpoint_dir}")

    state_dict = torch.load(model_path, map_location=device)
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(device)
    print("Model loaded successfully!")

    return pipeline


def load_lora_checkpoint(base_model_id, checkpoint_path, device="cuda"):
    """
    Load SD1.5 with fine-tuned LoRA weights from a checkpoint directory.
    
    Args:
        base_model_id: Base SD1.5 model ID
        checkpoint_path: Path to the checkpoint directory
        device: Device to load model on
    
    Returns:
        Loaded pipeline ready for inference
    """
    print(f"Loading base model: {base_model_id}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        variant="fp16" if device == "cuda" else None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    checkpoint_dir = Path(checkpoint_path)
    adapter_dir = checkpoint_dir / "lora_adapter"
    trainer_state_path = checkpoint_dir / "trainer_state.pt"

    print(f"Loading checkpoint from: {checkpoint_dir}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_dir}")
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer state file: {trainer_state_path}")

    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, adapter_dir)

    trainer_state = torch.load(trainer_state_path, map_location=device)
    step = trainer_state.get("step")
    config = trainer_state.get("config", {}) or {}
    if step is not None:
        print(f"Restored checkpoint step: {step}")
    if config:
        print(f"Restored checkpoint config: {config}")
    
    pipeline = pipeline.to(device)
    print("Model loaded successfully!")
    
    return pipeline


def build_scheduler(base_scheduler, sampler_name: str):
    sampler_name = sampler_name.lower()
    scheduler_map = {
        "ddim": DDIMScheduler,
        "dpm": DPMSolverMultistepScheduler,
        "dpmpp": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "heun": HeunDiscreteScheduler,
        "unipc": UniPCMultistepScheduler,
    }
    if sampler_name not in scheduler_map:
        valid = ", ".join(sorted(scheduler_map.keys()))
        raise ValueError(f"Unknown sampler '{sampler_name}'. Valid options: {valid}")
    scheduler_cls = scheduler_map[sampler_name]
    return scheduler_cls.from_config(base_scheduler.config)


def load_pipeline(base_model: str, checkpoint_dir: str, checkpoint_type: str, device: str, sampler: str = "ddim"):
    if checkpoint_type == "pretrain":
        print(f"Loading pre-trained base model only: {base_model}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            variant="fp16" if device == "cuda" else None,
        )
        pipeline.scheduler = build_scheduler(pipeline.scheduler, sampler)
        return pipeline.to(device)
    if checkpoint_type == "full":
        loader = load_full_checkpoint
    elif checkpoint_type == "lora":
        loader = load_lora_checkpoint
    else:
        raise ValueError("checkpoint_type must be one of: pretrain, full, lora")

    pipeline = loader(base_model_id=base_model, checkpoint_path=checkpoint_dir, device=device)
    pipeline.scheduler = build_scheduler(pipeline.scheduler, sampler)
    return pipeline


def _decode_latents_to_pil_images(pipeline, latents):
    latents = latents.detach().to(dtype=pipeline.vae.dtype)
    latents = latents / pipeline.vae.config.scaling_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return [Image.fromarray((image * 255).round().astype("uint8")) for image in images]


def save_samples(pairs, output_dir):
    if not pairs:
        return

    cols = min(len(pairs), 4)
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[axis] for axis in axes]

    flat_axes = [axis for row in axes for axis in row]
    for axis in flat_axes:
        axis.axis("off")

    for index, (image, prompt) in enumerate(pairs):
        row = index // cols
        col = index % cols
        axis = axes[row][col]
        axis.imshow(image)
        short_title = "\n".join(textwrap.wrap(prompt, width=40))
        axis.set_title(short_title, fontsize=9)
        axis.axis("off")

    plt.tight_layout()
    grid_path = output_dir / "samples.png"
    plt.savefig(grid_path, dpi=200)
    plt.close(fig)


def save_process_images(process_images, output_dir):
    if not process_images:
        return

    row_images = []
    max_cols = 0
    for prompt_steps in process_images:
        if not prompt_steps:
            row_images.append([])
            continue

        step_images = []
        for step_entry in prompt_steps:
            if isinstance(step_entry, tuple) and len(step_entry) == 2:
                _, images = step_entry
            else:
                images = step_entry
            step_images.append(images[0] if isinstance(images, list) else images)
        row_images.append(step_images)
        max_cols = max(max_cols, len(step_images))

    row_images = [images for images in row_images if images]
    if not row_images:
        return

    rows = len(row_images)
    cols = max_cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[axis] for axis in axes]

    for row_axes, prompt_steps in zip(axes, process_images):
        for axis in row_axes:
            axis.axis("off")

        for col_index, step_entry in enumerate(prompt_steps):
            if isinstance(step_entry, tuple) and len(step_entry) == 2:
                step_number, images = step_entry
            else:
                step_number, images = col_index + 1, step_entry

            image = images[0] if isinstance(images, list) else images
            axis = row_axes[col_index]
            axis.imshow(image)
            axis.set_title(f"step {step_number}", fontsize=9)
            axis.axis("off")

    plt.tight_layout()
    save_path = Path(output_dir) / "process_images.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def generate_image(
    pipeline,
    prompts=None,
    negative_prompt="furniture, sofa, chair, table, desk, room, indoor, objects, scene",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=None,
    return_process=False,
    num_process_images=8,
):
    """
    Generate images using the full fine-tuned SD1.5 model.
    """
    if seed is not None:
        torch.manual_seed(seed)

    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompts:
        raise ValueError("prompts must be a non-empty string or a non-empty list of strings")
    if num_process_images < 0:
        raise ValueError("num_process_images must be >= 0")

    process_step_indices = set()
    if return_process and num_process_images > 0:
        if num_process_images == 1:
            process_step_indices = {num_inference_steps - 1}
        else:
            process_step_indices = {
                round(i * (num_inference_steps - 1) / (num_process_images - 1))
                for i in range(num_process_images)
            }

    print(f"\nGenerating {len(prompts)} image(s)")

    process_images = []
    final_images = []

    with torch.no_grad():
        for single_prompt in prompts:
            prompt_process_images = []

            def step_callback(pipeline, step, timestep, callback_kwargs):
                if return_process and "latents" in callback_kwargs and step in process_step_indices:
                    step_images = _decode_latents_to_pil_images(pipeline, callback_kwargs["latents"])
                    prompt_process_images.append((step + 1, step_images))
                return callback_kwargs

            result = pipeline(
                prompt=single_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
                num_images_per_prompt=1,
                callback_on_step_end=step_callback if return_process else None,
                callback_on_step_end_tensor_inputs=["latents"] if return_process else None,
            )

            final_images.extend(result.images)
            if return_process:
                process_images.append(prompt_process_images)

    if return_process:
        return final_images, process_images
    return final_images


def main():
    parser = argparse.ArgumentParser(description="Generate images using a full fine-tuned SD1.5 model.")
    parser.add_argument(
        "--base-model",
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion v1.5 model ID.",
    )
    parser.add_argument(
        "--use-hf-model",
        action="store_true",
        help="Load model from Hugging Face Hub.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="../../runs/full_20260528_133401/checkpoints/final",
        help="Path to a checkpoint directory.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="theavenger/light-effect-generator",
        help="Optional Hugging Face model repo id to download checkpoint from (e.g. 'username/repo').",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["pretrain", "full", "lora"],
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
        "--num-steps",
        type=int,
        default=50,
        help="Number of inference steps.",
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
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-process-images",
        type=int,
        default=8,
        help="Number of intermediate process images to return per generated image.",
    )
    parser.add_argument(
        "--return-process",
        action="store_true",
        help="Save intermediate denoising process images.",
    )
    parser.add_argument(
        "--output",
        default="../../result",
        help="Output path for generated image.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_hf_model:
        hf_token = os.getenv("HF_TOKEN")
        print(f"Downloading checkpoint from Hugging Face repo: {args.hf_repo_id} ...")
        downloaded = download_hf_checkpoint(args.hf_repo_id, token=hf_token)
        print(f"Downloaded HF repo to: {downloaded}")
        checkpoint_dir_arg = str(downloaded)
    else:
        checkpoint_dir_arg = args.checkpoint_dir

    pipeline = load_pipeline(
        base_model=args.base_model,
        checkpoint_dir=checkpoint_dir_arg,
        checkpoint_type=args.checkpoint_type,
        device=device,
        sampler=args.sampler,
    )

    prompts = [
        "Abstract light effect, uniform from purple to bright yellow-green, energetic atmosphere, smooth texture, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft blue to lime green, calm atmosphere, diffuse light, no room, no furniture, no objects.",
        "Abstract light effect, uniform from bright teal to cool blue, balanced atmosphere, diffuse light, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft teal to gentle cyan, balanced atmosphere, smooth texture, no room, no furniture, no objects.",
        "Abstract light effect, uniform from light gray to light periwinkle, warm atmosphere, subtle glow, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from cool teal to gentle yellow, energetic atmosphere, diffuse glow, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from light periwinkle to bright teal, fresh atmosphere, soft blend, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft purple to warm gold, fresh atmosphere, diffuse light, no room, no furniture, no objects.",
    ]
    if args.return_process:
        final_images, process_images = generate_image(
            pipeline=pipeline,
            prompts=prompts,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            return_process=True,
            num_process_images=args.num_process_images,
        )
    else:
        final_images = generate_image(
            pipeline=pipeline,
            prompts=prompts,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_samples(list(zip(final_images, prompts)), output_dir)

    if args.return_process:
        save_process_images(process_images, output_dir)

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()