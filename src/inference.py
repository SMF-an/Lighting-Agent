import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import textwrap

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel
from PIL import Image


def load_checkpoint(base_model_id, checkpoint_path, device="cuda"):
    """
    Load SDXL with fine-tuned LoRA weights from a checkpoint directory.
    
    Args:
        base_model_id: Base SDXL model ID
        checkpoint_path: Path to the checkpoint directory
        device: Device to load model on
    
    Returns:
        Loaded pipeline ready for inference
    """
    print(f"Loading base model: {base_model_id}")
    pipeline = DiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    )
    
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
    grid_path = output_dir / f"samples.png"
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

        step_images = [step[0] if isinstance(step, list) else step for step in prompt_steps]
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

    for row_axes, images in zip(axes, row_images):
        for axis in row_axes:
            axis.axis("off")

        for col_index, image in enumerate(images):
            axis = row_axes[col_index]
            axis.imshow(image)
            axis.set_title(f"step {col_index + 1}", fontsize=9)
            axis.axis("off")

    plt.tight_layout()
    save_path = Path(output_dir) / "process_images.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def generate_image(
    pipeline,
    prompts=None,
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=None,
    return_process=False,
    num_process_images=8,
):
    """
    Generate an image using the fine-tuned SDXL model.
    
    Args:
        pipeline: Loaded diffusion pipeline
        prompts: Text prompt(s) for generation
        negative_prompt: Negative prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        return_process: Whether to return intermediate process images
        num_process_images: Number of intermediate process images to return per generated image
    
    Returns:
        Final PIL image(s), or a tuple of (final image(s), process image list) when return_process is True
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
                    prompt_process_images.append(step_images)
                return callback_kwargs

            result = pipeline(
                prompt=single_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=768,
                width=768,
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
    parser = argparse.ArgumentParser(description="Generate images using fine-tuned SDXL LoRA model.")
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model ID.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./sdxl_lora_weights/final",
        help="Path to a checkpoint directory.",
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
        "--output",
        default="result",
        help="Output path for generated image.",
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = load_checkpoint(
        base_model_id=args.base_model,
        checkpoint_path=args.checkpoint_dir,
        device=device,
    )
    
    prompts = [
        "A serene landscape with a river flowing through a forest, in the style of Studio Ghibli",
        "A futuristic cityscape at night with neon lights, in the style of Syd Mead",
        "A cozy cottage in the woods during autumn, in the style of Thomas Kinkade",
    ]
    negative_prompt = "low quality, blurry, bad anatomy, disfigured, deformed, extra limbs, close up, cropped, worst quality"
    if args.return_process:
        final_images, process_images = generate_image(
            pipeline=pipeline,
            prompts=prompts,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            return_process=True,
            num_process_images=args.num_process_images
        )
    else:
        final_images = generate_image(
            pipeline=pipeline,
            prompts=prompts,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_samples(list(zip(final_images, prompts)), output_dir)
    
    if args.return_process:
        save_process_images(process_images, output_dir)

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
