import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from peft import get_peft_model, LoraConfig
import textwrap
import torch.nn as nn
from tqdm.auto import tqdm

from data.dataset import build_train_dataloader


def setup_sdxl_lora(base_model_id, lora_rank=32, lora_alpha=32, lora_dropout=0.1):
    """
    Load SDXL base model and apply LoRA configuration.
    
    Args:
        base_model_id: HuggingFace model ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout
    
    Returns:
        pipeline with LoRA-enabled UNet
    """
    print(f"Loading base model: {base_model_id}")
    
    pipeline = DiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_v", "to_q"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    unet = pipeline.unet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    pipeline.unet = unet
    
    return pipeline


def save_loss_curve(loss_history, output_dir):
    if not loss_history:
        return

    steps = [item[0] for item in loss_history]
    losses = [item[1] for item in loss_history]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1.5, label="loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()


def generate_samples(pipeline, prompts, num_inference_steps=25, guidance_scale=7.0):
    generated_images = []
    for prompt in prompts:
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=768,
                width=768,
            )

        generated_images.append((result.images[0], prompt))

    return generated_images


def save_samples(images, output_dir, global_step):
    if not images:
        return

    sample_dir = Path(output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    cols = min(len(images), 4)
    rows = (len(images) + cols - 1) // cols
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

    for index, (image, prompt) in enumerate(images):
        row = index // cols
        col = index % cols
        axis = axes[row][col]
        axis.imshow(image)
        short_title = "\n".join(textwrap.wrap(prompt, width=40))
        axis.set_title(short_title, fontsize=9)
        axis.axis("off")

    plt.tight_layout()
    grid_path = sample_dir / f"step_{global_step:07d}.png"
    plt.savefig(grid_path, dpi=200)
    plt.close(fig)


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
):
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir / "lora_adapter")

    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_dir / "trainer_state.pt")

    print(f"Saved checkpoint to {checkpoint_dir}")

    return checkpoint_dir


def resume_from_checkpoint(path, model, optimizer, device):
    checkpoint_dir = Path(path)
    adapter_dir = checkpoint_dir / "lora_adapter"
    trainer_state_path = checkpoint_dir / "trainer_state.pt"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_dir}")
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer state file: {trainer_state_path}")

    # Load LoRA adapter weights into the existing PEFT-wrapped UNet.
    from peft import PeftModel

    loaded_model = PeftModel.from_pretrained(model, adapter_dir)
    model.load_state_dict(loaded_model.state_dict(), strict=False)

    trainer_state = torch.load(trainer_state_path, map_location=device)
    optimizer.load_state_dict(trainer_state["optimizer"])
    _move_optimizer_state_to_device(optimizer, device)

    config = trainer_state.get("config", {}) or {}
    step = int(trainer_state.get("step", 0))
    epoch_index = int(config.get("epoch_index", 0))
    batch_idx = int(config.get("batch_idx", -1))

    print(f"Resumed checkpoint from {checkpoint_dir}")
    return {
        "step": step,
        "epoch_index": epoch_index,
        "batch_idx": batch_idx,
        "config": config,
    }


def prepare_training_batch(batch, device):
    pixel_values = batch["pixel_values"].to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
    captions = batch.get("captions")
    if captions is None:
        raise KeyError("Batch is missing 'captions'. The dataset/collate function must return raw captions for SDXL conditioning.")
    return pixel_values, captions


def build_sdxl_added_conditions(pipeline, captions, pixel_values, device):
    if not hasattr(pipeline, "encode_prompt"):
        raise AttributeError("The loaded pipeline does not provide encode_prompt, which is required for SDXL training.")

    batch_size, _, height, width = pixel_values.shape

    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=captions,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    add_time_ids = pipeline._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pooled_prompt_embeds.shape[-1],
    ).to(device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)

    return prompt_embeds, {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    }


def training_loop(
    pipeline,
    train_dataloader,
    num_epochs=10,
    learning_rate=1e-4,
    gradient_accumulation_steps=1,
    sample_every_steps=200,
    checkpoint_every_steps=500,
    sample_prompts=None,
    resume_from_checkpoint_path: Optional[str] = None,
    output_dir="../runs",
):
    """
    Main training loop for SDXL LoRA fine-tuning.
    """
    if sample_prompts is None:
        sample_prompts = [
            "Soft gradient lighting, warm sunset hues transitioning from amber to soft magenta, cozy and relaxing atmosphere, cinematic lighting, 8k resolution. ",
            "Soft gradient lighting transitions from light yellow to pale pink, relaxing and immersive atmosphere for the retail cosmetics testing area. ",
            "Warm and flowing light, soft gradient of yellow and light orange, intimate and solemn atmosphere, comfort of the dining space. ",
            "Bright and warm tones, pale yellow and light orange, fresh and invigorating atmosphere, focus and vitality of the office space. "
        ]

    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=learning_rate,
    )
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.vae.to(device=device, dtype=torch.float32)
    pipeline.vae.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    start_epoch_index = 0
    start_batch_idx = -1
    if resume_from_checkpoint_path:
        resume_state = resume_from_checkpoint(resume_from_checkpoint_path, pipeline.unet, optimizer, device)
        start_step = resume_state["step"]
        start_epoch_index = resume_state["epoch_index"]
        start_batch_idx = resume_state["batch_idx"]

    total_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, initial=start_step, desc="Training", dynamic_ncols=True)
    loss_history = []
    global_step = start_step
    
    for epoch in range(start_epoch_index, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_loss = 0.0
        optimizer.zero_grad()
        pipeline.unet.train()
        
        for step, batch in enumerate(train_dataloader):
            if epoch == start_epoch_index and step <= start_batch_idx:
                continue

            pixel_values, captions = prepare_training_batch(batch, device)
           
            with torch.no_grad():
                latents = pipeline.vae.encode(pixel_values.float()).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
                latents = latents.to(dtype=pipeline.unet.dtype)
            
            encoder_hidden_states, added_cond_kwargs = build_sdxl_added_conditions(
                pipeline=pipeline,
                captions=captions,
                pixel_values=pixel_values,
                device=device,
            )
            encoder_hidden_states = encoder_hidden_states.to(dtype=pipeline.unet.dtype)
            added_cond_kwargs = {
                key: value.to(dtype=pipeline.unet.dtype) if isinstance(value, torch.Tensor) else value
                for key, value in added_cond_kwargs.items()
            }
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                pipeline.scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
                model_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
            
            loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at step {global_step}. "
                    "Try lowering the learning rate, reducing image size, or keeping VAE encoding in float32."
                )
            
            (loss / gradient_accumulation_steps).backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            loss_value = loss.detach().item()
            epoch_loss += loss_value
            loss_history.append((global_step, loss_value))
            global_step += 1
            
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=f"{epoch + 1}/{num_epochs}", loss=f"{loss_value:.4f}")

            if sample_every_steps > 0 and global_step % sample_every_steps == 0:
                pipeline.unet.eval()
                # For sampling, SDXL decode expects VAE dtype aligned with inference latents.
                restore_vae_dtype = pipeline.vae.dtype
                pipeline.vae.to(device=device, dtype=pipeline.unet.dtype)
                try:
                    sample_images = generate_samples(pipeline=pipeline, prompts=sample_prompts)
                    save_samples(images=sample_images, output_dir=output_dir, global_step=global_step)
                finally:
                    # Keep VAE in float32 during training for numerical stability.
                    pipeline.vae.to(device=device, dtype=restore_vae_dtype)
                pipeline.unet.train()

            if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                save_checkpoint(
                    path=Path(output_dir) / "checkpoints" / f"step_{global_step:07d}",
                    model=pipeline.unet,
                    optimizer=optimizer,
                    step=global_step,
                    config={
                        "epoch": epoch + 1,
                        "epoch_index": epoch,
                        "batch_idx": step,
                        "loss_value": loss_value,
                        "learning_rate": learning_rate,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "sample_every_steps": sample_every_steps,
                        "checkpoint_every_steps": checkpoint_every_steps,
                    },
                )
        
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        

    save_checkpoint(
        path=Path(output_dir) / "checkpoints" / "final",
        model=pipeline.unet,
        optimizer=optimizer,
        step=global_step,
        config={
            "epoch": num_epochs,
            "loss_value": loss_history[-1][1] if loss_history else None,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "sample_every_steps": sample_every_steps,
            "checkpoint_every_steps": checkpoint_every_steps,
        },
    )
    save_loss_curve(loss_history, output_dir)
    progress_bar.close()
    print(f"\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="SDXL LoRA fine-tuning for gradient lighting effects.")
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model ID on HuggingFace Hub.",
    )
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        help="Load training data from Hugging Face dataset.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="theavenger/light-effect-dataset",
        help="HuggingFace dataset repo ID (required if --use-hf-dataset).",
    )
    parser.add_argument(
        "--jsonl-path",
        default="../data/light_effect_captions.jsonl",
        help="Local JSONL dataset path (used if --use-hf-dataset not set).",
    )
    parser.add_argument(
        "--image-root",
        default="../data",
        help="Local image root directory.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA training.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--output-dir",
        default="../runs",
        help="Output directory for trained LoRA weights.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Image size for training (SDXL typically uses 768 or 1024).",
    )
    parser.add_argument(
        "--sample-every-steps",
        type=int,
        default=1000,
        help="Run inference sampling every N optimization steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=1000,
        help="Save LoRA weights every N optimization steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--sample-prompt",
        action="append",
        default=None,
        help="Prompt used for periodic sampling. Can be repeated multiple times.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint directory created by save_checkpoint.",
    )
    args = parser.parse_args()
    
    pipeline = setup_sdxl_lora(args.base_model, lora_rank=args.lora_rank)

    train_dataloader = build_train_dataloader(
        use_hf_dataset=args.use_hf_dataset,
        hf_repo_id=args.hf_repo_id,
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"sdxl_lora_{start_time}"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_loop(
        pipeline=pipeline,
        train_dataloader=train_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=1,
        sample_every_steps=args.sample_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        sample_prompts=args.sample_prompt,
        resume_from_checkpoint_path=args.resume_from_checkpoint,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
