import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import textwrap

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, StableDiffusionPipeline
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from data.dataset import build_train_dataloader


def setup_sd15_lora(base_model_id: str, lora_rank: int = 32, lora_alpha: int = 32, lora_dropout: float = 0.1):
    """
    Load Stable Diffusion v1.5 and apply LoRA to the UNet.
    """
    print(f"Loading base model (SD1.5 LoRA): {base_model_id}")

    train_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=train_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        variant="fp16" if train_dtype == torch.float16 else None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_v", "to_q"],
        lora_dropout=lora_dropout,
        bias="none",
    )

    unet = get_peft_model(pipeline.unet, lora_config)
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


def generate_samples(pipeline, prompts, negative_prompt="furniture, sofa, chair, table, desk, room, indoor, objects, scene", num_inference_steps=25, guidance_scale=7.0):
    generated_images = []
    for prompt in prompts:
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
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

    loaded_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
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
        raise KeyError("Batch is missing 'captions'. The dataset/collate function must return raw captions.")
    return pixel_values, captions


def encode_captions(pipeline, captions, device):
    text_inputs = pipeline.tokenizer(
        captions,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = pipeline.text_encoder(input_ids)[0]

    return prompt_embeds


def training_loop(
    pipeline,
    train_dataloader,
    num_epochs=10,
    learning_rate=1e-4,
    gradient_accumulation_steps=1,
    sample_every_steps=1000,
    checkpoint_every_steps=1000,
    sample_prompts=None,
    resume_from_checkpoint_path: Optional[str] = None,
    output_dir="../runs",
):
    """
    Main training loop for SD1.5 LoRA fine-tuning.
    """
    if sample_prompts is None:
        sample_prompts = [
            "Abstract light effect, uniform from purple to bright yellow-green, energetic atmosphere, smooth texture, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft blue to lime green, calm atmosphere, diffuse light, no room, no furniture, no objects.",
        "Abstract light effect, uniform from bright teal to cool blue, balanced atmosphere, diffuse light, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft teal to gentle cyan, balanced atmosphere, smooth texture, no room, no furniture, no objects.",
        "Abstract light effect, uniform from light gray to light periwinkle, warm atmosphere, subtle glow, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from cool teal to gentle yellow, energetic atmosphere, diffuse glow, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from light periwinkle to bright teal, fresh atmosphere, soft blend, no room, no furniture, no objects.",
        "Abstract light effect, smooth blend from soft purple to warm gold, fresh atmosphere, diffuse light, no room, no furniture, no objects.",
        ]

    trainable_params = [param for param in pipeline.unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.unet.to(dtype=amp_dtype if use_amp else torch.float32)
    pipeline.vae.to(device=device, dtype=torch.float32)
    pipeline.vae.eval()
    pipeline.text_encoder.eval()

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

            prompt_embeds = encode_captions(pipeline, captions, device)
            prompt_embeds = prompt_embeds.to(dtype=pipeline.unet.dtype)

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

            with torch.autocast(device_type=device, dtype=amp_dtype if use_amp else torch.float32, enabled=use_amp):
                model_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at step {global_step}. Try lowering the learning rate or batch size."
                )

            scaler.scale(loss / gradient_accumulation_steps).backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
                restore_vae_dtype = pipeline.vae.dtype
                pipeline.vae.to(device=device, dtype=pipeline.unet.dtype)
                try:
                    sample_images = generate_samples(pipeline=pipeline, prompts=sample_prompts)
                    save_samples(images=sample_images, output_dir=output_dir, global_step=global_step)
                finally:
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
                        "lora_rank": getattr(pipeline.unet.peft_config["default"], "r", None) if hasattr(pipeline.unet, "peft_config") else None,
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
            "lora_rank": getattr(pipeline.unet.peft_config["default"], "r", None) if hasattr(pipeline.unet, "peft_config") else None,
        },
    )
    save_loss_curve(loss_history, output_dir)
    progress_bar.close()
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion v1.5 LoRA fine-tuning for gradient lighting effects.")
    parser.add_argument(
        "--base-model",
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion v1.5 model ID on HuggingFace Hub.",
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
        default="../../data/light_effect_captions.jsonl",
        help="Local JSONL dataset path (used if --use-hf-dataset not set).",
    )
    parser.add_argument(
        "--image-root",
        default="../../data",
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
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../runs",
        help="Output directory for trained LoRA weights.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size for training.",
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

    pipeline = setup_sd15_lora(
        args.base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

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
    output_dir = Path(args.output_dir) / f"lora_{start_time}"
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
