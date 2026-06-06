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
from tqdm.auto import tqdm

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from data.dataset import build_train_dataloader


def setup_sd15_full_tune(base_model_id: str):
    """
    Load Stable Diffusion v1.5 for full UNet fine-tuning.

    The UNet is trained, while the VAE and text encoder remain frozen.
    """
    print(f"Loading base model (full tune): {base_model_id}")

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


def generate_samples(pipeline, prompts, negative_prompt="furniture, sofa, chair, table, desk, room, indoor, objects, scene", num_inference_steps=30, guidance_scale=7.0):
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

    model_dir = checkpoint_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    torch.save(model_to_save.state_dict(), model_dir / "unet_state_dict.pt")

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
    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    model_path = checkpoint_dir / "model" / "unet_state_dict.pt"

    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer state file: {trainer_state_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing UNet state dict: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    target_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target_model.load_state_dict(state_dict, strict=True)

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


def apply_caption_dropout(captions, dropout_prob, blank_caption=""):
    if dropout_prob <= 0:
        return captions

    dropped_captions = []
    for caption in captions:
        if torch.rand(1).item() < dropout_prob:
            dropped_captions.append(blank_caption)
        else:
            dropped_captions.append(caption)
    return dropped_captions


def encode_captions(pipeline, captions, device, train_text_encoder=False):
    text_inputs = pipeline.tokenizer(
        captions,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    with torch.set_grad_enabled(train_text_encoder):
        prompt_embeds = pipeline.text_encoder(input_ids)[0]

    return prompt_embeds


def decode_pred_original_to_rgb(pipeline, noisy_latents, timesteps, model_pred):
    """Decode the predicted original sample (pred_original_sample) via the VAE to RGB [0,1].

    Handles different scheduler prediction types ('epsilon', 'sample', 'v_prediction').
    Returns decoded RGB tensor in range [0,1], shape Bx3xHxW.
    """
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)

    prediction_type = getattr(pipeline.scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_prod_t * model_pred) / sqrt_alpha_prod_t
    elif prediction_type == "sample":
        pred_original_sample = model_pred
    elif prediction_type == "v_prediction":
        pred_original_sample = sqrt_alpha_prod_t * noisy_latents - sqrt_one_minus_alpha_prod_t * model_pred
    else:
        raise ValueError(f"Unsupported prediction type for decoding: {prediction_type}")

    decoded = pipeline.vae.decode(
        (pred_original_sample / pipeline.vae.config.scaling_factor).to(dtype=pipeline.vae.dtype),
        return_dict=False,
    )[0]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    return decoded


def rgb_to_lab(rgb):
    """Convert sRGB in [0, 1] to CIE Lab."""
    rgb = rgb.float().clamp(0, 1)

    linear_rgb = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )

    r = linear_rgb[:, 0]
    g = linear_rgb[:, 1]
    b = linear_rgb[:, 2]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    x = x / 0.95047
    y = y / 1.0
    z = z / 1.08883

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t):
        return torch.where(t > epsilon, t.clamp_min(1e-8).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([l, a, b], dim=1)


def compute_highfreq_loss(pipeline, noisy_latents, timesteps, model_pred, cutoff=0.15):
    decoded = decode_pred_original_to_rgb(pipeline, noisy_latents, timesteps, model_pred)

    luminance = 0.299 * decoded[:, 0] + 0.587 * decoded[:, 1] + 0.114 * decoded[:, 2]
    luminance = torch.nn.functional.interpolate(
        luminance.unsqueeze(1), size=(128, 128), mode="bilinear", align_corners=False
    ).squeeze(1)

    Fk = torch.fft.rfftn(luminance, dim=(1, 2))
    mag2 = (Fk.real ** 2 + Fk.imag ** 2)

    H, Wp = mag2.shape[1], mag2.shape[2]

    # Build a cross-shaped mask: keep the horizontal and vertical center bands,
    # filter everything outside the cross.
    fy = torch.fft.fftfreq(H, device=mag2.device).abs().unsqueeze(1)  # H x 1
    fx = torch.fft.rfftfreq(luminance.shape[2], device=mag2.device).unsqueeze(0)  # 1 x Wp

    row_band = fy <= float(cutoff)
    col_band = fx <= float(cutoff)
    mask = torch.logical_or(row_band, col_band).float()  # H x Wp
    mask = mask.unsqueeze(0)  # 1 x H x Wp

    # Penalize energy outside the cross, so the loss is large when off-axis frequencies dominate.
    outside_mask = 1.0 - mask
    hf_energy = (mag2 * outside_mask).sum(dim=(1, 2))
    total_energy = mag2.sum(dim=(1, 2))
    return (hf_energy / (total_energy + 1e-8)).mean()


def compute_color_loss(
    pipeline,
    noisy_latents,
    timesteps,
    model_pred,
    chroma_threshold=18.0,
    low_sat_threshold=0.20,
    low_sat_target=0.15,
):
    """Encourage richer colors without forcing oversaturation."""
    decoded = decode_pred_original_to_rgb(pipeline, noisy_latents, timesteps, model_pred)
    decoded = torch.nn.functional.interpolate(
        decoded.float(),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )

    lab = rgb_to_lab(decoded)
    chroma = torch.sqrt(lab[:, 1].pow(2) + lab[:, 2].pow(2) + 1e-8)
    mean_chroma = chroma.mean(dim=(1, 2))
    chroma_loss = torch.relu(1.0 - mean_chroma / (chroma_threshold + 1e-8))

    max_rgb = decoded.max(dim=1).values
    min_rgb = decoded.min(dim=1).values
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
    low_sat_frac = (saturation < low_sat_threshold).float().mean(dim=(1, 2))
    low_sat_loss = torch.relu(low_sat_frac - low_sat_target)

    return (chroma_loss + low_sat_loss).mean()


def training_loop(
    pipeline,
    train_dataloader,
    num_epochs=8,
    learning_rate=5e-6,
    text_encoder_learning_rate=1e-6,
    gradient_accumulation_steps=1,
    sample_every_steps=1000,
    checkpoint_every_steps=1000,
    resume_from_checkpoint_path: Optional[str] = None,
    output_dir="../runs",
    train_text_encoder: bool = True,
    caption_dropout: float = 0.1,
    use_highfreq_loss: bool = False,
    highfreq_weight: float = 0.01,
    highfreq_cutoff: float = 0.15,
    use_color_loss: bool = False,
    color_loss_weight: float = 0.002,
    color_chroma_threshold: float = 18.0,
    color_low_sat_threshold: float = 0.20,
    color_low_sat_target: float = 0.15,
):
    """
    Main training loop for SD1.5 full UNet fine-tuning.
    """
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

    optimizer_param_groups = [
        {
            "params": pipeline.unet.parameters(),
            "lr": learning_rate,
        }
    ]
    if train_text_encoder:
        optimizer_param_groups.append(
            {
                "params": pipeline.text_encoder.parameters(),
                "lr": text_encoder_learning_rate,
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(train_text_encoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.unet.to(dtype=amp_dtype if use_amp else torch.float32)
    pipeline.vae.to(device=device, dtype=torch.float32)
    pipeline.vae.eval()
    if train_text_encoder:
        pipeline.text_encoder.train()
    else:
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
            captions = apply_caption_dropout(captions, caption_dropout, blank_caption="")

            with torch.no_grad():
                latents = pipeline.vae.encode(pixel_values.float()).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
                latents = latents.to(dtype=pipeline.unet.dtype)

            prompt_embeds = encode_captions(
                pipeline,
                captions,
                device,
                train_text_encoder=train_text_encoder,
            )
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
                mse_loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")
                hf_loss = torch.zeros((), device=device, dtype=mse_loss.dtype)
                color_loss = torch.zeros((), device=device, dtype=mse_loss.dtype)

                loss = mse_loss

                if use_highfreq_loss:
                    hf_loss = compute_highfreq_loss(
                        pipeline=pipeline,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        model_pred=model_pred,
                        cutoff=highfreq_cutoff
                    )
                    loss = loss + highfreq_weight * hf_loss

                if use_color_loss:
                    color_loss = compute_color_loss(
                        pipeline=pipeline,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        model_pred=model_pred,
                        chroma_threshold=color_chroma_threshold,
                        low_sat_threshold=color_low_sat_threshold,
                        low_sat_target=color_low_sat_target,
                    )
                    loss = loss + color_loss_weight * color_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at step {global_step}. Try lowering the learning rate or batch size."
                )

            scaler.scale(loss / gradient_accumulation_steps).backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    if train_text_encoder:
                        clip_params = list(pipeline.unet.parameters()) + list(pipeline.text_encoder.parameters())
                    else:
                        clip_params = pipeline.unet.parameters()
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if train_text_encoder:
                        clip_params = list(pipeline.unet.parameters()) + list(pipeline.text_encoder.parameters())
                    else:
                        clip_params = pipeline.unet.parameters()
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            loss_value = loss.detach().item()
            epoch_loss += loss_value
            loss_history.append((global_step, loss_value))
            global_step += 1

            print(
                f"step={global_step:07d} "
                f"mse={mse_loss.detach().item():.6f} "
                f"highfreq={hf_loss.detach().item():.6f} "
                f"color={color_loss.detach().item():.6f} "
                f"total={loss_value:.6f}"
            )
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
                        "tune_mode": "full_with_text_encoder" if train_text_encoder else "full",
                        "train_text_encoder": train_text_encoder,
                        "text_encoder_learning_rate": text_encoder_learning_rate,
                        "caption_dropout": caption_dropout,
                        "use_highfreq_loss": use_highfreq_loss,
                        "highfreq_weight": highfreq_weight,
                        "highfreq_cutoff": highfreq_cutoff,
                        "use_color_loss": use_color_loss,
                        "color_loss_weight": color_loss_weight,
                        "color_chroma_threshold": color_chroma_threshold,
                        "color_low_sat_threshold": color_low_sat_threshold,
                        "color_low_sat_target": color_low_sat_target,
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
            "tune_mode": "full_with_text_encoder" if train_text_encoder else "full",
            "train_text_encoder": train_text_encoder,
            "text_encoder_learning_rate": text_encoder_learning_rate,
            "caption_dropout": caption_dropout,
            "use_highfreq_loss": use_highfreq_loss,
            "highfreq_weight": highfreq_weight,
            "highfreq_cutoff": highfreq_cutoff,
            "use_color_loss": use_color_loss,
            "color_loss_weight": color_loss_weight,
            "color_chroma_threshold": color_chroma_threshold,
            "color_low_sat_threshold": color_low_sat_threshold,
            "color_low_sat_target": color_low_sat_target,
        },
    )
    save_loss_curve(loss_history, output_dir)
    progress_bar.close()
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion v1.5 full UNet fine-tuning.")
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
        default=1,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for full UNet training.",
    )
    parser.add_argument(
        "--text-encoder-learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for text encoder when jointly fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../runs",
        help="Output directory for trained weights.",
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
        default=2000,
        help="Run inference sampling every N optimization steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=4000,
        help="Save UNet weights every N optimization steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--no-text-encoder",
        action="store_true",
        help="Disable joint text encoder fine-tuning and train UNet only.",
    )
    parser.add_argument(
        "--caption-dropout",
        type=float,
        default=0.0,
        help="Randomly drop captions during training to reduce overfitting.",
    )
    parser.add_argument(
        "--use-highfreq-loss",
        action="store_true",
        help="Enable a high-frequency penalty loss to encourage smooth gradients.",
    )
    parser.add_argument(
        "--highfreq-weight",
        type=float,
        default=10,
        help="Weight for high-frequency loss when --use-highfreq-loss is enabled.",
    )
    parser.add_argument(
        "--highfreq-cutoff",
        type=float,
        default=0.03,
        help="band half-width",
    )
    parser.add_argument(
        "--use-color-loss",
        action="store_true",
        help="Enable a color-consistency loss to reduce gray, low-saturation outputs.",
    )
    parser.add_argument(
        "--color-loss-weight",
        type=float,
        default=0.005,
        help="Weight for color loss when --use-color-loss is enabled.",
    )
    parser.add_argument(
        "--color-chroma-threshold",
        type=float,
        default=18.0,
        help="Lower-bound chroma target used by the color loss.",
    )
    parser.add_argument(
        "--color-low-sat-threshold",
        type=float,
        default=0.20,
        help="Per-pixel saturation threshold used by the color loss.",
    )
    parser.add_argument(
        "--color-low-sat-target",
        type=float,
        default=0.15,
        help="Target fraction of low-saturation pixels allowed by the color loss.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint directory created by save_checkpoint.",
    )
    args = parser.parse_args()

    pipeline = setup_sd15_full_tune(args.base_model)

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
    output_dir = Path(args.output_dir) / f"full_{start_time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_loop(
        pipeline=pipeline,
        train_dataloader=train_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        text_encoder_learning_rate=args.text_encoder_learning_rate,
        gradient_accumulation_steps=1,
        sample_every_steps=args.sample_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_from_checkpoint_path=args.resume_from_checkpoint,
        output_dir=output_dir,
        train_text_encoder=not args.no_text_encoder,
        caption_dropout=args.caption_dropout,
        use_highfreq_loss=args.use_highfreq_loss,
        highfreq_weight=args.highfreq_weight,
        highfreq_cutoff=args.highfreq_cutoff,
        use_color_loss=args.use_color_loss,
        color_loss_weight=args.color_loss_weight,
        color_chroma_threshold=args.color_chroma_threshold,
        color_low_sat_threshold=args.color_low_sat_threshold,
        color_low_sat_target=args.color_low_sat_target,
    )


if __name__ == "__main__":
    main()