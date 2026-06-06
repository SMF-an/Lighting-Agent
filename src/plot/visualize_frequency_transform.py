from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def compute_cross_mask_frequency_loss(luminance: torch.Tensor, cutoff: float) -> float:
    """Compute the same cross-mask ratio used by training, but directly from an image."""
    Fk = torch.fft.rfftn(luminance, dim=(1, 2))
    mag2 = (Fk.real ** 2 + Fk.imag ** 2)

    H, Wp = mag2.shape[1], mag2.shape[2]
    fy = torch.fft.fftfreq(H, device=mag2.device).abs().unsqueeze(1)  # H x 1
    fx = torch.fft.rfftfreq(luminance.shape[2], device=mag2.device).unsqueeze(0)  # 1 x Wp

    row_band = fy <= float(cutoff)
    col_band = fx <= float(cutoff)
    mask = torch.logical_or(row_band, col_band).float().unsqueeze(0)  # 1 x H x Wp

    outside_mask = 1.0 - mask
    hf_energy = (mag2 * outside_mask).sum(dim=(1, 2))
    total_energy = mag2.sum(dim=(1, 2))
    loss = (hf_energy / (total_energy + 1e-8)).mean()
    return float(loss.item())


def build_figure(image_path: Path, output_path: Path, cutoff: float = 0.03) -> None:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # to tensor B x C x H x W
    decoded = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    
    luminance = 0.299 * decoded[:, 0] + 0.587 * decoded[:, 1] + 0.114 * decoded[:, 2]
    luminance = torch.nn.functional.interpolate(
        luminance.unsqueeze(1), size=(128, 128), mode="bilinear", align_corners=False
    ).squeeze(1)

    loss_value = compute_cross_mask_frequency_loss(luminance, cutoff=cutoff)

    # Convert to numpy grayscale image for numpy FFT routines
    img = (luminance[0].detach().cpu().numpy() * 255.0).astype(np.float32)

    # centered FFT for display
    fft2 = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft2)
    log_mag = np.log1p(np.abs(fft_shifted))

    # Build a cross-shaped mask in frequency domain: keep the horizontal and vertical center bands.
    H, W = img.shape
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    row_band = np.abs(fy) <= float(cutoff)
    col_band = np.abs(fx) <= float(cutoff)

    # Align with fftshifted spectrum ordering before combining.
    row_band_shifted = np.fft.fftshift(row_band).astype(np.float32)[:, None]
    col_band_shifted = np.fft.fftshift(col_band).astype(np.float32)[None, :]
    mask2d = np.maximum(row_band_shifted, col_band_shifted)

    filtered_shifted = fft_shifted * mask2d
    filtered_log_mag = np.log1p(np.abs(filtered_shifted))

    # reconstruct spatial image from filtered FFT
    unshifted = np.fft.ifftshift(filtered_shifted)
    reconstructed = np.fft.ifft2(unshifted)
    reconstructed = np.abs(reconstructed)

    # Keep both spectra on the same color scale for fair comparison
    vmin = float(log_mag.min())
    vmax = float(log_mag.max())

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Luminance (128x128)")
    ax1.imshow(img, cmap="gray")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Log Magnitude Spectrum")
    ax2.imshow(log_mag, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title(f"Cross Mask Log Spectrum (cutoff={cutoff})")
    ax3.imshow(filtered_log_mag, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Filtered Image (Inverse FFT)")
    ax4.imshow(reconstructed, cmap="gray")
    ax4.axis("off")

    fig.suptitle(f"Cross-mask frequency loss: {loss_value:.6f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    
    image_path = Path("../../data/images/1008.png")
    output_path = Path("../../result/frequency_transform_comparison.png")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    build_figure(image_path=image_path, output_path=output_path)
    print(f"Saved comparison figure to {output_path}")
