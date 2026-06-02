from pathlib import Path
import re

import numpy as np
from PIL import Image

# Optional KD-tree acceleration for nearest-color search
from scipy.spatial import cKDTree 


BATCH_SIZE = 20000
STRICT_VALIDATE_OUTPUT = True
# Scale factor for error diffusion (0..1). Lower => fewer visible noise dots.
DIFFUSION_SCALE = 0.6


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0

    rgb_linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    xyz = np.empty_like(rgb_linear, dtype=np.float32)
    xyz[..., 0] = (
        rgb_linear[..., 0] * 0.4124564
        + rgb_linear[..., 1] * 0.3575761
        + rgb_linear[..., 2] * 0.1804375
    )
    xyz[..., 1] = (
        rgb_linear[..., 0] * 0.2126729
        + rgb_linear[..., 1] * 0.7151522
        + rgb_linear[..., 2] * 0.0721750
    )
    xyz[..., 2] = (
        rgb_linear[..., 0] * 0.0193339
        + rgb_linear[..., 1] * 0.1191920
        + rgb_linear[..., 2] * 0.9503041
    )

    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz_scaled = xyz / white

    delta = 6 / 29
    fxyz = np.where(
        xyz_scaled > delta**3,
        np.cbrt(xyz_scaled),
        xyz_scaled / (3 * delta**2) + 4 / 29,
    )

    lab = np.empty_like(xyz, dtype=np.float32)
    lab[..., 0] = 116 * fxyz[..., 1] - 16
    lab[..., 1] = 500 * (fxyz[..., 0] - fxyz[..., 1])
    lab[..., 2] = 200 * (fxyz[..., 1] - fxyz[..., 2])
    return lab


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
        lab = lab.astype(np.float32)
        # f^-1
        fy = (lab[..., 0] + 16.0) / 116.0
        fx = fy + lab[..., 1] / 500.0
        fz = fy - lab[..., 2] / 200.0

        delta = 6.0 / 29.0
        fx3 = fx ** 3
        fz3 = fz ** 3
        fy3 = fy ** 3

        xr = np.where(fx3 > delta**3, fx3, (fx - 4.0 / 29.0) * 3.0 * delta**2)
        yr = np.where(fy3 > delta**3, fy3, (fy - 4.0 / 29.0) * 3.0 * delta**2)
        zr = np.where(fz3 > delta**3, fz3, (fz - 4.0 / 29.0) * 3.0 * delta**2)

        white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        xyz = np.empty(lab.shape, dtype=np.float32)
        xyz[..., 0] = xr * white[0]
        xyz[..., 1] = yr * white[1]
        xyz[..., 2] = zr * white[2]

        # XYZ to linear RGB
        rgb_linear = np.empty(lab.shape, dtype=np.float32)
        rgb_linear[..., 0] = 3.2404542 * xyz[..., 0] - 1.5371385 * xyz[..., 1] - 0.4985314 * xyz[..., 2]
        rgb_linear[..., 1] = -0.9692660 * xyz[..., 0] + 1.8760108 * xyz[..., 1] + 0.0415560 * xyz[..., 2]
        rgb_linear[..., 2] = 0.0556434 * xyz[..., 0] - 0.2040259 * xyz[..., 1] + 1.0572252 * xyz[..., 2]

        # linear to sRGB
        a = 0.0031308
        srgb = np.where(
            rgb_linear <= a,
            rgb_linear * 12.92,
            1.055 * np.power(np.clip(rgb_linear, 0.0, None), 1.0 / 2.4) - 0.055,
        )
        srgb = np.clip(srgb, 0.0, 1.0)
        return (srgb * 255.0).astype(np.uint8)
    

def parse_sdl_file(sdl_file_path: Path) -> np.ndarray:
    colors = []
    pattern = re.compile(r"\(([\d.]+),([\d.]+)\),\((\d+),\s*(\d+),\s*(\d+)\)")

    with sdl_file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                r = int(match.group(3))
                g = int(match.group(4))
                b = int(match.group(5))
                colors.append([r, g, b])

    if not colors:
        raise ValueError(f"未从 SDL 文件中解析到任何颜色: {sdl_file_path}")

    return np.array(colors, dtype=np.uint8)


def map_rgb_array_to_gamut(rgb_array: np.ndarray, gamut_colors: np.ndarray) -> np.ndarray:
    height, width, _ = rgb_array.shape
    pixels = rgb_array.reshape(-1, 3)

    unique_colors, inverse_indices = np.unique(pixels, axis=0, return_inverse=True)
    mapped_unique = np.zeros_like(unique_colors, dtype=np.uint8)

    total = len(unique_colors)
    print(f"  唯一颜色数: {total}")

    gamut_colors_lab = srgb_to_lab(gamut_colors)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_lab = srgb_to_lab(unique_colors[start:end])

        diff = batch_lab[:, np.newaxis, :] - gamut_colors_lab[np.newaxis, :, :]
        distances = np.sum(diff * diff, axis=2)
        nearest_indices = np.argmin(distances, axis=1)
        mapped_unique[start:end] = gamut_colors[nearest_indices]

        progress = int(end / total * 100)
        print(f"  颜色映射进度: {progress}%", end="\r")

    print("  颜色映射进度: 100%")

    mapped_pixels = mapped_unique[inverse_indices]
    return mapped_pixels.reshape(height, width, 3)


def map_rgb_array_to_gamut_dither(rgb_array: np.ndarray, gamut_colors: np.ndarray) -> np.ndarray:
    """Lab-space Floyd–Steinberg dithering.

    Convert the image to Lab, perform nearest-color lookup against the gamut
    in Lab space, and diffuse the Lab error using standard Floyd–Steinberg
    coefficients. Return an sRGB uint8 array mapped to the gamut colors.
    """
    h, w, _ = rgb_array.shape

    # convert whole image to Lab (float)
    lab_img = srgb_to_lab(rgb_array.astype(np.float32))
    arr = lab_img.copy()

    # precompute gamut in Lab and also keep uint8 RGB gamut
    gamut_lab = srgb_to_lab(gamut_colors.astype(np.float32)).astype(np.float32)

    # try to build KD-tree for fast nearest lookup
    tree = None
    if cKDTree is not None:
        try:
            tree = cKDTree(gamut_lab)
        except Exception:
            tree = None

    mapped_idx = np.zeros((h, w), dtype=np.int32)

    # serpentine Floyd–Steinberg scan
    for y in range(h):
        left_to_right = (y % 2 == 0)
        if left_to_right:
            x_iter = range(0, w)
        else:
            x_iter = range(w - 1, -1, -1)

        for x in x_iter:
            pixel_lab = arr[y, x]

            # nearest palette index via KD-tree or brute force
            if tree is not None:
                _, idx = tree.query(pixel_lab, k=1)
                idx = int(idx)
                nearest_lab = gamut_lab[idx]
            else:
                diff = gamut_lab - pixel_lab[np.newaxis, :]
                dists = np.sum(diff * diff, axis=1)
                idx = int(np.argmin(dists))
                nearest_lab = gamut_lab[idx]

            mapped_idx[y, x] = idx
            err = pixel_lab - nearest_lab
            # scale diffusion strength
            err = err * DIFFUSION_SCALE

            # distribute error depending on scan direction
            if left_to_right:
                if x + 1 < w:
                    arr[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x - 1 >= 0:
                        arr[y + 1, x - 1] += err * (3.0 / 16.0)
                    arr[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * (1.0 / 16.0)
            else:
                if x - 1 >= 0:
                    arr[y, x - 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * (3.0 / 16.0)
                    arr[y + 1, x] += err * (5.0 / 16.0)
                    if x - 1 >= 0:
                        arr[y + 1, x - 1] += err * (1.0 / 16.0)

    # produce final RGB using exact palette entries to avoid numerical mismatch
    rgb_mapped = gamut_colors[mapped_idx]
    return rgb_mapped


def validate_rgb_array_in_sdl_table(rgb_array: np.ndarray, gamut_colors: np.ndarray) -> None:
    pixels = rgb_array.reshape(-1, 3)
    unique_pixels = np.unique(pixels, axis=0)
    gamut_set = {tuple(color.tolist()) for color in gamut_colors}

    invalid_colors = [tuple(color.tolist()) for color in unique_pixels if tuple(color.tolist()) not in gamut_set]
    if invalid_colors:
        preview = invalid_colors[:10]
        raise ValueError(f"输出图片存在不在 SDL 表中的颜色，共 {len(invalid_colors)} 个，示例: {preview}")


def convert_image_to_sdl(image_path: Path, gamut_colors: np.ndarray) -> None:
    with Image.open(image_path) as image:
        original_mode = image.mode

        if original_mode in {"RGBA", "LA"}:
            rgba_image = image.convert("RGBA")
            rgba_array = np.array(rgba_image, dtype=np.uint8)
            rgb_array = rgba_array[:, :, :3]
            alpha_channel = rgba_array[:, :, 3]

            mapped_rgb = map_rgb_array_to_gamut(rgb_array, gamut_colors)
            if STRICT_VALIDATE_OUTPUT:
                validate_rgb_array_in_sdl_table(mapped_rgb, gamut_colors)
            mapped_rgba = np.dstack([mapped_rgb, alpha_channel])
            result = Image.fromarray(mapped_rgba, mode="RGBA")
        else:
            rgb_image = image.convert("RGB")
            rgb_array = np.array(rgb_image, dtype=np.uint8)
            mapped_rgb = map_rgb_array_to_gamut(rgb_array, gamut_colors)
            if STRICT_VALIDATE_OUTPUT:
                validate_rgb_array_in_sdl_table(mapped_rgb, gamut_colors)
            result = Image.fromarray(mapped_rgb, mode="RGB")

        if result.mode == "RGBA":
            background = Image.new("RGB", result.size, (255, 255, 255))
            background.paste(result, mask=result.getchannel("A"))
            result = background
        else:
            result = result.convert("RGB")
            
    return result


def convert_image_to_sdl_with_dither(image_path: Path, gamut_colors: np.ndarray) -> Image.Image:
    """Convert image to SDL gamut using Lab-space Floyd–Steinberg dithering.

    Returns a PIL `Image` in RGB mode (alpha is composited to white).
    """
    with Image.open(image_path) as image:
        original_mode = image.mode

        if original_mode in {"RGBA", "LA"}:
            rgba_image = image.convert("RGBA")
            rgba_array = np.array(rgba_image, dtype=np.uint8)
            rgb_array = rgba_array[:, :, :3]
            alpha_channel = rgba_array[:, :, 3]

            mapped_rgb = map_rgb_array_to_gamut_dither(rgb_array, gamut_colors)
            if STRICT_VALIDATE_OUTPUT:
                validate_rgb_array_in_sdl_table(mapped_rgb, gamut_colors)
            mapped_rgba = np.dstack([mapped_rgb, alpha_channel])
            result = Image.fromarray(mapped_rgba, mode="RGBA")
        else:
            rgb_image = image.convert("RGB")
            rgb_array = np.array(rgb_image, dtype=np.uint8)
            mapped_rgb = map_rgb_array_to_gamut_dither(rgb_array, gamut_colors)
            if STRICT_VALIDATE_OUTPUT:
                validate_rgb_array_in_sdl_table(mapped_rgb, gamut_colors)
            result = Image.fromarray(mapped_rgb, mode="RGB")

        if result.mode == "RGBA":
            background = Image.new("RGB", result.size, (255, 255, 255))
            background.paste(result, mask=result.getchannel("A"))
            result = background
        else:
            result = result.convert("RGB")

    return result


def main() -> None:
    
    input_image = Path("../../result/eval/full_color_high_freq_text/samples/Abstract_light_effect_smooth_blend_from_cool_teal_to_gentle_yellow_energetic_atmosphere_diffuse_glow_no_room_no_furnitur/sample_001.png")
    output_dir = Path("../../result")
    sdl_file = Path("../../data/SDL2_0.txt")
    
    if not input_image.exists():
        raise FileNotFoundError(f"输入图片不存在: {input_image}")
    if not sdl_file.exists():
        raise FileNotFoundError(f"SDL 文件不存在: {sdl_file}")

    print(f"读取 SDL 色域文件: {sdl_file}")
    gamut_colors = parse_sdl_file(sdl_file)
    print(f"SDL 颜色数量: {len(gamut_colors)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = convert_image_to_sdl(input_image, gamut_colors)
    output_path = output_dir / "sdl_converted_simple.png"
    result.save(output_path)

    dithered_img = convert_image_to_sdl_with_dither(input_image, gamut_colors)
    dither_path = output_dir / "sdl_converted_dithered.png"
    dithered_img.save(dither_path)


if __name__ == "__main__":
    main()
