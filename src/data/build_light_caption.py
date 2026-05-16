import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def build_system_prompt():
    return (
        "You are a lighting-effect image annotation assistant. "
        "Write one concise English caption that can be directly used as a lighting prompt. "
        "Focus on color, gradient, glow, mood, style and the most suitable real-world scene for this lighting. "
        "Use bright warm tones only. "
        "Do not use the words black, dark, or shadow. "
        "Return plain text only, not JSON. "
        "Examples: Soft gradient lighting, warm sunset hues transitioning from amber to soft magenta, cozy and relaxing atmosphere, cinematic lighting, 8k resolution. "
        "Soft gradient lighting transitions from light yellow to pale pink, relaxing and immersive atmosphere for the retail cosmetics testing area. "
        "Warm and flowing light, soft gradient of yellow and light orange, intimate and solemn atmosphere, comfort of the dining space. "
        "Bright and warm tones, pale yellow and light orange, fresh and invigorating atmosphere, focus and vitality of the office space. "
    )


def build_user_prompt(image_name):
    return (
        f"Describe the lighting image '{image_name}' in one concise English sentence. "
    )


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def call_vlm_for_effect(image_path):
    model_id = os.getenv("VLM_MODEL_ID", "qwen-vl-max-latest")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    data_url = image_to_data_url(image_path)

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": build_user_prompt(image_path.name)},
                ],
            },
        ],
        "temperature": 0.2,
    }

    response = client.chat.completions.create(
        model=payload["model"],
        messages=payload["messages"],
        temperature=payload["temperature"],
    )

    return response.choices[0].message.content.strip(), model_id


def list_images(image_dir):
    return sorted(path for path in image_dir.rglob("*.png") if path.is_file())


def build_record(image_path, caption, root_dir):
    return {
        "image": image_path.relative_to(root_dir).as_posix(),
        "caption": caption
    }


def export_captions(image_dir, output_path):
    images = list_images(image_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            caption, model_id = call_vlm_for_effect(image_path)
            record = build_record(image_path, caption, image_dir.parent)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            print(f"[{count:04d}] {image_path.name} -> {caption}")

    return count


def main():
    parser = argparse.ArgumentParser(description="Module 2: Build image-caption pairs with Qwen-VL")
    parser.add_argument(
        "--image-dir",
        default="data/trial/images",
        help="Directory containing lighting images",
    )
    parser.add_argument(
        "--output",
        default="data/trial/light_effect_captions.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    total = export_captions(image_dir, output_path)
    print(f"Done. Wrote {total} caption records to {output_path}")


if __name__ == "__main__":
    main()