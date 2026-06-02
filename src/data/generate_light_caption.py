import base64
import re
import json
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


FORBIDDEN_SCENE_TERMS = [
    "living room",
    "dining room",
    "bedroom",
    "kitchen",
    "bathroom",
    "hallway",
    "interior",
    "room",
    "furniture",
    "sofa",
    "chair",
    "bed",
    "table",
    "lamp",
    "window",
    "curtain",
    "house",
    "office",
    "space",
    "scene",
    "store",
    "shop",
    "retail",
    "cosmetics",
    "testing",
    "gallery",
    "stage",
    "corridor",
]


def build_system_prompt(mode="controlled"):
    if mode == "free":
        return (
            "You are a lighting-effect image annotation assistant. "
            "Write one concise English caption that can be directly used as a lighting prompt. "
            "Focus on color, gradient, glow, mood and style. "
            "Do not mention rooms, furniture, buildings, people, or any real-world scene. "
            "Do not use the words black, dark, or shadow. "
            "Return plain text only, not JSON. "
        )

    return (
        "You are a strict lighting-effect image annotation assistant. "
        "Return JSON only, with exactly these keys: primary_color, secondary_color, transition, mood, detail. "
        "Use short, concrete phrases. "
        "Choose colors from concise visual descriptions only. "
        "Never mention rooms, furniture, buildings, people, objects, or any real-world scene. "
        "Keep transition to a short phrase like 'soft gradient' or 'smooth blend'. "
        "Keep mood to a short phrase like 'calm', 'dreamy', 'serene', 'fresh', 'energetic', or 'warm'. "
        "Keep detail to a short phrase like 'soft glow', 'diffuse light', or 'smooth texture'. "
        "Do not use the words black, dark, or shadow. "
        "Do not add extra keys. "
        "Example: {\"primary_color\": \"warm amber\", \"secondary_color\": \"soft magenta\", \"transition\": \"soft gradient\", \"mood\": \"calm\", \"detail\": \"soft glow\"}. "
    )


def build_user_prompt(image_name):
    return (
        f"Describe the lighting image '{image_name}' in a controlled, scene-free way. "
        "If JSON is requested, use only the allowed keys and keep the values short. "
    )


def build_caption_from_schema(schema):
    primary_color = str(schema.get("primary_color", "warm amber")).strip().lower()
    secondary_color = str(schema.get("secondary_color", "soft magenta")).strip().lower()
    transition = str(schema.get("transition", "soft gradient")).strip().lower()
    mood = str(schema.get("mood", "calm")).strip().lower()
    detail = str(schema.get("detail", "soft glow")).strip().lower()

    transition = re.sub(r"[^a-z0-9\- ]+", "", transition).strip() or "soft gradient"
    detail = re.sub(r"[^a-z0-9\- ]+", "", detail).strip() or "soft glow"

    return (
        f"Abstract light effect, {transition} from {primary_color} to {secondary_color}, "
        f"{mood} atmosphere, {detail}, no room, no furniture, no objects."
    )


def sanitize_caption(text):
    # remove common code fences and inline code markers
    caption = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
    caption = re.sub(r"`", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    if not caption:
        return "Abstract light effect, soft gradient from warm amber to soft magenta, calm atmosphere, soft glow, no room, no furniture, no objects."

    lowered = caption.lower()
    for term in FORBIDDEN_SCENE_TERMS:
        lowered = lowered.replace(term, "")

    lowered = re.sub(r"\b(black|dark|shadow)\b", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip(" ,.;:")

    if not lowered:
        lowered = "abstract light effect, soft gradient, calm atmosphere, soft glow"

    if "room" not in lowered and "furniture" not in lowered and "objects" not in lowered:
        lowered = f"{lowered}, no room, no furniture, no objects"

    lowered = lowered[0].upper() + lowered[1:] if lowered else lowered
    return lowered


def extract_json_from_text(text):
    """Try to extract a JSON object from text. Returns the JSON string or None."""
    # first look for ```json { ... } ``` blocks
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.IGNORECASE)
    if m:
        return m.group(1)

    # next look for any { ... } that looks like a JSON object
    m2 = re.search(r"(\{\s*\"?[a-zA-Z0-9_\- ]+\"?\s*:\s*[^}]+\})", text, flags=re.S)
    if m2:
        return m2.group(1)

    return None


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def call_vlm_for_effect(image_path, caption_mode="controlled"):
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
            {"role": "system", "content": build_system_prompt(caption_mode)},
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

    raw_caption = response.choices[0].message.content.strip()

    if caption_mode == "free":
        return sanitize_caption(raw_caption), model_id

    # Try to extract a JSON object from the VLM response (handles ```json {...}``` blocks)
    json_candidate = extract_json_from_text(raw_caption)
    if json_candidate:
        try:
            schema = json.loads(json_candidate)
            if isinstance(schema, dict):
                return build_caption_from_schema(schema), model_id
        except Exception:
            pass

    try:
        schema = json.loads(raw_caption)
        if isinstance(schema, dict):
            return build_caption_from_schema(schema), model_id
    except Exception:
        pass

    return sanitize_caption(raw_caption), model_id


def list_images(image_dir):
    return sorted(path for path in image_dir.rglob("*.png") if path.is_file())


def build_record(image_path, caption, root_dir):
    return {
        "image": image_path.relative_to(root_dir).as_posix(),
        "caption": caption
    }


def export_captions(image_dir, output_path, caption_mode="controlled"):
    images = list_images(image_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            caption, model_id = call_vlm_for_effect(image_path, caption_mode=caption_mode)
            record = build_record(image_path, caption, image_dir.parent)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            print(f"[{count:04d}] {image_path.name} -> {caption}")

    return count


def main():

    image_dir = Path("../../data/images")
    output_path = Path("../../data/light_effect_captions.jsonl")

    total = export_captions(image_dir, output_path, caption_mode="controlled")
    print(f"Done. Wrote {total} caption records to {output_path}")


if __name__ == "__main__":
    main()