import argparse
import json
import os
import re

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


SCENE_PALETTE_RULES = [
    {
        "name": "education",
        "keywords": [
            "classroom",
            "教室",
            "school",
            "学校",
            "lesson",
            "lecture",
            "study",
            "学习",
            "desk",
            "blackboard",
            "whiteboard",
            "library",
            "图书馆",
            "reading",
            "阅读",
            "study room",
            "quiet",
            "安静",
            "专注",
            "office",
            "办公",
            "work",
            "meeting",
            "conference",
        ],
        "primary_color": "muted beige",
        "secondary_color": "soft yellow",
        "mood": "focused",
        "note": "clean, focused, low-saturation palette",
    },
    {
        "name": "restaurant",
        "keywords": ["restaurant", "dining", "eatery", "bistro", "dinner", "meal", "cafe", "coffee shop", "coffeehouse", "espresso", "latte", "餐厅", "用餐", "晚餐", "饭店"],
        "primary_color": "warm amber",
        "secondary_color": "pale cream",
        "mood": "cozy",
        "note": "cozy, inviting, subtle warm tones",
    },
    {
        "name": "bar",
        "keywords": ["bar", "pub", "lounge", "cocktail", "nightlife", "酒吧", "吧台", "夜生活"],
        "primary_color": "muted magenta",
        "secondary_color": "deep purple",
        "mood": "moody",
        "note": "dim, atmospheric, restrained highlights",
    },
    {
        "name": "hotel lounge",
        "keywords": ["hotel", "hotel lounge", "hotel lobby", "hotel living room", "lobby", "酒店", "客厅", "大堂"],
        "primary_color": "soft ivory",
        "secondary_color": "muted gold",
        "mood": "elegant",
        "note": "refined, calm, upscale ambient light",
    }
]


FEW_SHOT_EXAMPLES = [
    {
        "scene": "办公空间，清新自然光，用于营造活力、专注的感觉",
        "output": {
            "primary_color": "muted beige",
            "secondary_color": "soft yellow",
            "transition": "soft gradient",
            "mood": "focused",
            "detail": "soft glow",
        },
    },
    {
        "scene": "酒吧吧台，抽象霓虹样式光，用于微醺、社交、停留延长。",
        "output": {
            "primary_color": "muted magenta",
            "secondary_color": "deep purple",
            "transition": "smooth blend",
            "mood": "moody",
            "detail": "diffuse light",
        },
    },
    {
        "scene": "飘香的咖啡厅，人们在这里能够惬意地品尝咖啡。",
        "output": {
            "primary_color": "warm amber",
            "secondary_color": "pale cream",
            "transition": "soft gradient",
            "mood": "cozy",
            "detail": "soft glow",
        },
    },
]


def scene_keyword_in_text(text, keyword):
    if keyword.isascii():
        return keyword.lower() in text.lower()
    return keyword in text


def build_few_shot_block():
    lines = ["Few-shot examples:"]
    for example in FEW_SHOT_EXAMPLES:
        lines.append(f"Scene description: {example['scene']}")
        lines.append(f"Target JSON: {json.dumps(example['output'], ensure_ascii=False)}")
    return "\n".join(lines)


def build_system_prompt():
    return (
        "You are a strict lighting-effect scene-to-prompt translator. "
        "Return JSON only, with exactly these keys: primary_color, secondary_color, transition, mood, detail. "
        "Use short, concrete phrases. "
        "The input scene description may be in English or Chinese; handle both directly. "
        "Choose colors from the recommended palette. "
        "Never mention rooms, furniture, buildings, people, objects, or any real-world scene. "
        "Keep transition to a short phrase like 'soft gradient' or 'smooth blend'. "
        "Keep mood to a short phrase like 'calm', 'dreamy', 'serene', 'fresh', 'energetic', or 'warm'. "
        "Keep detail to a short phrase like 'soft glow', 'diffuse light', or 'smooth texture'. "
        "Do not use the words black, dark, or shadow. "
        "Do not add extra keys. "
        "The final prompt must be convertible to this exact sentence style: "
        "'Abstract light effect, {transition} from {primary_color} to {secondary_color}, {mood} atmosphere, {detail}, no room, no furniture, no objects.' "
    )


def recommend_palette(input_scene_description):
    lowered = input_scene_description.lower()
    for rule in SCENE_PALETTE_RULES:
        if not rule["keywords"]:
            continue
        if any(scene_keyword_in_text(lowered, keyword) for keyword in rule["keywords"]):
            return rule
    return None


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
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.IGNORECASE)
    if m:
        return m.group(1)

    m2 = re.search(r"(\{\s*\"?[a-zA-Z0-9_\- ]+\"?\s*:\s*[^}]+\})", text, flags=re.S)
    if m2:
        return m2.group(1)

    return None


def call_llm_for_effect(input_scene_description):
    palette = recommend_palette(input_scene_description)
    user_content_parts = [
        "Please translate the following scene description into a controlled lighting caption schema:\n\n",
        f"Scene description: {input_scene_description}\n\n",
        f"{build_few_shot_block()}\n\n"
    ]

    if palette:
        user_content_parts.append(
            f"Recommended palette: primary_color={palette['primary_color']}, secondary_color={palette['secondary_color']}, mood={palette['mood']}. "
        )
        user_content_parts.append(
            "Prefer this palette unless the description strongly suggests a different color pair.\n\n"
        )

    user_content_parts.append(
        "The scene description may be written in Chinese or English. If it is Chinese, infer the scene type directly before choosing colors.\n\n"
    )
    user_content_parts.append(
        "Return JSON only with keys primary_color, secondary_color, transition, mood, detail."
    )

    user_content = "".join(user_content_parts)

    payload = {
        "model": os.getenv("LLM_MODEL_ID", "gpt-4o"),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content},
        ],
    }

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=payload["model"],
        messages=payload["messages"],
        response_format=payload["response_format"],
        temperature=0.2,
    )

    raw_content = response.choices[0].message.content.strip()

    json_candidate = extract_json_from_text(raw_content)
    if json_candidate:
        try:
            schema = json.loads(json_candidate)
            if isinstance(schema, dict):
                return build_caption_from_schema(schema)
        except Exception:
            pass

    try:
        schema = json.loads(raw_content)
        if isinstance(schema, dict):
            return build_caption_from_schema(schema)
    except Exception:
        pass

    return sanitize_caption(raw_content)


def main():
    parser = argparse.ArgumentParser(description="Module 1: Scene Description -> Controlled Lighting Prompt translator")
    parser.add_argument("scene", help="Scene description text", nargs="+")
    args = parser.parse_args()
    scene_text = " ".join(args.scene)
    llm_result = call_llm_for_effect(scene_text)
    print("Original description:\n", scene_text)
    print("\nGenerated image prompt:\n", llm_result)


if __name__ == "__main__":
    main()
