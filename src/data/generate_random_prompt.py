import json
import random
from pathlib import Path


DEFAULT_OUTPUT_TEMPLATE = (
    "Abstract light effect, {transition} from {primary_color} to {secondary_color}, "
    "{mood} atmosphere, {detail}, no room, no furniture, no objects."
)


VOCABULARY = {
    "transition": [
        "smooth blend",
        "smooth gradient",
        "uniform",
    ],
    "color": [
        "blue",
        "bright green",
        "bright yellow",
        "bright yellow-green",
        "cool blue",
        "cool cyan",
        "cyan",
        "deep blue",
        "deep purple",
        "deep red",
        "golden yellow",
        "green",
        "magenta",
        "pink",
        "purple",
        "red",
        "soft beige",
        "soft blue",
        "soft coral",
        "soft cyan",
        "soft gray",
        "soft green",
        "soft lavender",
        "soft mint",
        "soft mint green",
        "soft peach",
        "soft pink",
        "soft purple",
        "soft red",
        "soft teal",
        "soft yellow",
        "soft yellow-green",
        "teal",
        "vibrant green",
        "vibrant magenta",
        "vibrant pink",
        "vibrant red",
        "vibrant yellow",
        "violet",
        "vivid green",
        "vivid magenta",
        "vivid pink",
        "vivid purple",
        "warm amber",
        "warm beige",
        "warm coral",
        "warm orange",
        "warm peach",
        "warm red",
        "bright cyan",
        "bright lime green",
        "bright magenta",
        "bright pink",
        "bright red",
        "bright teal",
        "clear blue",
        "cool green",
        "cool teal",
        "coral",
        "emerald green",
        "faint green",
        "faint purple",
        "gentle blue",
        "gentle coral",
        "gentle cyan",
        "gentle green",
        "gentle lavender",
        "gentle lime",
        "gentle magenta",
        "gentle mint",
        "gentle peach",
        "gentle pink",
        "gentle purple",
        "gentle rose",
        "gentle teal",
        "gentle yellow",
        "intense magenta",
        "intense red",
        "lavender",
        "light beige",
        "light blue",
        "light coral",
        "light cyan",
        "light gray",
        "light green",
        "light lavender",
        "light magenta",
        "light mint",
        "light peach",
        "light periwinkle",
        "light pink",
        "light purple",
        "light teal",
        "light yellow",
        "lime green",
        "muted blue",
        "muted green",
        "muted olive",
        "muted teal",
        "orange",
        "pale blue",
        "pale cream",
        "pale gray",
        "pale green",
        "pale lavender",
        "pale mint",
        "pale peach",
        "pale pink",
        "pale yellow",
        "pastel pink",
        "rich magenta",
        "rich purple",
        "rich red",
        "soft gold",
        "soft magenta",
        "soft orange",
        "soft teal",
        "teal blue",
        "teal green",
        "violet blue",
        "vivid red",
        "warm cream",
        "warm gold",
        "warm yellow",
        "yellow",
    ],
    "mood": [
        "balanced",
        "calm",
        "dreamy",
        "dynamic",
        "energetic",
        "fresh",
        "serene",
        "vibrant",
        "warm",
    ],
    "detail": [
        "diffuse glow",
        "diffuse light",
        "gentle glow",
        "smooth texture",
        "soft blend",
        "soft glow",
        "subtle glow",
        "uniform glow",
    ],
}


def sample_prompt(vocab: dict, rng: random.Random):
    transition = rng.choice(vocab.get("transition", ["soft gradient"]))
    primary_color = rng.choice(vocab.get("color", ["warm amber"]))

    secondary_color = rng.choice(vocab.get("color", ["soft magenta"]))
    while secondary_color == primary_color:
        secondary_color = rng.choice(vocab.get("color", ["soft magenta"]))

    mood = rng.choice(vocab.get("mood", ["calm"]))
    detail = rng.choice(vocab.get("detail", ["soft glow"]))

    return DEFAULT_OUTPUT_TEMPLATE.format(
        transition=transition,
        primary_color=primary_color,
        secondary_color=secondary_color,
        mood=mood,
        detail=detail,
    )


def main():
    
    count = 16
    seed = 42
    output_path = "../../result/eval/prompts.json"
    vocab = VOCABULARY

    rng = random.Random(seed)
    prompts = [sample_prompt(vocab, rng) for _ in range(count)]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for idx, prompt in enumerate(prompts, start=1):
                record = {"id": idx, "prompt": prompt}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(prompts)} prompts to {output_path}")
    else:
        for idx, prompt in enumerate(prompts, start=1):
            print(f"[{idx:03d}] {prompt}")


if __name__ == "__main__":
    main()