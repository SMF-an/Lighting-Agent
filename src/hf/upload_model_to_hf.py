import argparse
import json
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, get_full_repo_name


load_dotenv()


def build_model_card(repo_id: str, checkpoint_type: str, base_model: str) -> str:
    return f"""---
language: en
license: mit
library_name: diffusers
tags:
- stable-diffusion
- text-to-image
- finetuned
- {checkpoint_type}
base_model: {base_model}
---

# {repo_id}

This repository contains a fine-tuned Stable Diffusion v1.5 checkpoint for generating light effects.

"""


def build_upload_patterns(checkpoint_type: str) -> list[str]:
    common_patterns = ["trainer_state.pt", "README.md", "checkpoint_meta.json"]
    if checkpoint_type == "lora":
        return common_patterns + ["lora_adapter/**"]
    if checkpoint_type == "full":
        return common_patterns + ["model/**"]
    raise ValueError(f"Unsupported checkpoint type: {checkpoint_type}")


def write_checkpoint_meta(checkpoint_dir: str, checkpoint_type: str, base_model: str) -> Path:
    trainer_state_path = Path(checkpoint_dir) / "trainer_state.pt"
    trainer_state = {}
    if trainer_state_path.exists():
        # Keep this lightweight and robust even if torch is not installed for this script.
        trainer_state = {"trainer_state_present": True}

    payload = {
        "checkpoint_type": checkpoint_type,
        "base_model": base_model,
        "checkpoint_dir": str(checkpoint_dir),
        "trainer_state": trainer_state,
    }
    out_path = Path(checkpoint_dir) / "checkpoint_meta.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def upload_checkpoint(
    model_name: str,
    checkpoint_dir: str,
    base_model: str,
    checkpoint_type: str | None = None
) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token not found. Set HF_TOKEN.")

    repo_id = get_full_repo_name(model_name)
    api = HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    model_card = build_model_card(
        repo_id=repo_id,
        checkpoint_type=checkpoint_type,
        base_model=base_model
    )

    api.upload_file(
        path_or_fileobj=BytesIO(model_card.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Add model card",
    )

    write_checkpoint_meta(
        checkpoint_dir=checkpoint_dir,
        checkpoint_type=checkpoint_type,
        base_model=base_model,
    )

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        allow_patterns=build_upload_patterns(checkpoint_type),
        commit_message=f"Upload {checkpoint_type} checkpoint weights",
    )

    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a fine-tuned checkpoint to Hugging Face Hub.")
    parser.add_argument(
        "--model-name",
        default="light-effect-generator",
        help="Target Hugging Face model repo name, e.g. 'my-username/sd15-light-effect-lora' or short name.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="../../runs/full_20260524_124658/checkpoints/final",
        help="Path to checkpoint directory (contains either lora_adapter/ or model/unet_state_dict.pt).",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["lora", "full"],
        default="full",
        help="Optional checkpoint type override",
    )
    parser.add_argument(
        "--base-model",
        default="runwayml/stable-diffusion-v1-5",
        help="Base model ID used for fine-tuning.",
    )
    args = parser.parse_args()

    repo_id = upload_checkpoint(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        checkpoint_type=args.checkpoint_type,
    )

    print(f"Upload complete: {repo_id}")


if __name__ == "__main__":
    main()
