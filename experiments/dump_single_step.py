import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.instrumented_molmoact import (
    InstrumentedMolmoAct,
    to_jsonable,
)

import os

SIMPLER_ROOT = Path(
    "/root/autodl-tmp/molmoact_project/code/SimplerEnv"
)

if not SIMPLER_ROOT.exists():
    raise FileNotFoundError(
        f"SimplerEnv root not found: {SIMPLER_ROOT}"
    )

os.chdir(SIMPLER_ROOT)

print("Using SimplerEnv working directory:", os.getcwd())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    image = np.asarray(
        Image.open(args.image).convert("RGB"),
        dtype=np.uint8,
    )

    print("backend:", args.backend)
    print("image:", args.image)
    print("image shape:", image.shape)
    print("instruction:", args.instruction)

    model = InstrumentedMolmoAct(
        saved_model_path=args.checkpoint,
        policy_setup="google_robot",
        backend=args.backend,
        quiet=True,
    )

    model.reset(args.instruction)

    raw_action, action, _ = model.step(
        image,
        args.instruction,
    )

    result = dict(model.last_debug)

    result["backend"] = args.backend
    result["image_path"] = args.image
    result["instruction"] = args.instruction

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable(result),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== RESULT =====")
    print("depth:", result.get("depth"))
    print("trace:", result.get("trace"))
    print("parsed_action:", result.get("parsed_action"))
    print("inference_time_s:", result.get("inference_time_s"))

    print("\nsaved:", args.output)


if __name__ == "__main__":
    main()