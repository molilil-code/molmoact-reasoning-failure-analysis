import argparse
import json
import os
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


SIMPLER_ROOT = Path(
    "/root/autodl-tmp/molmoact_project/code/SimplerEnv"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument(
        "--backend",
        required=True,
        choices=["hf", "vllm"],
    )

    parser.add_argument(
        "--image-dir",
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--instruction",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    os.chdir(SIMPLER_ROOT)

    print("Loading:", args.backend)

    model = InstrumentedMolmoAct(
        saved_model_path=args.checkpoint,
        policy_setup="google_robot",
        backend=args.backend,
        quiet=True,
    )

    model.reset(args.instruction)

    image_dir = Path(args.image_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for step in args.steps:

            image_path = (
                image_dir
                / f"rgb_{step:03d}.jpg"
            )

            if not image_path.exists():
                print(
                    "missing:",
                    image_path,
                )
                continue

            image = np.asarray(
                Image.open(image_path).convert("RGB"),
                dtype=np.uint8,
            )

            print(
                f"[{args.backend}] step={step}"
            )

            model.reset(args.instruction)

            raw_action, action, _ = model.step(
                image,
                args.instruction,
            )

            record = {
                "backend": args.backend,
                "step": step,
                "image_path": str(image_path),
                "instruction": args.instruction,
                **model.last_debug,
            }

            fout.write(
                json.dumps(
                    to_jsonable(record),
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()

            print(
                "  time:",
                model.last_debug.get(
                    "inference_time_s"
                )
            )

    print("saved:", output_path)


if __name__ == "__main__":
    main()