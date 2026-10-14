"""Write the public IOI prompts to data/public/circuits/ioi_public.json, served at
/datasets/circuits/ioi_public.json.

    python -m scripts.ioi_export_public

The file has the scored prompts and the ABC prompts the means come from, so a player
can rebuild the learn-tier score exactly. Only the public split is ever exported.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.challenges.circuits import (NAMES, OBJECTS, PLACES, PUBLIC_SEED, TEMPLATES,
                                     generate_abc, generate_ioi)

OUT = Path("data/public/circuits/ioi_public.json")
N_PER_TEMPLATE = 6
N_MEAN_PER_TEMPLATE = 32


def build() -> dict:
    ioi = generate_ioi(PUBLIC_SEED, N_PER_TEMPLATE)
    abc = generate_abc(PUBLIC_SEED + 1, N_MEAN_PER_TEMPLATE)
    return {
        "model_id": "gpt2",
        "split": "public",
        "seed": PUBLIC_SEED,
        "metric": "faithfulness = (LD_circuit - LD_ablated) / (LD_full - LD_ablated), "
                  "LD = mean over prompts of logit(' ' + io) - logit(' ' + s) at the last token",
        "ablation": "mean over abc_prompts of the same template, per position",
        "templates": list(TEMPLATES),
        "names": NAMES, "places": PLACES, "objects": OBJECTS,
        "prompts": [p._asdict() for p in ioi],
        "abc_prompts": [p._asdict() for p in abc],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
