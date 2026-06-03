from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.registry import DISEASE_REGISTRY
from app.ml.training import train_disease_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train publication-grade disease prediction models.")
    parser.add_argument(
        "--disease",
        default="diabetes",
        choices=["all", *sorted(DISEASE_REGISTRY)],
        help="Disease registry ID to train, or all. Defaults to diabetes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disease_ids = sorted(DISEASE_REGISTRY) if args.disease == "all" else [args.disease]
    results = {disease_id: train_disease_model(disease_id) for disease_id in disease_ids}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
