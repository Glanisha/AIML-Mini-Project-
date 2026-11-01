"""End-to-end, reproducible OralSight ML pipeline.

    python run_pipeline.py                 # audit -> train -> calibrate -> export -> evaluate
    python run_pipeline.py --skip-train    # re-run calibration/export/evaluation on saved checkpoints

Raw data must be in ml/data/raw/{benign,malignant} (see the root README.md).
"""

import argparse

from oralsight_ml import audit, calibrate, evaluate, export, train
from oralsight_ml.common import ARTIFACTS, load_json, seed_everything

ARCHS = ["efficientnet_b0", "mobilenet_v3_large"]
VERSION = "1.0.0"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()

    seed_everything()
    audit.run()
    if not args.skip_train:
        for arch in ARCHS:
            train.run(arch, epochs=args.epochs)

    # Architecture is chosen on out-of-fold (development) AUC, never on the test set
    oof = {a: load_json(ARTIFACTS / a / "history.json")["summary"]["oof_auc"] for a in ARCHS}
    chosen = max(oof, key=oof.get)
    print(f"Out-of-fold AUC: {oof} -> deploying {chosen}")

    calibrate.run(chosen)
    export.run(chosen, version=f"{chosen}-ens5-v{VERSION}")
    evaluate.run(ARCHS)
