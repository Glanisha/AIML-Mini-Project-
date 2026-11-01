"""R8: package the model bundle served by the backend (ONNX + config), and verify parity."""

import datetime as dt
import shutil

import numpy as np
import onnxruntime as ort
import torch

from app.inference.preprocess import IMAGE_SIZE, MEAN, STD

from .common import ARTIFACTS, BUNDLE_DIR, CLASSES, DATASET_CITATION, N_FOLDS, POSITIVE_CLASS, load_json, save_json
from .data import eval_tensor, load_manifest
from .models import CamExport, CamNet, FrozenEncoder

OPSET = 17


def _export(module: torch.nn.Module, path, output_names: list[str]) -> None:
    module.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dynamic = {"input": {0: "batch"}, **{n: {0: "batch"} for n in output_names}}
    kwargs = dict(input_names=["input"], output_names=output_names, dynamic_axes=dynamic, opset_version=OPSET)
    try:
        torch.onnx.export(module, (dummy,), str(path), dynamo=False, **kwargs)
    except Exception:
        # Newer PyTorch releases may only ship the dynamo-based exporter
        torch.onnx.export(module, (dummy,), str(path), dynamo=True, external_data=False, **kwargs)


def run(arch: str, version: str) -> dict:
    calib = load_json(ARTIFACTS / "calibration.json")
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    # Parity check inputs: a few real dev images through the deployed preprocessing
    df = load_manifest()
    probe = eval_tensor(df[df.split == "dev"].path.head(4).tolist())

    members, max_diff = [], 0.0
    for k in range(N_FOLDS):
        ckpt = torch.load(ARTIFACTS / arch / f"fold{k}.pt", map_location="cpu")
        net = CamNet(arch, pretrained=False)
        net.load_state_dict(ckpt["state_dict"])
        name = f"member_{k}.onnx"
        _export(CamExport(net), BUNDLE_DIR / name, ["logits", "cam"])
        members.append(name)

        with torch.no_grad():
            ref = net.eval()(probe).numpy()
        got = ort.InferenceSession(str(BUNDLE_DIR / name), providers=["CPUExecutionProvider"]).run(None, {"input": probe.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))

    _export(FrozenEncoder(), BUNDLE_DIR / "ood_backbone.onnx", ["embedding"])
    shutil.copy(ARTIFACTS / "ood_stats.npz", BUNDLE_DIR / "ood.npz")

    config = {
        "version": version,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "architecture": arch,
        "classes": CLASSES,
        "positive_class": POSITIVE_CLASS,
        "members": members,
        "input": {"size": IMAGE_SIZE, "mean": MEAN.tolist(), "std": STD.tolist(), "resize": "bilinear, whole image"},
        "calibration": {"method": "temperature scaling (out-of-fold)", "temperature": calib["temperature"]},
        "decision": {
            "operating_threshold": calib["operating_threshold"],
            "target_sensitivity": calib["target_sensitivity"],
            "abstain_margin": calib["abstain_margin"],
            "min_member_agreement": calib["min_member_agreement"],
        },
        "quality": calib["quality"],
        "ood": {
            "method": "k-NN cosine distance (k=5) to training images in frozen ImageNet EfficientNet-B0 embedding space",
            "backbone": "ood_backbone.onnx",
            "stats": "ood.npz",
            "threshold": calib["ood"]["threshold"],
        },
        "training_data": DATASET_CITATION,
        "onnx_parity_max_abs_diff": max_diff,
    }
    save_json(config, BUNDLE_DIR / "config.json")
    size_mb = sum(f.stat().st_size for f in BUNDLE_DIR.iterdir()) / 1e6
    print(f"Bundle {version}: {len(members)} members, ONNX parity max |diff| {max_diff:.2e}, {size_mb:.1f} MB")
    return config
