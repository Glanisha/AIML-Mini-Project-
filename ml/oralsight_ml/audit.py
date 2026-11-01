"""R1/R2: dataset audit, duplicate grouping and patient-proxy splits.

The dataset has no patient IDs, so near-duplicate photos (same lesion, re-shot or
re-saved) are clustered with a perceptual hash and kept in the same split.
"""

import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from .common import ARTIFACTS, CLASSES, DATA_RAW, N_FOLDS, REPORTS, SEED, save_json

NEAR_DUP_HAMMING = 6  # out of 64 bits; empirically flags re-saves/crops of the same photo


def dhash(image: Image.Image, size: int = 8) -> int:
    g = np.asarray(image.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def build_manifest() -> pd.DataFrame:
    rows = []
    for label in CLASSES:
        for path in sorted((DATA_RAW / label).iterdir()):
            raw = path.read_bytes()
            with Image.open(path) as im:
                fmt = im.format
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                hashes = [dhash(im), dhash(ImageOps.mirror(im))]
            rows.append(
                {
                    "path": str(path.relative_to(DATA_RAW)).replace("\\", "/"),
                    "label": label,
                    "y": CLASSES.index(label),
                    "md5": hashlib.md5(raw).hexdigest(),
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "bytes": len(raw),
                    "dhash": hashes[0],
                    "dhash_mirror": hashes[1],
                }
            )
    return pd.DataFrame(rows)


def group_near_duplicates(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Union-find over pairs whose dHash (or mirrored dHash) is within NEAR_DUP_HAMMING."""
    n = len(df)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h = df["dhash"].to_numpy(dtype=object)
    hm = df["dhash_mirror"].to_numpy(dtype=object)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(bin(h[i] ^ h[j]).count("1"), bin(h[i] ^ hm[j]).count("1"))
            if d <= NEAR_DUP_HAMMING or df.md5.iat[i] == df.md5.iat[j]:
                pairs.append((i, j, d))
                parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group, pairs


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Hold out ~20% as a locked test set, then 5 stratified, grouped CV folds on the rest."""
    df = df.copy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer.split(df, df.y, df.group))
    df["split"] = "dev"
    df.loc[df.index[test_idx], "split"] = "test"
    df["fold"] = -1

    dev = df[df.split == "dev"]
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, val_idx) in enumerate(inner.split(dev, dev.y, dev.group)):
        df.loc[dev.index[val_idx], "fold"] = k
    return df


def run() -> pd.DataFrame:
    df = build_manifest()
    df["group"], pairs = group_near_duplicates(df)

    conflicts = [(i, j) for i, j, _ in pairs if df.label.iat[i] != df.label.iat[j]]
    exact = [(i, j) for i, j, _ in pairs if df.md5.iat[i] == df.md5.iat[j]]
    df = make_splits(df)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ARTIFACTS / "manifest.csv", index=False)
    df[["path", "label", "group", "split", "fold"]].to_csv(REPORTS / "splits.csv", index=False)

    # Each duplicate group must sit entirely inside one split/fold
    leak = df.groupby("group")[["split", "fold"]].nunique().max().max() > 1
    group_sizes = Counter(Counter(df.group).values())

    audit = {
        "source": "Mendeley Data mhjyrn35p4 v2: original_data only (provided augmented_data excluded to prevent leakage)",
        "licence": "CC BY 4.0",
        "n_images": len(df),
        "class_counts": df.label.value_counts().to_dict(),
        "formats": df.format.value_counts().to_dict(),
        "width": {"min": int(df.width.min()), "median": int(df.width.median()), "max": int(df.width.max())},
        "height": {"min": int(df.height.min()), "median": int(df.height.median()), "max": int(df.height.max())},
        "patient_ids_available": False,
        "exact_duplicate_pairs": len(exact),
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_hamming_threshold": NEAR_DUP_HAMMING,
        "cross_label_duplicate_pairs": [[df.path.iat[i], df.path.iat[j]] for i, j in conflicts],
        "n_groups": int(df.group.nunique()),
        "group_size_histogram": {int(k): int(v) for k, v in sorted(group_sizes.items())},
        "split_counts": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "fold_counts": df[df.split == "dev"].groupby(["fold", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "group_leakage_across_splits": bool(leak),
        "duplicate_pairs": [[df.path.iat[i], df.path.iat[j], int(d)] for i, j, d in pairs],
    }
    save_json(audit, REPORTS / "dataset_audit.json")
    print(f"Audit: {len(df)} images, {len(pairs)} near-duplicate pairs, {len(conflicts)} cross-label, leakage={leak}")
    print(df.groupby(["split", "label"]).size())
    return df


if __name__ == "__main__":
    run()
