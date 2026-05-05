"""Select 50 high-value samples (40 pos + 10 neg) and copy to data/sample/.

Run from the project root:
    python scripts/build_sample_subset.py
"""

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
N_POSITIVE = 40
N_NEGATIVE = 10

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DB = PROJECT_ROOT / "data" / "contrail_samples.db"
DST_ROOT = PROJECT_ROOT / "data" / "sample" / "train"
DST_DB = PROJECT_ROOT / "data" / "sample" / "contrail_samples.db"

TABLE = "contrail_samples"
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    sample_id          TEXT PRIMARY KEY,
    frame_paths        TEXT NOT NULL,
    mask_path          TEXT NOT NULL,
    n_frames           INTEGER NOT NULL,
    image_height       INTEGER NOT NULL,
    image_width        INTEGER NOT NULL,
    mean_intensity     REAL NOT NULL,
    std_intensity      REAL NOT NULL,
    laplacian_variance REAL NOT NULL,
    edge_density       REAL NOT NULL,
    contrast_ratio     REAL NOT NULL,
    foreground_ratio   REAL NOT NULL,
    complexity_group   TEXT
)"""


def _load_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {TABLE}").fetchall()
    conn.close()
    return rows


def _copy_and_remap(row: sqlite3.Row) -> dict:
    """Copy sample directory to DST_ROOT and return updated row dict."""
    mask_src = Path(row["mask_path"])
    src_dir = mask_src.parent
    dst_dir = DST_ROOT / row["sample_id"]

    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        shutil.copy2(str(f), str(dst_dir / f.name))

    new_mask = str(dst_dir / mask_src.name)
    old_fps = json.loads(row["frame_paths"])
    new_fps = [str(dst_dir / Path(p).name) for p in old_fps]

    return {
        "sample_id": row["sample_id"],
        "frame_paths": json.dumps(new_fps),
        "mask_path": new_mask,
        "n_frames": row["n_frames"],
        "image_height": row["image_height"],
        "image_width": row["image_width"],
        "mean_intensity": row["mean_intensity"],
        "std_intensity": row["std_intensity"],
        "laplacian_variance": row["laplacian_variance"],
        "edge_density": row["edge_density"],
        "contrast_ratio": row["contrast_ratio"],
        "foreground_ratio": row["foreground_ratio"],
        "complexity_group": row["complexity_group"],
    }


def main() -> None:
    """Select, copy, and index the 50-sample demo subset."""
    all_rows = _load_rows(SRC_DB)
    positives = sorted(
        [r for r in all_rows if r["foreground_ratio"] > 0],
        key=lambda r: r["foreground_ratio"],
        reverse=True,
    )
    negatives = [r for r in all_rows if r["foreground_ratio"] == 0]

    rng = np.random.default_rng(SEED)
    selected_pos = positives[:N_POSITIVE]
    neg_idx = rng.choice(len(negatives), N_NEGATIVE, replace=False)
    selected_neg = [negatives[int(i)] for i in sorted(neg_idx)]
    selected = selected_pos + selected_neg

    print(f"Selecting {len(selected_pos)} positive + {len(selected_neg)} negative "
          f"= {len(selected)} samples")

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    if DST_DB.exists():
        DST_DB.unlink()

    conn = sqlite3.connect(DST_DB)
    conn.execute(DDL)

    cols_order = [
        "sample_id", "frame_paths", "mask_path", "n_frames",
        "image_height", "image_width", "mean_intensity", "std_intensity",
        "laplacian_variance", "edge_density", "contrast_ratio",
        "foreground_ratio", "complexity_group",
    ]
    placeholders = ", ".join(f":{c}" for c in cols_order)
    sql = f"INSERT OR REPLACE INTO {TABLE} ({', '.join(cols_order)}) VALUES ({placeholders})"

    for i, row in enumerate(selected, 1):
        new_row = _copy_and_remap(row)
        conn.execute(sql, new_row)
        print(f"  [{i:02d}/{len(selected)}] {row['sample_id']}  "
              f"fg={row['foreground_ratio']:.5f}")

    conn.commit()
    count = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    conn.close()
    print(f"\nDB written: {DST_DB}  ({count} entries)")


if __name__ == "__main__":
    main()
