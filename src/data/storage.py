"""Core data model (ContrailSample) and SQLite-backed persistent storage."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DB_FILE: Path = Path("data/contrail_samples.db")
TABLE_NAME: str = "contrail_samples"
BAND_FILES: Tuple[str, ...] = (
    "band_11.npy", "band_13.npy", "band_14.npy", "band_15.npy", "band_16.npy"
)
VALID_RECORDS_FILE: str = "valid_records.txt"
MASK_FILE: str = "human_pixel_masks.npy"
N_FRAMES: int = 8
MIDDLE_FRAME_IDX: int = 4
CONTRAST_EPS: float = 1e-6
EDGE_GRADIENT_THRESHOLD: float = 0.1
COMPLEXITY_LOW_QUANTILE: float = 33.3
COMPLEXITY_HIGH_QUANTILE: float = 66.7


# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class ContrailSample:
    """Single contrail segmentation sample with pre-computed metadata.

    Attributes:
        sample_id: Unique record identifier from the dataset.
        frame_paths: Paths to per-band .npy files [band_11, band_13, band_14, band_15, band_16].
        mask_path: Path to the binary segmentation mask .npy file.
        n_frames: Number of temporal frames stored in each band file.
        image_shape: Spatial dimensions (height, width).
        mean_intensity: Mean pixel intensity across all bands and frames.
        std_intensity: Standard deviation of pixel intensity.
        laplacian_variance: Variance of the Laplacian (edge sharpness proxy).
        edge_density: Fraction of pixels identified as edges via gradient.
        contrast_ratio: 99th-pct / (1st-pct + eps) intensity ratio.
        foreground_ratio: Fraction of mask pixels labelled 1 (EDA only,
            excluded from complexity features).
        complexity_group: Complexity tier ('low'/'mid'/'high').  None until
            :func:`assign_complexity_groups` is called.
    """

    sample_id: str
    frame_paths: List[Path]
    mask_path: Path
    n_frames: int
    image_shape: Tuple[int, int]
    mean_intensity: float
    std_intensity: float
    laplacian_variance: float
    edge_density: float
    contrast_ratio: float
    foreground_ratio: float
    complexity_group: Optional[str] = None


# ── Feature computation ───────────────────────────────────────────────────────
def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Linearly scale *arr* to [0, 1]; returns arr unchanged if constant."""
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + CONTRAST_EPS)


def _compute_features(
    bands: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    """Compute per-sample feature values from raw numpy arrays.

    Args:
        bands: Stacked band array of shape (n_bands, n_frames, H, W).
        mask: Binary mask array of shape (H, W).

    Returns:
        Dict mapping each feature name to its scalar value.
    """
    from scipy.ndimage import laplace  # noqa: PLC0415

    middle = bands[:, MIDDLE_FRAME_IDX, :, :]  # (n_bands, H, W)
    gray = middle.mean(axis=0)                  # (H, W)
    norm_all = _normalize_01(bands)
    gray_norm = _normalize_01(gray)

    gy, gx = np.gradient(gray_norm)
    grad_mag = np.sqrt(gy ** 2 + gx ** 2)
    p1, p99 = float(np.percentile(norm_all, 1)), float(np.percentile(norm_all, 99))

    return {
        "mean_intensity": float(norm_all.mean()),
        "std_intensity": float(norm_all.std()),
        "laplacian_variance": float(laplace(gray_norm).var()),
        "edge_density": float((grad_mag > EDGE_GRADIENT_THRESHOLD).mean()),
        "contrast_ratio": (p99 + CONTRAST_EPS) / (p1 + CONTRAST_EPS),
        "foreground_ratio": float(mask.mean()),
    }


def _build_one_sample(record_dir: Path) -> Optional[ContrailSample]:
    """Try to build a :class:`ContrailSample` from a single record directory.

    Args:
        record_dir: Directory for one record, containing band and mask files.

    Returns:
        A :class:`ContrailSample` on success, or ``None`` if files are missing
        or unreadable.
    """
    band_paths = [record_dir / f for f in BAND_FILES]
    mask_path = record_dir / MASK_FILE

    if not all(p.exists() for p in band_paths + [mask_path]):
        logger.debug("Skipping incomplete record: %s", record_dir.name)
        return None
    try:
        # Each band file: (H, W, 8); stack → (n_bands, H, W, 8);
        # transpose → (n_bands, 8, H, W) = (n_bands, n_frames, H, W)
        bands = np.stack([np.load(p) for p in band_paths]).transpose(0, 3, 1, 2)
        mask = np.load(mask_path).squeeze(-1).astype(np.float32)  # (H, W, 1) → (H, W)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", record_dir.name, exc)
        return None

    h, w = int(bands.shape[-2]), int(bands.shape[-1])
    return ContrailSample(
        sample_id=record_dir.name,
        frame_paths=band_paths,
        mask_path=mask_path,
        n_frames=N_FRAMES,
        image_shape=(h, w),
        **_compute_features(bands, mask),
    )


def _load_valid_ids(data_root: Path) -> Optional[set]:
    """Load allowed record IDs from ``valid_records.txt`` if the file exists.

    Args:
        data_root: Directory that may contain ``valid_records.txt``.

    Returns:
        Set of valid record ID strings, or ``None`` if the file is absent.
    """
    path = data_root / VALID_RECORDS_FILE
    if not path.exists():
        return None
    ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    logger.info("Loaded %d valid record IDs from %s", len(ids), path)
    return ids


def build_samples_from_dir(raw_dir: Path) -> List[ContrailSample]:
    """Scan a raw data directory and build :class:`ContrailSample` objects.

    Expected layout::

        raw_dir/<record_id>/band_11.npy            # (H, W, 8) float32
        raw_dir/<record_id>/band_13.npy            # (H, W, 8) float32
        raw_dir/<record_id>/band_14.npy            # (H, W, 8) float32
        raw_dir/<record_id>/band_15.npy            # (H, W, 8) float32
        raw_dir/<record_id>/band_16.npy            # (H, W, 8) float32
        raw_dir/<record_id>/human_pixel_masks.npy  # (H, W, 1) int32

    A ``valid_records.txt`` file in ``raw_dir.parent`` is used to filter out
    known-bad records when present.

    Args:
        raw_dir: Directory containing one sub-folder per record.

    Returns:
        List of :class:`ContrailSample` objects, one per valid record.
    """
    valid_ids = _load_valid_ids(raw_dir.parent)
    all_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())

    if valid_ids is not None:
        record_dirs = [d for d in all_dirs if d.name in valid_ids]
        logger.info(
            "Filtered to %d / %d records using valid_records.txt",
            len(record_dirs), len(all_dirs),
        )
    else:
        record_dirs = all_dirs
        logger.info("Scanning all %d record directories in %s", len(record_dirs), raw_dir)

    samples = [s for rd in record_dirs if (s := _build_one_sample(rd)) is not None]
    logger.info("Built %d samples from %s", len(samples), raw_dir)
    return samples


def assign_complexity_groups(
    samples: List[ContrailSample],
) -> List[ContrailSample]:
    """Assign low/mid/high complexity groups via 33/66 percentile binning.

    Uses image-derived features only; foreground_ratio is excluded to
    prevent data leakage from ground-truth labels.  Modifies samples in-place.

    Args:
        samples: List of :class:`ContrailSample` objects to annotate.

    Returns:
        The same list with ``complexity_group`` set on every sample.
    """
    feature_names = [
        "mean_intensity",
        "std_intensity",
        "laplacian_variance",
        "edge_density",
        "contrast_ratio",
    ]
    mat = np.array([[getattr(s, f) for f in feature_names] for s in samples])

    # Composite complexity score: average normalised rank across all features
    scores = np.zeros(len(samples))
    for col in range(mat.shape[1]):
        norm_ranks = mat[:, col].argsort().argsort().astype(float)
        scores += norm_ranks / max(len(samples) - 1, 1)

    q33, q66 = np.percentile(scores, [COMPLEXITY_LOW_QUANTILE, COMPLEXITY_HIGH_QUANTILE])
    for sample, score in zip(samples, scores):
        sample.complexity_group = "low" if score <= q33 else ("mid" if score <= q66 else "high")

    return samples


# ── SQLite store ──────────────────────────────────────────────────────────────
class ContrailSampleStore:
    """Persistent SQLite-backed store for :class:`ContrailSample` objects.

    Args:
        db_path: Filesystem path for the SQLite database file.
    """

    _DDL: str = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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

    def __init__(self, db_path: Path = DB_FILE) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(self._DDL)

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection with row_factory configured."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _to_row(s: ContrailSample) -> dict:
        """Serialise a ContrailSample to a plain dict for SQLite insertion."""
        return {
            "sample_id": s.sample_id,
            "frame_paths": json.dumps([str(p) for p in s.frame_paths]),
            "mask_path": str(s.mask_path),
            "n_frames": s.n_frames,
            "image_height": s.image_shape[0],
            "image_width": s.image_shape[1],
            "mean_intensity": s.mean_intensity,
            "std_intensity": s.std_intensity,
            "laplacian_variance": s.laplacian_variance,
            "edge_density": s.edge_density,
            "contrast_ratio": s.contrast_ratio,
            "foreground_ratio": s.foreground_ratio,
            "complexity_group": s.complexity_group,
        }

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ContrailSample:
        """Deserialise a SQLite row back into a ContrailSample."""
        return ContrailSample(
            sample_id=row["sample_id"],
            frame_paths=[Path(p) for p in json.loads(row["frame_paths"])],
            mask_path=Path(row["mask_path"]),
            n_frames=row["n_frames"],
            image_shape=(row["image_height"], row["image_width"]),
            mean_intensity=row["mean_intensity"],
            std_intensity=row["std_intensity"],
            laplacian_variance=row["laplacian_variance"],
            edge_density=row["edge_density"],
            contrast_ratio=row["contrast_ratio"],
            foreground_ratio=row["foreground_ratio"],
            complexity_group=row["complexity_group"],
        )

    def upsert_many(self, samples: Sequence[ContrailSample]) -> None:
        """Bulk insert or replace samples.

        Args:
            samples: Collection of :class:`ContrailSample` objects.
        """
        rows = [self._to_row(s) for s in samples]
        if not rows:
            return
        cols = ", ".join(rows[0])
        placeholders = ", ".join(f":{k}" for k in rows[0])
        sql = f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.executemany(sql, rows)
        logger.info("Upserted %d samples into the database", len(rows))

    def get_all(self) -> List[ContrailSample]:
        """Return every stored sample.

        Returns:
            List of all :class:`ContrailSample` objects in the database.
        """
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM {TABLE_NAME}").fetchall()
        return [self._from_row(r) for r in rows]

    def count(self) -> int:
        """Return the number of samples currently stored."""
        with self._connect() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    def update_complexity_groups(self, updates: Dict[str, str]) -> None:
        """Bulk-update complexity_group for a mapping of sample_id → group.

        Args:
            updates: Mapping from sample_id to group label ('low'/'mid'/'high').
        """
        sql = f"UPDATE {TABLE_NAME} SET complexity_group = ? WHERE sample_id = ?"
        with self._connect() as conn:
            conn.executemany(sql, [(g, sid) for sid, g in updates.items()])
        logger.info("Updated complexity groups for %d samples", len(updates))
