"""File integrity checks and image/mask quality validation."""

import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .storage import ContrailSample
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 65_536        # 64 KiB read buffer for hashing
MIN_SPATIAL_DIM: int = 16       # minimum acceptable H or W
MASK_ALLOWED_VALUES: frozenset = frozenset({0, 1})


# ── Low-level helpers ─────────────────────────────────────────────────────────
def hash_file(path: Path, algorithm: str = "md5") -> str:
    """Compute the hex digest of a file.

    Args:
        path: Path to the file to hash.
        algorithm: Hash algorithm name recognised by :mod:`hashlib`.

    Returns:
        Hexadecimal digest string.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def validate_npy_image(path: Path) -> Tuple[bool, str]:
    """Check that a .npy file is a plausible image or image stack.

    Conditions:
    - Loadable with numpy
    - ndim in {2, 3, 4}
    - First two axes (H, W) >= MIN_SPATIAL_DIM
    - All values are finite

    Band files are stored as ``(H, W, T)``; spatial dims are the first two axes.

    Args:
        path: Path to the .npy file.

    Returns:
        ``(True, '')`` on success or ``(False, reason)`` on failure.
    """
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        return False, f"Cannot load {path}: {exc}"

    if arr.ndim not in {2, 3, 4}:
        return False, f"Unexpected ndim={arr.ndim} in {path}"

    h, w = arr.shape[0], arr.shape[1]
    if h < MIN_SPATIAL_DIM or w < MIN_SPATIAL_DIM:
        return False, f"Spatial dims ({h}, {w}) below minimum {MIN_SPATIAL_DIM} in {path}"

    if not np.isfinite(arr).all():
        return False, f"Non-finite values found in {path}"

    return True, ""


def validate_npy_mask(path: Path) -> Tuple[bool, str]:
    """Check that a .npy file is a valid binary segmentation mask.

    Conditions:
    - Loadable with numpy
    - ndim in {2, 3}
    - Only values in {0, 1}
    - All values are finite

    Args:
        path: Path to the .npy mask file.

    Returns:
        ``(True, '')`` on success or ``(False, reason)`` on failure.
    """
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        return False, f"Cannot load {path}: {exc}"

    if arr.ndim not in {2, 3}:
        return False, f"Unexpected ndim={arr.ndim} in {path}"

    unique = set(np.unique(arr).astype(int).tolist())
    extra = unique - MASK_ALLOWED_VALUES
    if extra:
        return False, f"Non-binary values {extra} found in mask {path}"

    if not np.isfinite(arr).all():
        return False, f"Non-finite values found in {path}"

    return True, ""


# ── Sample-level validation ───────────────────────────────────────────────────
def validate_sample(sample: ContrailSample) -> Tuple[bool, List[str]]:
    """Run all integrity checks for a single :class:`ContrailSample`.

    Args:
        sample: The sample to validate.

    Returns:
        ``(all_valid, list_of_error_messages)`` — list is empty when valid.
    """
    errors: List[str] = []

    for fp in sample.frame_paths:
        if not fp.exists():
            errors.append(f"Missing frame file: {fp}")
            continue
        ok, reason = validate_npy_image(fp)
        if not ok:
            errors.append(reason)

    if not sample.mask_path.exists():
        errors.append(f"Missing mask file: {sample.mask_path}")
    else:
        ok, reason = validate_npy_mask(sample.mask_path)
        if not ok:
            errors.append(reason)

    return len(errors) == 0, errors


# ── Dataset-level validation ──────────────────────────────────────────────────
def validate_dataset(
    samples: List[ContrailSample],
    strict: bool = False,
) -> List[ContrailSample]:
    """Validate all samples and return only those that pass every check.

    Args:
        samples: Candidates to validate.
        strict: When ``True``, raises :class:`ValueError` if any sample fails.

    Returns:
        Filtered list containing only valid samples.

    Raises:
        ValueError: If *strict* is ``True`` and at least one sample is invalid.
    """
    valid: List[ContrailSample] = []
    n_invalid = 0

    for sample in samples:
        ok, errors = validate_sample(sample)
        if ok:
            valid.append(sample)
        else:
            n_invalid += 1
            for msg in errors:
                logger.warning("[%s] %s", sample.sample_id, msg)

    logger.info(
        "Validation: %d valid, %d invalid (total %d)",
        len(valid), n_invalid, len(samples),
    )

    if strict and n_invalid:
        raise ValueError(
            f"Dataset validation failed: {n_invalid} invalid sample(s)"
        )

    return valid
