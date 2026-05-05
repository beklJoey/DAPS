"""Dataset acquisition: resolve local data path via environment variable."""

import os
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH_ENV: str = "DATA_PATH"
DEFAULT_DATA_PATH: Path = Path(r"D:\AMLSII_data\contrails_subset")


def acquire_dataset() -> Path:
    """Resolve and validate the local dataset path.

    Reads the ``DATA_PATH`` environment variable when set; otherwise falls
    back to the default ``D:\\AMLSII_data``.  No downloading or copying is
    performed — the data is used in-place.

    Returns:
        Absolute path to the dataset root directory.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    env_val = os.environ.get(DATA_PATH_ENV)
    data_path = Path(env_val) if env_val else DEFAULT_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path}\n"
            f"Set the {DATA_PATH_ENV} environment variable to the correct path."
        )

    logger.info("Using dataset at %s", data_path)
    return data_path
