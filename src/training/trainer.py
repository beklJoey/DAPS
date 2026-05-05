"""Training infrastructure: dataset, data loaders, training loop, multi-seed runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib  # pylint: disable=wrong-import-order
matplotlib.use("Agg")  # must precede pyplot import  # pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import torch  # pylint: disable=wrong-import-position
import torch.nn as nn  # pylint: disable=wrong-import-position
from torch.utils.data import DataLoader, Dataset  # pylint: disable=wrong-import-position

from ..data.storage import ContrailSample, MIDDLE_FRAME_IDX, N_FRAMES  # pylint: disable=wrong-import-position
from ..models import build_model  # pylint: disable=wrong-import-position
from ..utils.logger import get_logger  # pylint: disable=wrong-import-position
from ..utils.seed import SEEDS, get_generator, set_seed  # pylint: disable=wrong-import-position
from .losses import build_loss  # pylint: disable=wrong-import-position

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
# remainder → test

N_EPOCHS: int = 5
BATCH_SIZE: int = 8
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-4
NUM_WORKERS: int = 0          # 0 = main-process loading (safe on Windows)
IOU_THRESHOLD: float = 0.5    # default binarisation threshold during training
FIGURES_DIR: Path = Path("figures/training")
CHECKPOINTS_DIR: Path = Path("checkpoints")
BANDS_PER_FRAME: int = 5   # band_11, band_13, band_14, band_15, band_16


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """All hyper-parameters for a single training run.

    Attributes:
        model_name: ``'unet'`` or ``'temporal_unet'``.
        loss_name: ``'bce'``, ``'dice'``, or ``'bce_dice'``.
        n_frames: Number of temporal frames to stack (1, 3, or 5).
        use_augmentation: Whether to apply data augmentation.
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial Adam learning rate.
        weight_decay: L2 regularisation coefficient.
        seeds: List of random seeds for multi-seed runs.
        run_name: Human-readable experiment label (used in filenames).
    """

    model_name: str = "unet"
    loss_name: str = "bce_dice"
    n_frames: int = 1
    use_augmentation: bool = True
    n_epochs: int = N_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    seeds: List[int] = field(default_factory=lambda: list(SEEDS))
    run_name: str = "experiment"


# ── Dataset ───────────────────────────────────────────────────────────────────
def _select_frame_indices(n_frames: int, total_frames: int = N_FRAMES) -> List[int]:
    """Choose T frame indices centred on the middle (labelled) frame.

    Args:
        n_frames: Number of frames T to select (1, 3, or 5).
        total_frames: Total frames available in each band file.

    Returns:
        Sorted list of T frame indices.
    """
    mid = MIDDLE_FRAME_IDX
    half = n_frames // 2
    indices = list(range(mid - half, mid - half + n_frames))
    indices = [max(0, min(i, total_frames - 1)) for i in indices]
    return indices


def _load_and_stack(
    sample: ContrailSample,
    frame_indices: List[int],
    channel_mean: Optional[np.ndarray],
    channel_std: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load band .npy files and assemble a stacked image tensor.

    Each band file has shape ``(H, W, 8)``; five bands are stacked and
    T frames are selected, producing a ``(T*5, H, W)`` tensor.

    Args:
        sample: Source sample with band file paths.
        frame_indices: Frame indices to select (centred on middle frame).
        channel_mean: Per-band mean array ``(5,)``; ``None`` → skip normalisation.
        channel_std: Per-band std array ``(5,)``; ``None`` → skip normalisation.

    Returns:
        ``(image, mask)`` — image ``(T*5, H, W)`` float32, mask ``(H, W)`` float32.
    """
    # Stack: (5, H, W, 8); select T frames → (5, H, W, T)
    bands = np.stack([np.load(p) for p in sample.frame_paths])  # (5, H, W, 8)
    frames = bands[:, :, :, frame_indices].astype(np.float32)   # (5, H, W, T)

    # Normalize each band to [0, 1] independently
    for b in range(frames.shape[0]):
        b_min, b_max = frames[b].min(), frames[b].max()
        frames[b] = (frames[b] - b_min) / (b_max - b_min + 1e-6)

    # Optional per-band standardisation using training-set statistics
    if channel_mean is not None and channel_std is not None:
        for b in range(frames.shape[0]):
            frames[b] = (frames[b] - channel_mean[b]) / (channel_std[b] + 1e-6)

    # (5, H, W, T) → (T, 5, H, W) → (T*5, H, W)
    n_t = len(frame_indices)
    h, w = frames.shape[1], frames.shape[2]
    image = frames.transpose(3, 0, 1, 2).reshape(n_t * BANDS_PER_FRAME, h, w)

    # Mask: (H, W, 1) int32 → (H, W) float32
    mask = np.load(sample.mask_path).squeeze(-1).astype(np.float32)
    return image, mask


class ContrailDataset(Dataset):
    """PyTorch Dataset for contrail segmentation samples.

    Args:
        samples: List of :class:`~src.data.storage.ContrailSample` objects.
        n_frames: Number of temporal frames to stack per sample.
        channel_mean: Per-band normalisation mean ``(3,)`` or ``None``.
        channel_std: Per-band normalisation std ``(3,)`` or ``None``.
        augment: Apply training-time spatial augmentations when ``True``.
    """

    def __init__(
        self,
        samples: List[ContrailSample],
        n_frames: int = 1,
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.frame_indices = _select_frame_indices(n_frames)
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.augment = augment

    def __len__(self) -> int:  # noqa: D105
        return len(self.samples)

    def _apply_augmentation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial and photometric augmentations consistently to image and mask."""
        try:
            import albumentations as A
        except ImportError:
            return image, mask

        # Build spatial-only pipeline applied to both image and mask
        spatial = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        # Albumentations expects HWC; image is (C, H, W) → transpose
        img_hwc = image.transpose(1, 2, 0)
        result = spatial(image=img_hwc, mask=mask)
        img_hwc, mask = result["image"], result["mask"]

        # Photometric on image channels only (brightness/contrast jitter)
        photo = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
        ])
        img_hwc = photo(image=img_hwc)["image"]
        return img_hwc.transpose(2, 0, 1), mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        sample = self.samples[idx]
        image, mask = _load_and_stack(
            sample, self.frame_indices, self.channel_mean, self.channel_std
        )
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        return torch.from_numpy(image), torch.from_numpy(mask)


# ── Split helpers ─────────────────────────────────────────────────────────────
def split_samples(
    samples: List[ContrailSample], seed: int = 42
) -> Tuple[List[ContrailSample], List[ContrailSample], List[ContrailSample]]:
    """Record-level 80/10/10 train/val/test split.

    Args:
        samples: Full list of samples (all records).
        seed: Random seed for reproducible shuffling.

    Returns:
        ``(train_samples, val_samples, test_samples)``
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples)).tolist()
    n_train = int(len(samples) * TRAIN_RATIO)
    n_val = int(len(samples) * VAL_RATIO)
    train = [samples[i] for i in indices[:n_train]]
    val = [samples[i] for i in indices[n_train: n_train + n_val]]
    test = [samples[i] for i in indices[n_train + n_val:]]
    logger.info("Split: %d train / %d val / %d test", len(train), len(val), len(test))
    return train, val, test


def compute_channel_stats(
    samples: List[ContrailSample],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-band mean and std from the training set.

    Each band file has shape ``(H, W, 8)``.  Values are first normalised to
    [0, 1] per file, then global statistics are accumulated in one pass using
    vectorised numpy sums (no per-element Python loop).

    Args:
        samples: Training samples used to derive statistics.

    Returns:
        ``(mean, std)`` each shaped ``(5,)`` — one value per satellite band.
    """
    sums = np.zeros(BANDS_PER_FRAME, dtype=np.float64)
    sum_sqs = np.zeros(BANDS_PER_FRAME, dtype=np.float64)
    counts = np.zeros(BANDS_PER_FRAME, dtype=np.float64)

    for sample in samples:
        for b, band_path in enumerate(sample.frame_paths):
            arr = np.load(band_path).astype(np.float64)   # (H, W, 8)
            b_min, b_max = arr.min(), arr.max()
            arr = (arr - b_min) / (b_max - b_min + 1e-6)
            sums[b] += arr.sum()
            sum_sqs[b] += (arr ** 2).sum()
            counts[b] += arr.size

    means = sums / counts
    stds = np.sqrt(np.maximum(sum_sqs / counts - means ** 2, 0.0))
    return means.astype(np.float32), stds.astype(np.float32)


def build_dataloaders(
    train_samples: List[ContrailSample],
    val_samples: List[ContrailSample],
    test_samples: List[ContrailSample],
    config: TrainingConfig,
    channel_mean: Optional[np.ndarray] = None,
    channel_std: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders.

    Args:
        train_samples: Training split samples.
        val_samples: Validation split samples.
        test_samples: Test split samples.
        config: Training configuration controlling frames and augmentation.
        channel_mean: Per-band normalisation mean (from training set).
        channel_std: Per-band normalisation std (from training set).
        seed: Random seed for DataLoader generator.

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    gen = get_generator(seed)

    def _make(samples: List[ContrailSample], augment: bool) -> DataLoader:
        ds = ContrailDataset(
            samples, config.n_frames, channel_mean, channel_std, augment=augment
        )
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=augment,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            generator=gen if augment else None,
            drop_last=augment,
        )

    return (
        _make(train_samples, augment=config.use_augmentation),
        _make(val_samples, augment=False),
        _make(test_samples, augment=False),
    )


# ── Metric helpers ────────────────────────────────────────────────────────────
def _batch_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    """Compute mean IoU over a batch.

    Args:
        logits: Raw model output ``(B, 1, H, W)``.
        targets: Binary masks ``(B, H, W)`` or ``(B, 1, H, W)``.
        threshold: Sigmoid threshold for binarisation.

    Returns:
        Mean IoU across the batch (scalar float).
    """
    preds = (torch.sigmoid(logits) > threshold).float().squeeze(1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(iou.mean().item())


# ── Epoch loops ───────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full training epoch.

    Args:
        model: Model to train (set to training mode internally).
        loader: Training data loader.
        optimizer: Parameter optimiser.
        criterion: Loss function.
        device: Computation device.

    Returns:
        ``(mean_loss, mean_iou)`` over all batches.
    """
    model.train()
    total_loss = total_iou = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += _batch_iou(logits.detach(), masks, IOU_THRESHOLD)

    n = max(len(loader), 1)
    return total_loss / n, total_iou / n


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full evaluation epoch.

    Args:
        model: Model to evaluate (set to eval mode internally).
        loader: Evaluation data loader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        ``(mean_loss, mean_iou)`` over all batches.
    """
    model.eval()
    total_loss = total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            total_iou += _batch_iou(logits, masks, IOU_THRESHOLD)

    n = max(len(loader), 1)
    return total_loss / n, total_iou / n


# ── Single-seed training ──────────────────────────────────────────────────────
def _save_curves(history: Dict[str, List[float]], run_name: str, seed: int) -> None:
    """Persist loss and IoU training curves to ``figures/training/``.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``,
            ``train_iou``, ``val_iou`` — each a list of per-epoch values.
        run_name: Experiment name used in the filename.
        seed: Seed value used in the filename.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_iou"], label="train")
    axes[1].plot(epochs, history["val_iou"], label="val")
    axes[1].set_title("IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(f"{run_name} | seed={seed}")
    fig.tight_layout()
    save_path = FIGURES_DIR / f"{run_name}_seed{seed}.png"
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved training curve → %s", save_path)


def _step_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Execute one train + val epoch step.

    Returns:
        ``(tr_loss, tr_iou, vl_loss, vl_iou)``
    """
    tr_loss, tr_iou = train_epoch(model, train_loader, optimizer, criterion, device)
    vl_loss, vl_iou = eval_epoch(model, val_loader, criterion, device)
    scheduler.step(vl_iou)
    return tr_loss, tr_iou, vl_loss, vl_iou


def _run_epoch_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    seed: int,
) -> Tuple[Dict[str, List[float]], Optional[dict]]:
    """Iterate over all epochs; return per-epoch history and best state dict."""
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []
    }
    best_val_iou, best_state = -1.0, None

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_iou, vl_loss, vl_iou = _step_epoch(
            model, train_loader, val_loader, optimizer, scheduler, criterion, device
        )
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(vl_iou)

        if vl_iou > best_val_iou:
            best_val_iou = vl_iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            "[%s|seed%d] %02d/%02d  tr=%.4f/%.4f  vl=%.4f/%.4f  (%.1fs)",
            config.run_name, seed, epoch, config.n_epochs,
            tr_loss, tr_iou, vl_loss, vl_iou, time.time() - t0,
        )

    return history, best_state


def _save_best_checkpoint(
    model: nn.Module, best_state: Optional[dict], run_name: str, seed: int
) -> None:
    """Load *best_state* into *model* and persist the checkpoint to disk.

    Args:
        model: Model whose weights are updated.
        best_state: State dict recorded at the best validation epoch.
        run_name: Experiment name used in the filename.
        seed: Seed value used in the filename.
    """
    if best_state is None:
        return
    model.load_state_dict(best_state)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / f"{run_name}_seed{seed}_best.pt"
    torch.save(best_state, ckpt_path)
    logger.info("Saved best checkpoint → %s", ckpt_path)


def run_training_one_seed(
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Train a model with a single seed and return test metrics.

    Args:
        config: Training configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        device: Computation device.
        seed: Random seed (applied before model initialisation).

    Returns:
        Dict of final test metrics: ``test_loss``, ``test_iou``.
    """
    set_seed(seed)
    model = build_model(config.model_name, config.n_frames).to(device)
    criterion = build_loss(config.loss_name).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    history, best_state = _run_epoch_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        criterion, device, config, seed,
    )
    _save_curves(history, config.run_name, seed)
    _save_best_checkpoint(model, best_state, config.run_name, seed)

    test_loss, test_iou = eval_epoch(model, test_loader, criterion, device)
    logger.info("[%s|seed%d] TEST  loss=%.4f  iou=%.4f", config.run_name, seed, test_loss, test_iou)
    return {"test_loss": test_loss, "test_iou": test_iou}


# ── Multi-seed orchestrator ───────────────────────────────────────────────────
def _aggregate_seed_results(
    all_metrics: List[Dict[str, float]], run_name: str
) -> Dict[str, float]:
    """Compute mean ± std across per-seed metric dicts.

    Args:
        all_metrics: List of metric dicts, one per seed.
        run_name: Experiment name (for logging).

    Returns:
        Dict with ``mean_<key>`` and ``std_<key>`` entries.
    """
    summary: Dict[str, float] = {}
    for key in all_metrics[0]:
        values = np.array([m[key] for m in all_metrics])
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std())
        logger.info(
            "[%s] %s = %.4f ± %.4f",
            run_name, key, summary[f"mean_{key}"], summary[f"std_{key}"],
        )
    return summary


def train_with_seeds(
    config: TrainingConfig,
    samples: List[ContrailSample],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Train a model with multiple seeds and report mean ± std.

    Performs a record-level 80/10/10 split, computes per-band normalisation
    statistics from the training set, then runs :func:`run_training_one_seed`
    for each seed in ``config.seeds``.

    Args:
        config: Training configuration.
        samples: Full pool of :class:`~src.data.storage.ContrailSample` objects.
        device: Computation device; auto-detected when ``None``.

    Returns:
        Dict with ``mean_<metric>`` and ``std_<metric>`` keys.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running %s on %s (%d seeds)", config.run_name, device, len(config.seeds))

    train_s, val_s, test_s = split_samples(samples, seed=config.seeds[0])
    logger.info("Computing channel normalisation statistics …")
    ch_mean, ch_std = compute_channel_stats(train_s)

    all_metrics: List[Dict[str, float]] = []
    for seed in config.seeds:
        tr_loader, vl_loader, ts_loader = build_dataloaders(
            train_s, val_s, test_s, config, ch_mean, ch_std, seed=seed
        )
        all_metrics.append(
            run_training_one_seed(config, tr_loader, vl_loader, ts_loader, device, seed)
        )

    return _aggregate_seed_results(all_metrics, config.run_name)
