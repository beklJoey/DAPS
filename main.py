"""Single entry point for all contrail-segmentation experiments.

Runs 7 experiments sequentially from a single command::

    python main.py              # run all experiments (skips completed ones)
    python main.py --force      # re-run everything even if checkpoints exist
    python main.py --exp 4 7    # run only experiments 4 and 7
    python main.py --smoke-test # CI: Exp 1 + Exp 5 only, 2 epochs each

Each experiment checks for an existing checkpoint and skips if found (unless
``--force`` is set).  Results are appended to
``artifacts/tables/final_corrected_model_comparison.csv``.

Experiments
-----------
1. UNet + BCE                         (baseline)
2. UNet + BCE-Dice                    (loss ablation)
3. TemporalUNet T=3                   (temporal context)
4. UNet + BCE + pos_weight=136        (class-imbalance handling)
5. UNet + Dice + BCE (0.5/0.5)        (combined loss, no pos_weight)
6. UNet + BCE + pos_weight + aug+samp (augmentation + weighted sampler)
7. UNet + BCE + pos_weight + 500 extra samples  (expanded dataset)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent
DB_PATH: Path = PROJECT_ROOT / "data" / "sample" / "contrail_samples.db"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
PYTHON: str = sys.executable

# Experiment definitions: id, display name, script, checkpoint path
@dataclass
class ExperimentDef:
    """Metadata for a single training experiment.

    Attributes:
        exp_id: Integer identifier (1–7).
        name: Human-readable display name.
        script: Path to the Python training script relative to project root.
        checkpoint: Path to the saved best-model checkpoint.
    """

    exp_id: int
    name: str
    script: Path
    checkpoint: Path


EXPERIMENTS: list[ExperimentDef] = [
    ExperimentDef(
        exp_id=1,
        name="UNet + BCE (orig)",
        script=PROJECT_ROOT / "scripts" / "train_exp1_unet_bce.py",
        checkpoint=ARTIFACTS_DIR / "models" / "unet_bce_seed42_best.pt",
    ),
    ExperimentDef(
        exp_id=2,
        name="UNet + BCE-Dice (orig)",
        script=PROJECT_ROOT / "scripts" / "train_exp2_unet_bce_dice.py",
        checkpoint=ARTIFACTS_DIR / "models" / "unet_bce_dice_seed42_best.pt",
    ),
    ExperimentDef(
        exp_id=3,
        name="TemporalUNet T=3",
        script=PROJECT_ROOT / "scripts" / "retrain_temporal_unet.py",
        checkpoint=PROJECT_ROOT / "artifacts" / "models"
                   / "temporal_unet_t3_epoch30_seed42_best.pt",
    ),
    ExperimentDef(
        exp_id=4,
        name="UNet + BCE + pos_weight=136",
        script=PROJECT_ROOT / "scripts" / "train_exp4_unet_pw136.py",
        checkpoint=PROJECT_ROOT / "artifacts" / "models"
                   / "unet_bce_pw136_seed42_best.pt",
    ),
    ExperimentDef(
        exp_id=5,
        name="UNet + Dice + BCE (0.5/0.5)",
        script=PROJECT_ROOT / "scripts" / "train_exp5_unet_dice_bce.py",
        checkpoint=PROJECT_ROOT / "artifacts" / "models"
                   / "unet_dice_bce_seed42_best.pt",
    ),
    ExperimentDef(
        exp_id=6,
        name="UNet + BCE + aug + sampler",
        script=PROJECT_ROOT / "scripts" / "train_aug_sampler.py",
        checkpoint=PROJECT_ROOT / "artifacts" / "models"
                   / "unet_bce_pw136_aug_sampler_seed42_best.pt",
    ),
    ExperimentDef(
        exp_id=7,
        name="UNet + BCE + pos_weight + 500 extra samples",
        script=PROJECT_ROOT / "scripts" / "train_expanded.py",
        checkpoint=PROJECT_ROOT / "artifacts" / "models"
                   / "unet_bce_pw136_expanded500_seed42_best.pt",
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    """Create all required output directories under ``artifacts/``."""
    for sub in ("models", "tables", "plots", "predictions"):
        (ARTIFACTS_DIR / sub).mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)


def _check_data() -> bool:
    """Return True if the sample database is populated, False otherwise.

    Returns:
        Whether ``data/contrail_samples.db`` exists and is non-empty.
    """
    if not DB_PATH.exists():
        return False
    import sqlite3  # pylint: disable=import-outside-toplevel
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM contrail_samples"
        ).fetchone()[0]
    return count > 0


def _run_script(
    script: Path,
    exp_name: str,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Execute a training script as a subprocess.

    Args:
        script: Absolute path to the Python script to run.
        exp_name: Display name used in log messages.
        extra_env: Additional environment variables merged into the subprocess env.

    Returns:
        Exit code of the subprocess (0 = success).
    """
    print(f"\n{'='*70}")
    print(f"  Running: {exp_name}")
    print(f"  Script : {script.name}")
    print(f"{'='*70}")
    t0 = time.time()
    env = {**os.environ, "CONTRAIL_DB_PATH": str(DB_PATH)}
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    elapsed = (time.time() - t0) / 60
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  → {status}  ({elapsed:.1f} min)")
    return result.returncode


def _run_plots() -> None:
    """Generate the final comparison bar charts and save to ``artifacts/plots/``."""
    plot_script = PROJECT_ROOT / "scripts" / "plot_comparison.py"
    if plot_script.exists():
        subprocess.run([PYTHON, str(plot_script)], cwd=str(PROJECT_ROOT), check=False)


# ── Main ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with attributes ``exp``, ``force``,
        and ``smoke_test``.
    """
    parser = argparse.ArgumentParser(
        description="Run contrail segmentation experiments (1–7).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", nargs="+", type=int, metavar="N",
        help="IDs of experiments to run (default: all 1–7).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run experiments even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="CI mode: Exp 1 + Exp 5 only, 2 epochs each.",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate all experiments end-to-end.

    Iterates over the experiment list, skips ones with existing checkpoints
    (unless ``--force``), runs the training script, and generates plots.
    In ``--smoke-test`` mode only Exp 1 and Exp 5 are run with 2 epochs each.
    """
    args = parse_args()
    smoke: bool = args.smoke_test

    if smoke:
        selected_ids: set[int] = {5}
        force = True
        extra_env: dict[str, str] = {"SMOKE_TEST_EPOCHS": "2"}
    else:
        selected_ids = set(args.exp) if args.exp else {e.exp_id for e in EXPERIMENTS}
        force = args.force
        extra_env = {}

    _ensure_dirs()

    print(
        "\nNOTE: This demo runs on a 50-sample subset (data/sample/).\n"
        "Full 621-sample reproduction: replace data/sample/ with full dataset "
        "and run python main.py --full\n"
    )
    if smoke:
        print("[SMOKE TEST] Exp 5 only (UNet+BCE+pw=136) — 2 epochs.\n")

    if not _check_data():
        print(
            "ERROR: Sample database not found or empty.\n"
            "  Expected: data/sample/contrail_samples.db\n"
            "  Run: python scripts/build_sample_subset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    results: list[tuple[str, str]] = []
    for exp in EXPERIMENTS:
        if exp.exp_id not in selected_ids:
            if smoke:
                print(f"\n[Exp {exp.exp_id}] skipped in smoke test mode")
                results.append((exp.name, "smoke skip"))
            continue

        if exp.checkpoint.exists() and not force:
            print(f"\n[Exp {exp.exp_id}] SKIPPED — checkpoint exists: {exp.checkpoint.name}")
            results.append((exp.name, "skipped"))
            continue

        if not exp.script.exists():
            print(
                f"\n[Exp {exp.exp_id}] SKIPPED — script not found: {exp.script.name}",
                file=sys.stderr,
            )
            results.append((exp.name, "script missing"))
            continue

        rc = _run_script(exp.script, exp.name, extra_env=extra_env or None)
        results.append((exp.name, "ok" if rc == 0 else f"failed rc={rc}"))

    _run_plots()

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, status in results:
        print(f"  [{status.upper():^12}]  {name}")
    print()


if __name__ == "__main__":
    main()
