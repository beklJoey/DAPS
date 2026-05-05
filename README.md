# Contrail Segmentation — ELEC0135 Assignment

**Student number**: 25218855
**GitHub**: https://github.com/UCL-ELEC0135/new-assignment-beklJoey

---

## Setup

```bash
conda env create -f environment.yml
conda activate contrail-segmentation
```

### Data

Download the GOES-16 contrail dataset from NOAA/Google and place it at:

```
D:/AMLSII_data/contrails_subset/
```

Then build the sample database:

```bash
python -c "
from src.data.storage import ContrailSampleStore, build_samples_from_dir
from src.data.validation import validate_dataset
store = ContrailSampleStore('data/contrail_samples.db')
samples = build_samples_from_dir('D:/AMLSII_data/contrails_subset/train')
store.upsert_many(validate_dataset(samples))
print(f'Stored {store.count()} samples')
"
```

The extra 4-channel samples (for Experiment 7) should be placed under:

```
data/extra/contrails/*.npy    # shape (256, 256, 4) float16
data/extra/train_df.csv       # record_id column
```

---

## Usage

```bash
python main.py
```

All results, plots and checkpoints are saved to `artifacts/`.

To force re-run of already completed experiments:

```bash
python main.py --force
```

To run only specific experiments (e.g. 4 and 7):

```bash
python main.py --exp 4 7
```

---

## Experiments

Seven experiments comparing U-Net architectures and loss functions for contrail
segmentation under 136:1 class imbalance.

| Exp | Model | Loss | pos_weight | Train samples |
|-----|-------|------|------------|---------------|
| 1 | U-Net | BCE | — | 96 |
| 2 | U-Net | BCE-Dice | — | 96 |
| 3 | TemporalUNet T=3 | BCE | — | 96 |
| 4 | U-Net | Dice+BCE 0.5/0.5 | — | 96 |
| 5 | U-Net | Weighted BCE | 136 | 96 |
| 6 | U-Net | Weighted BCE+aug | 136 | 96 |
| 7 | U-Net | Weighted BCE | 136 | 496 |

All experiments use seed=42, Adam (lr=3e-4), ReduceLROnPlateau, and early
stopping (patience=8) on validation `positive_only_iou`.

---

## Results

Best model: **Exp 7** — pos_iou=0.0803, F1=0.097, empty_fpr=0.767

| Model | Threshold | pos_iou | Precision | Recall | F1 | Empty FPR |
|-------|-----------|---------|-----------|--------|----|-----------|
| Exp 1: UNet+BCE | 0.1 | 0.0107 | 0.0033 | 1.000 | 0.0065 | 1.000 |
| Exp 2: UNet+BCE-Dice | 0.1 | 0.0107 | 0.0033 | 1.000 | 0.0065 | 1.000 |
| Exp 3: TemporalUNet T=3 | 0.3 | 0.0116 | 0.0036 | 0.991 | 0.0073 | 1.000 |
| Exp 4: UNet+BCE pw=136 | 0.6 | 0.0314 | 0.0208 | 0.493 | 0.0399 | 1.000 |
| Exp 5: UNet+Dice+BCE | 0.5 | 0.0092 | 0.0022 | 0.042 | 0.0042 | 1.000 |
| Exp 6: UNet+BCE+aug+samp | 0.7 | 0.0283 | 0.0142 | 0.150 | 0.0260 | 1.000 |
| **Exp 7: UNet+BCE+500 samples** | **0.8** | **0.0803** | **0.0538** | **0.504** | **0.0971** | **0.767** |

Expanding the dataset from 96 to 496 training samples raised pos_iou by +157%
and reduced Empty FPR from 1.000 to 0.767.

---

## Project structure

```
contrail-segmentation/
├── main.py                       # single entry point (runs all 7 experiments)
├── requirements.txt
├── environment.yml
├── src/
│   ├── data/
│   │   ├── acquisition.py        # dataset path resolution
│   │   ├── storage.py            # SQLite sample store, ContrailSample
│   │   └── validation.py         # sample integrity checks
│   ├── evaluation/
│   │   ├── metrics.py            # IoU, precision, recall, F1, empty-FPR
│   │   ├── robustness.py         # threshold robustness analysis
│   │   └── threshold.py          # threshold scan utilities
│   ├── models/
│   │   ├── unet.py               # U-Net (configurable channels)
│   │   └── temporal_unet.py      # TemporalUNet (multi-frame)
│   ├── training/
│   │   ├── losses.py             # BCE, Dice, CombinedLoss
│   │   └── trainer.py            # dataset, data loaders, train loop
│   └── utils/
│       ├── logger.py             # structured logging
│       └── seed.py               # global seed management
├── train_exp1_unet_bce.py        # Experiment 1
├── train_exp2_unet_bce_dice.py   # Experiment 2
├── retrain_temporal_unet.py      # Experiment 3
├── train_exp4_unet_pw136.py      # Experiment 4
├── train_exp5_unet_dice_bce.py   # Experiment 5
├── train_aug_sampler.py          # Experiment 6
├── train_expanded.py             # Experiment 7
├── plot_comparison.py            # bar chart figures
├── plot_pred_comparison.py       # qualitative visualisation
├── data/                         # (gitignored) sample DB + extra npy
└── artifacts/                    # (gitignored) models, tables, plots
    ├── models/                   # saved best checkpoints (.pt)
    ├── tables/                   # CSV metrics and training histories
    └── plots/                    # generated figures (.png)
```

---

## Reproducibility

All random seeds are fixed via `src/utils/seed.py`:

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
```

---

## AI Usage Disclosure

This project used Claude (Anthropic) for code assistance and report writing
support, in accordance with UCL Category 2 GenAI usage policy.
