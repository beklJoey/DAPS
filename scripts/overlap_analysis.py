"""Overlap analysis: AMLSII train records vs current DB — no torch import needed."""
import sqlite3
import sys
from pathlib import Path

DB_PATH   = Path("data/contrail_samples.db")
AMLSII    = Path("D:/AMLSII_data/contrails_subset/train")
EXTRA_DIR = Path("data/extra/contrails")
BAND_FILES = ("band_11.npy", "band_13.npy", "band_14.npy", "band_15.npy", "band_16.npy")
MASK_FILE  = "human_pixel_masks.npy"

# ── 1. Read all sample_ids from DB ───────────────────────────────────────────
if not DB_PATH.exists():
    sys.exit(f"DB not found: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
db_ids = {row[0] for row in conn.execute("SELECT sample_id FROM contrail_samples")}
conn.close()
print(f"DB records : {len(db_ids)}")

# ── 2. List AMLSII train subdirectories ──────────────────────────────────────
if not AMLSII.exists():
    sys.exit(f"AMLSII dir not found: {AMLSII}")
amlsii_all  = sorted(d.name for d in AMLSII.iterdir() if d.is_dir())
print(f"AMLSII dirs: {len(amlsii_all)}")

# ── 3. Check which AMLSII records have all required files ────────────────────
amlsii_complete = []
for name in amlsii_all:
    d = AMLSII / name
    ok = all((d / f).exists() for f in BAND_FILES) and (d / MASK_FILE).exists()
    if ok:
        amlsii_complete.append(name)
print(f"AMLSII complete (all band+mask files): {len(amlsii_complete)}")

# ── 4. Compute overlap ───────────────────────────────────────────────────────
amlsii_set    = set(amlsii_complete)
in_db         = amlsii_set & db_ids
not_in_db     = amlsii_set - db_ids
print(f"AMLSII records already in DB : {len(in_db)}")
print(f"AMLSII records NOT in DB     : {len(not_in_db)}  <-- can add directly")

# ── 5. Extra/contrails overlap ───────────────────────────────────────────────
if EXTRA_DIR.exists():
    extra_ids = {p.stem for p in EXTRA_DIR.glob("*.npy")}
    print(f"\nExtra npy files : {len(extra_ids)}")
    extra_have_amlsii = extra_ids & amlsii_set
    extra_have_full   = extra_ids & amlsii_set - db_ids   # not yet in DB but have AMLSII folder
    print(f"Extra with AMLSII full-band folder        : {len(extra_have_amlsii)}")
    print(f"Extra with AMLSII folder AND not in DB    : {len(extra_have_full)}")
    extra_no_amlsii = extra_ids - amlsii_set
    print(f"Extra with no AMLSII folder (4ch only)    : {len(extra_no_amlsii)}")
else:
    print(f"\nExtra dir not found: {EXTRA_DIR}")

# ── 6. Print first 10 addable records ────────────────────────────────────────
addable = sorted(not_in_db)
print(f"\nFirst 10 addable AMLSII records:")
for r in addable[:10]:
    print(f"  {r}")

# ── 7. Save addable list ──────────────────────────────────────────────────────
out = Path("data/extra/addable_amlsii_records.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(addable))
print(f"\nSaved {len(addable)} addable record IDs -> {out}")
