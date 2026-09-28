from pathlib import Path

# Resolved from this file, so paths work regardless of the working directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BINS_DIR = PROCESSED_DIR / "bins"
