"""Repository paths, resolved from this file so scripts work from any working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
TS_DATABASE_DIR = DATA_DIR / "ts_database"
TS_DATABASE_150HZ_DIR = TS_DATABASE_DIR / "150Hz"
FEMBEM_DIR = DATA_DIR / "fembem"

OUTPUT_DIR = REPO_ROOT / "outputs"


def output_path(filename: str) -> Path:
    """Return a path in OUTPUT_DIR, creating the directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename
