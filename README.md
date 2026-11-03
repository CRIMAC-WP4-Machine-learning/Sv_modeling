# Sv_modeling

Python-based simulation of acoustic backscattering from fish aggregations, including coherent and incoherent summation of individual fish scattering signals and estimation of the volume backscattering coefficient ($s_v$).

## Repository structure

```text
Sv_modeling/
├── src/sv_modeling/          # Installable package (shared code)
│   ├── database.py           # TS(f)/f_bs(f) database: metadata, closest-file lookup, loading
│   ├── fembem.py             # Reader for FEM/BEM (COMSOL) TS exports
│   ├── geometry.py           # Incident angle between fish axis and echosounder
│   ├── paths.py              # Repository paths (data/, outputs/)
│   ├── plotting.py           # 3-D fish-school plot
│   ├── signals.py            # Transmit sweep signal
│   └── spectrum.py           # Energy spectral density
├── scripts/                  # Simulation and analysis entry points
│   ├── DistributeFish_sv.py                   # Random school, coherent superposition
│   ├── DistributeFish_Pbs_compareCOMSOL.py    # Fixed targets vs FEM/BEM solution
│   └── DistributeFish_IdealSpect_sv_beam.py   # Fish in a beam: coherent/incoherent s_v and density
├── data/                     # Input data - NOT in git, see Data
│   ├── fembem/               # FEM/BEM (COMSOL) TS results and parameter files
│   └── ts_database/          # Modelled TS(f) and f_bs(f) (CSV)
├── docs/figures/             # Reference figures
├── tests/                    # pytest tests
├── outputs/                  # Generated figures (git-ignored, created on first run)
├── pyproject.toml            # Project configuration and dependencies
└── uv.lock                   # Locked dependency versions
```

## Requirements

- Python 3.12 or newer
- Git
- [uv](https://docs.astral.sh/uv/) (recommended)

## Installation

```bash
git clone https://github.com/CRIMAC-WP4-Machine-learning/Sv_modeling.git
cd Sv_modeling
uv sync
```

`uv sync` creates `.venv`, installs the dependencies from `pyproject.toml`/`uv.lock`, and installs the `sv_modeling` package in editable mode, so scripts can `import sv_modeling` without any `sys.path` changes.

To activate the environment manually:

```bash
source .venv/bin/activate        # Linux / WSL
.venv\Scripts\Activate.ps1       # Windows PowerShell
```

If you need a `requirements.txt` (e.g. for pip), generate it from the lock file:

```bash
uv export --no-hashes --no-dev > requirements.txt
```

## Data

Input data is not stored in git (`data/` is git-ignored). Before running the scripts, copy it into:

```text
data/
├── fembem/                  # FEM/BEM (COMSOL) TS exports (*.txt + *_Param.txt), ~1.6 MB
│                            #   used by DistributeFish_sv.py and DistributeFish_Pbs_compareCOMSOL.py
└── ts_database/
    ├── 150Hz/               # 18 CSV files, ~0.9 MB - used by all scripts
    └── 1-260Khz/            # optional, not used by the scripts yet
```

Location of the source files: _TODO: add link to data storage (e.g. IMR share / Zenodo)_.

Each TS-database CSV has columns `Freq_kHz`, `TS` (dB) and complex `f_bs`. Model parameters (`a`, `b`, `rhos`, `IncAngle`, `depth`, `length`) are encoded in the filename and parsed by `sv_modeling.database.Extract_database_metadata`.

## Running the simulations

```bash
uv run python scripts/<script_name>.py
```

Data paths are resolved relative to the repository (see `src/sv_modeling/paths.py`), so scripts can be run from any working directory. Figures are written to `outputs/`.

## Using the package

```python
from sv_modeling.database import Extract_database_metadata, Find_closestfile_in_database
from sv_modeling.paths import TS_DATABASE_150HZ_DIR

df = Extract_database_metadata(TS_DATABASE_150HZ_DIR)
row = Find_closestfile_in_database(df, _Depth=50, _Length=0.0, _IncAngl=88)
```

## Tests

```bash
uv run pytest
```

## Project information

Developed as part of research activities associated with CRIMAC and the Institute of Marine Research (IMR), Norway.

Repository: https://github.com/CRIMAC-WP4-Machine-learning/Sv_modeling
