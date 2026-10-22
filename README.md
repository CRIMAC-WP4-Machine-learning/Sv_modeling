
# Sv_modeling

Python-based simulation of acoustic backscattering from fish aggregations, including coherent and incoherent summation of individual fish scattering signals and estimation of the volume backscattering coefficient ($s_v$).

## Repository structure

```text
Sv_modeling/
├── src/                  # Source code
├── scripts/              # Simulation and analysis scripts
├── FEM_BEM/              # Numerical scattering model data
├── model_results/        # Simulation results
├── pyproject.toml        # Project configuration and dependencies
├── requirements.txt      # Python dependencies
├── uv.lock               # Locked dependency versions
└── README.md             # Project documentation
```

## Requirements

- Python 3.12 or newer
- Git
- [uv](https://docs.astral.sh/uv/) (recommended)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CRIMAC-WP4-Machine-learning/Sv_modeling.git
cd Sv_modeling
```

### 2. Create the virtual environment and install dependencies

Using `uv`:

```bash
uv sync
```

This creates a `.venv` directory and installs the dependencies specified in `pyproject.toml` and `uv.lock`.

### 3. Activate the virtual environment

On Linux or WSL Ubuntu:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the simulations

Simulation and analysis scripts are located in the `scripts/` directory.

Run a script from the repository root using:

```bash
uv run python scripts/<script_name>.py
```

Replace `<script_name>.py` with the name of the script you want to execute.

Alternatively, if the virtual environment is already activated:

```bash
python scripts/<script_name>.py
```

**Note:** Run scripts from the repository root so relative paths to input data and output directories are resolved consistently, where applicable.

## Project directories

- **`src/`** — Python source code used by the simulations.
- **`scripts/`** — Scripts for running simulations and analyzing results.
- **`FEM_BEM/`** — Numerical scattering-model data and related resources.
- **`model_results/`** — Directory for simulation outputs and results.

## Project information

Developed as part of research activities associated with CRIMAC and the Institute of Marine Research (IMR), Norway.

Repository: https://github.com/CRIMAC-WP4-Machine-learning/Sv_modeling
