# Sv_modeling

## Overview

**Sv_modeling** is a Python-based research project for simulating acoustic backscattering from aggregations of fish and investigating the relationship between individual fish scattering properties and the volume backscattering coefficient, \(s_v\).

The model combines frequency-dependent complex backscattering amplitudes of individual fish with their spatial positions and orientations to calculate the acoustic response of a fish aggregation.

A particular focus is the comparison between **coherent** and **incoherent** summation of the scattered signals and the implications for estimating fish number density from acoustic measurements.

## Objectives

The main objectives are to:

- Simulate frequency-dependent acoustic backscattering from fish aggregations.
- Account for fish positions, orientations, and propagation phase.
- Investigate coherent interference between signals scattered by individual fish.
- Compare coherent and incoherent volume backscattering estimates.
- Evaluate the accuracy and variability of acoustic fish-density estimates.
- Explore the effects of frequency bandwidth, spatial distribution, and fish orientation.

## Theoretical background

### 1. Individual fish backscattering

The complex far-field backscattering amplitude of fish \(i\) is represented by

\[
f_{bs,i}(f)
\]

where \(f\) is acoustic frequency.

The corresponding differential backscattering cross section is

\[
\sigma_{bs,i}(f)=|f_{bs,i}(f)|^2.
\]

The target strength is defined as

\[
TS_i(f)=10\log_{10}\left(\frac{\sigma_{bs,i}(f)}{1\,\mathrm{m}^2}\right).
\]

### 2. Propagation and coherent summation

For a fish at range \(R_i\), the complex backscattered signal is modeled as

\[
p_i(f)=\frac{f_{bs,i}(f)}{R_i^2}
\exp\left(-j\frac{4\pi fR_i}{c}\right)
\]

where \(c\) is the sound speed in water.

The total coherently summed signal from an aggregation of \(N\) fish is

\[
p_{\mathrm{far}}(f)=\sum_{i=1}^{N}p_i(f).
\]

This formulation includes the relative propagation phases of signals scattered by different fish.

The squared magnitude of the total signal is

\[
|p_{\mathrm{far}}(f)|^2
=
\sum_{i=1}^{N}|p_i(f)|^2
+
2\sum_{i<j}\operatorname{Re}
\left[p_i(f)p_j^*(f)\right].
\]

The second term represents coherent interference between fish.

### 3. Volume backscattering coefficient

The linear volume backscattering coefficient is related to fish number density by

\[
s_v=n_b\overline{\sigma}_{bs}
\]

where:

- \(s_v\) is the volume backscattering coefficient, in \(\mathrm{m}^{-1}\).
- \(n_b\) is the number of fish per unit volume, in \(\mathrm{m}^{-3}\).
- \(\overline{\sigma}_{bs}\) is the mean differential backscattering cross section, in \(\mathrm{m}^2\).

The corresponding volume backscattering strength is

\[
S_v=10\log_{10}\left(\frac{s_v}{1\,\mathrm{m}^{-1}}\right).
\]

### 4. Broadband coherent backscattering

For the frequency interval \([f_1,f_2]\), the band-averaged coherent backscattering quantity is

\[
\mathrm{TIPS}_{coh}
=
\frac{R_0^2}{f_2-f_1}
\int_{f_1}^{f_2}|p_{\mathrm{far}}(f)|^2\,df
\]

where \(R_0\) is a reference range.

The estimated volume backscattering coefficient, using a representative aggregation range \(R\) and aggregation volume \(V\), is

\[
s_{v,coh}
=
\frac{R^4}{VR_0^2}\mathrm{TIPS}_{coh}.
\]

The corresponding fish number density is estimated as

\[
\hat n_b=\frac{s_{v,coh}}{\overline{\sigma}_{bs}}.
\]

### 5. Incoherent backscattering

In the incoherent approximation, the interference terms are neglected:

\[
\mathrm{TIPS}_{inc}
=
\frac{R_0^2}{f_2-f_1}
\sum_{i=1}^{N}
\int_{f_1}^{f_2}|p_i(f)|^2\,df.
\]

The incoherent volume backscattering estimate is

\[
s_{v,inc}
=
\frac{R^4}{VR_0^2}\mathrm{TIPS}_{inc}.
\]

Comparison of \(s_{v,coh}\) and \(s_{v,inc}\) provides a way to investigate the influence of coherent interference on acoustic abundance estimation.

The approximation using a common range \(R\) is most appropriate when the range variation across the aggregation is small relative to its distance from the transducer.

## Simulation approach

The general simulation workflow consists of:

1. Define the number of fish and the geometry of the aggregation.
2. Generate fish positions and orientations.
3. Obtain complex frequency-dependent backscattering amplitudes from scattering-model data.
4. Calculate the propagation phase and geometrical spreading for each fish.
5. Sum the individual signals coherently in the frequency domain.
6. Integrate the backscattered intensity over the selected frequency interval.
7. Calculate coherent and incoherent volume backscattering estimates.
8. Compare estimated and true fish number densities.

Random realizations of fish positions and orientations can be used to investigate variability in the results.

## Repository structure

```text
Sv_modeling/
├── src/                  # Source code
├── scripts/              # Research and simulation scripts
├── FEM_BEM/              # Numerical scattering-model resources
├── model_results/        # Model outputs and results
├── pyproject.toml        # Python project configuration
├── requirements.txt      # Python dependencies
├── uv.lock               # Locked dependency versions
└── README.md
```

## Installation

### Requirements

- Python 3.12 or newer
- Git
- Python package manager (`uv` recommended)

### Clone the repository

```bash
git clone https://github.com/CRIMAC-WP4-Machine-learning/Sv_modeling.git
cd Sv_modeling
```

### Create the Python environment using uv

If `uv` is installed, run:

```bash
uv sync
```

This creates or updates the project environment and installs dependencies using the project configuration and lock file.

To activate the environment on Linux or WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, dependencies can be installed using `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Important modeling assumptions

The simplified acoustic formulation assumes:

- Single scattering from each fish, neglecting multiple scattering between fish.
- Spherical spreading of the incident and backscattered waves.
- A homogeneous propagation medium with constant sound speed.
- Known complex backscattering amplitudes for individual fish.
- Coherent superposition of the individual scattered pressure fields.
- A frequency-independent weighting across the selected frequency interval.

The last assumption corresponds to an idealized flat spectral weighting. A realistic broadband echosounder simulation may require accounting for the transmitted pulse spectrum, transducer response, filtering, pulse compression, and beam directivity.

## Applications

The model can be used to investigate:

- Coherent interference in fish aggregations.
- Broadband versus narrowband acoustic backscattering.
- Sensitivity of volume backscattering estimates to fish positions and orientations.
- Variability and bias in acoustic abundance estimation.
- Relationships between individual target strength and aggregation-scale backscattering.
- The validity of incoherent scattering assumptions in fisheries acoustics.

## Project information

Developed within the research activities associated with **CRIMAC** and the **Institute of Marine Research (IMR), Norway**.

Repository: https://github.com/CRIMAC-WP4-Machine-learning/Sv_modeling

## Notes

This repository is intended for research and model development. Results should be interpreted in the context of the implemented physical assumptions and numerical approximations.
