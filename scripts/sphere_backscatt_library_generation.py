"""
Generate complex backscattering amplitude f_bs for spherical gas bubbles
over specified depth, radius and frequency ranges.

Check if the required ".csv" exist in data/ts_database/spherical.
IF NOT, run sphere_backscatt_library_generation.py first for Gas content (e.g. O2, CO2, C2H6 etc), 
at required depth and size distribution

Results are saved to data/ts_database/spherical/.

Run from anywhere, e.g. from the repository root:
    uv run python scripts/sphere_backscatt_library_generation.py
    uv run python scripts/sphere_backscatt_library_generation.py --config path/to/other.yaml

Parameters are read from configs/sphere_backscatt_library_generation.yaml unless --config is given.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sv_modeling.f_bs_sphere import func_formfunc_FluidFiledSphere_Rinf
from sv_modeling.config import config_from_cli
from sv_modeling.equations_of_state import func_VanDerWaals
from sv_modeling.paths import TS_DATABASE_SPHERE_DIR
from sv_modeling.gas_sound_speed import func_c_gas


#%% Inputs (configs/sphere_backscatt_library_generation.yaml or --config)
cfg = config_from_cli("sphere_backscatt_library_generation.yaml", description=__doc__, copy_to_outputs=False)
cfg.require("bubbles", "bubble_library")
bub, lib = cfg.bubbles, cfg.bubble_library

Bubble_Depth_start = lib.depth_start_m  # m
Bubble_Depth_end = lib.depth_end_m  # m
Depth_resolution = lib.depth_resolution_m  # m

freq_start = lib.freq_start_hz  # Hz
freq_end = lib.freq_end_hz  # Hz
freq_resolution = lib.freq_resolution_hz  # Hz

R_bubble_min = bub.radius_min_m  # m
R_bubble_max = bub.radius_max_m  # m
R_bubble_resolution = bub.radius_resolution_m  # m

Gas = bub.gas
P0 = lib.surface_pressure_pa
ro_w = lib.water_density_kg_m3  # kg/m^3
c_w = bub.sound_speed_m_s  # m/s
Temperature_C = lib.temperature_c  # Celsius


#%% Initial calculations
D_vec = np.arange(Bubble_Depth_start, Bubble_Depth_end + 0.1 * Depth_resolution, Depth_resolution)
freq_vec = np.arange(freq_start, freq_end + 0.1 * freq_resolution, freq_resolution)
R_bubble_vec = np.arange(R_bubble_min, R_bubble_max + 0.1 * R_bubble_resolution, R_bubble_resolution)

c_gas = func_c_gas(Temperature_C, Gas)

results = []

# test: ============================
# D = 50
# Pressure = P0 + ro_w * D * 9.81
# ro_gas = func_VanDerWaals(Pressure, Temperature_C, Gas)
# R = 1e-3# 
# f_bs = func_formfunc_FluidFiledSphere_Rinf(
#             R, ro_w, ro_gas, c_w, c_gas, freq_vec,
#             R_inf=None, use_exact_farfield=False
#         )

# plt.plot(freq_vec, 20*np.log10(np.abs(f_bs)))
# plt.show()
#============================




#%% Calculate backscattering
for D in D_vec:

    Pressure = P0 + ro_w * D * 9.81
    ro_gas = func_VanDerWaals(Pressure, Temperature_C, Gas)

    # print('C, ro ', c_gas, ro_gas)
    for R in R_bubble_vec:

        f_bs = func_formfunc_FluidFiledSphere_Rinf(
            R, ro_w, ro_gas, c_w, c_gas, freq_vec,
            R_inf=None, use_exact_farfield=False
        )

        df_temp = pd.DataFrame({
            "depth_m": D,
            "frequency_Hz": freq_vec,
            "radius_m": R,
            "f_bs": f_bs
        })

        results.append(df_temp)


#%% Combine results
df = pd.concat(results, ignore_index=True)


#%% Save CSV
TS_DATABASE_SPHERE_DIR.mkdir(parents=True, exist_ok=True)

filename = (f"{Gas}_depth{Bubble_Depth_start:g}-{Bubble_Depth_end:g}m_R"
            f"{R_bubble_min*1e3:g}-{R_bubble_max*1e3:g}mm.csv")
filepath = TS_DATABASE_SPHERE_DIR / filename

df.to_csv(filepath, index=False)

print(f"Saved: {filepath}")
print(f"Number of rows: {len(df):,}")
print(df.head())


