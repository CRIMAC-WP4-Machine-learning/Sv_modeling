"""
Coherent superposition of backscatter from a random fish school (N fish in a
spheroid at ~50 m depth). Optionally compare with FEM/BEM results.

Run from anywhere, e.g. from the repository root:
    uv run python scripts/DistributeFish_sv_orientation.py
    uv run python scripts/DistributeFish_sv_orientation.py --config path/to/other.yaml

Parameters are read from configs/DistributeFish_sv_orientation.yaml unless --config is given.
Figures are written to outputs/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from sv_modeling.config import SpheroidSchool, config_from_cli
from sv_modeling.database import Extract_database_metadata
from sv_modeling.fembem import func_FEM_BEM
from sv_modeling.paths import FEMBEM_DIR, TS_DATABASE_150HZ_DIR, output_path
from sv_modeling.plotting import plot_school_and_source
from sv_modeling.school import orient_fish, place_fish_spheroid
from sv_modeling.scattering import coherent_backscatter, total_ts
from sv_modeling.signals import func_Gen_SweepSig_Explicit
from sv_modeling.spectrum import ESD_BKH

#%% Configuration (configs/DistributeFish_sv_orientation.yaml or --config)
cfg = config_from_cli("DistributeFish_sv_orientation.yaml", description=__doc__)
cfg.require("transmit_signal", "echosounder.beam_direction", "echosounder.beam_half_angle_deg", "orientation", "fembem")
school = cfg.school_as(SpheroidSchool)

#%% Directories and databases
# Database directory containing csv files with TS(f) and f_bs(f):
database_dir = TS_DATABASE_150HZ_DIR
print('database_dir: ', database_dir)

FEM_BEM_dir = FEMBEM_DIR
print('FEM_BEM_dir: ', FEM_BEM_dir)

# df_database has filename, target (fish) length, incident angle, depth and
# prolate spheroid dimensions of the swimbladder "a, b"
df_database = Extract_database_metadata(database_dir)
#%%====================================================
# Transmit signal
tx = cfg.transmit_signal
T_effectiveDuration = tx.effective_duration_ms * 1e-3  # Sweep duration (sec)
T_center = tx.center_time_ms * 1e-3
T_whole = tx.total_duration_ms * 1e-3  # Whole time signal with zeros (sec)

Fs = tx.sampling_rate_hz  # Sampling rate (Hz)
F_start = tx.f_start_hz  # Start frequency (Hz)
F_end = tx.f_end_hz  # End frequency (Hz)
mexpo = tx.ramp_exponent  # Fast ramping. For slow ramping select smaller value

[t_vec,p_vec]=func_Gen_SweepSig_Explicit(T_whole,T_center,T_effectiveDuration,Fs,F_start,F_end,mexpo)
dt = np.mean(np.diff(t_vec))
Output = ESD_BKH(p_vec, dt)
freq_vec = Output[:,0]
ESD_vec = Output[:,1]
df = 1 / (len(p_vec) * dt)

#Energy of signal in time and frequency domains:
E_time = np.sum(np.abs(p_vec)**2) * dt

E_freq = np.sum(ESD_vec) * df

print(f'E_time={E_time:.6e}')
print(f'E_freq={E_freq:.6e}')
#%% ##====================================================


FEM_BEM_files = [cfg.fembem.file]

[f_bem, TS_bem] = func_FEM_BEM(FEM_BEM_dir, FEM_BEM_files[0])

#%% main part
Observation_point = np.array(cfg.echosounder.position_m)  # Echosounder location
Source_direction = np.array(cfg.echosounder.beam_direction)  # Echosounder pointing direction (beam axis), plotting only
Beam_half_angle_deg = cfg.echosounder.beam_half_angle_deg  # Echosounder beam half-angle (deg), plotting only

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Example: place N points (fish location) with no overlap in a prolate or oblate spheroid
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
rng = np.random.default_rng(cfg.seed)  # seed: null in the config gives a new school each run

N = school.n_targets
a, b = school.semi_axis_horizontal_m, school.semi_axis_vertical_m  # spheroid semi-axes (m): horizontal, vertical
min_dist = school.min_dist_m  # m, minimum distance between fish
Average_school_Depth = school.depth_m  # m

# Orientation ranges (deg). Theta: azimuth in XY plane from +X. Phi: angle from +Z.
theta_range_deg = cfg.orientation.theta_range_deg
phi_range_deg = cfg.orientation.phi_range_deg

# I. Fish positions, no overlap
points = place_fish_spheroid(N, a, b, Average_school_Depth, min_dist, rng)

# II. Fish orientations ("normal" or "uniform" within the ranges)
orientations, theta, phi = orient_fish(N, theta_range_deg, phi_range_deg, rng, distribution=cfg.orientation.distribution)

# Orientation histograms
plt.figure()
plt.subplot(1,2,1)
plt.hist(np.degrees(theta), bins=30, density=False, alpha=0.6, color='skyblue', edgecolor='black')
plt.xlabel(r'$\theta$ (deg)', fontsize=12)
plt.ylabel(r'# objects', fontsize=12)

plt.subplot(1,2,2)
plt.hist(np.degrees(phi), bins=30, density=False, alpha=0.6, color='skyblue', edgecolor='black')
plt.xlabel(r'$\phi$ (deg)', fontsize=12)
plt.ylabel(r'# objects', fontsize=12)

# Plot the fish school of N fish
plot_school_and_source(points, orientations, Observation_point, Source_direction, beam_half_angle_deg=Beam_half_angle_deg)
plt.savefig(output_path("Plot_"+os.path.splitext(FEM_BEM_files[0])[0]+".png"), dpi=300, bbox_inches='tight')  # PNG

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

result = coherent_backscatter(
    points, orientations, df_database, database_dir, Observation_point
)
freq = result.freq_hz / 1000  # kHz, for plotting
p_far = result.p_far
Total_p_TS = total_ts(p_far, result.distances[0])
plt.figure()
plt.plot(freq, Total_p_TS, color = [0, 0, 0], dashes = [3,0], linewidth = 1, label = 'Superposition')

# plt.plot(f_bem/1000, TS_bem, color=[1,0,0], dashes = [2,2], label = 'FEM-BEM')

window_L = int(len(Total_p_TS) / 5)
# Ensure window_length is odd
if window_L % 2 == 0:
    window_L += 1
smoothed = savgol_filter(Total_p_TS, window_length=window_L, polyorder=3)
# plt.plot(freq, smoothed, color = [1, 0, 0], dashes = [3,0], linewidth = 3)

plt.xlabel(' Frequency (kHz)', fontsize = 12)
plt.legend(
    loc='upper right',     # location of legend
    fontsize=12,          # font size
    facecolor='white',    # legend background color
)

plt.savefig(output_path("Multitarget_"+os.path.splitext(FEM_BEM_files[0])[0]+".png"), dpi=300, bbox_inches='tight')  # PNG
plt.savefig(output_path("Multitarget_"+os.path.splitext(FEM_BEM_files[0])[0]+".jpg"), dpi=300, bbox_inches='tight')  # JPG

plt.show()

# delta_hz = np.mean(freq[1:].values - freq[:-1].values) * 1000
# time, ifft = inverse_frequency_response(Total_p_TS.values, delta_hz)
# plt.plot(time[1:] * 1500 * 0.5, ifft[1:], linewidth = 1, label = 'iFFT')
# plt.xlabel('Distance [m]', fontsize=12)
# plt.legend()
# plt.xlim([0.0, 1.0])
# plt.show()
