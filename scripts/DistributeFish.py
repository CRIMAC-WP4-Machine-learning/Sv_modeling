"""
Coherent superposition of backscatter from N fish, optionally compared with
the FEM/BEM (COMSOL) solution for the same geometry.

The school comes from the config (school.geometry):
  spheroid : random school in a spheroid      (configs/DistributeFish.yaml)
  fixed    : explicit target positions, e.g. the 4-target COMSOL case
             (configs/DistributeFish_compareCOMSOL.yaml)
If the config has a ``fembem`` section, the FEM/BEM TS(f) is plotted with the
superposition result.

Coordinates: z is up, so targets below the echosounder have negative z.

Run from anywhere, e.g. from the repository root:
    uv run python scripts/DistributeFish.py
    uv run python scripts/DistributeFish.py --config configs/DistributeFish_compareCOMSOL.yaml

Parameters are read from configs/DistributeFish.yaml unless --config is given.
Figures are written to outputs/ and named after the config file.

Replaces DistributeFish_sv.py, DistributeFish_sv_orientation.py and
DistributeFish_Pbs_compareCOMSOL.py.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from sv_modeling.config import FixedSchool, SpheroidSchool, config_from_cli
from sv_modeling.database import Extract_database_metadata
from sv_modeling.fembem import func_FEM_BEM
from sv_modeling.paths import FEMBEM_DIR, TS_DATABASE_150HZ_DIR, output_path
from sv_modeling.plotting import plot_school_and_source
from sv_modeling.school import angles_to_unit_vectors, orient_fish, place_fish_spheroid
from sv_modeling.scattering import coherent_backscatter, total_ts
from sv_modeling.signals import func_Gen_SweepSig_Explicit
from sv_modeling.spectrum import ESD_BKH

#%% Configuration (configs/DistributeFish.yaml or --config)
cfg = config_from_cli("DistributeFish.yaml", description=__doc__)
cfg.require("transmit_signal", "echosounder.beam_direction", "echosounder.beam_half_angle_deg")
school = cfg.school_as(SpheroidSchool, FixedSchool)
if isinstance(school, SpheroidSchool):
    cfg.require("orientation")

run_name = cfg.path.stem  # used in the output file names

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

FIG = plt.figure(figsize=(8,7))
ax = FIG.add_subplot(211)
ax.plot(t_vec, p_vec)
ax.set_xlabel('Time (s)', fontsize=12)

ax = FIG.add_subplot(212)
ax.plot(freq_vec, ESD_vec, color=[0,0,1])
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('ESD', fontsize=12)
plt.savefig(output_path(f"TransmitSignal_{run_name}.png"), dpi=300, bbox_inches='tight')
plt.show()

# Energy of signal in time and frequency domains:
E_time = np.sum(np.abs(p_vec)**2) * dt
E_freq = np.sum(ESD_vec) * df

print(f'E_time={E_time:.6e}')
print(f'E_freq={E_freq:.6e}')

#%% ##====================================================
# FEM/BEM (COMSOL) reference, if given in the config
if cfg.fembem is not None:
    [f_bem, TS_bem] = func_FEM_BEM(FEM_BEM_dir, cfg.fembem.file)

#%% main part
Observation_point = np.array(cfg.echosounder.position_m)  # Echosounder location
Source_direction = np.array(cfg.echosounder.beam_direction)  # Echosounder pointing direction (beam axis), plotting only
Beam_half_angle_deg = cfg.echosounder.beam_half_angle_deg  # Echosounder beam half-angle (deg), plotting only

if isinstance(school, SpheroidSchool):
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Place N points (fish location) with no overlap in a prolate or oblate spheroid
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
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
    arrow_length = 0.2
else:
    # Fixed targets. They must match the geometry of the FEM-BEM file when comparing.
    points = np.array(school.points_m, dtype=float)
    theta = np.radians(school.theta_deg)
    phi = np.radians(school.phi_deg)
    orientations = angles_to_unit_vectors(theta, phi)
    arrow_length = 0.1

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
plt.tight_layout()
plt.savefig(output_path(f"Orientation_{run_name}.png"), dpi=300, bbox_inches='tight')

# Plot the fish school with the echosounder beam
plot_school_and_source(points, orientations, Observation_point, Source_direction,
                       beam_half_angle_deg=Beam_half_angle_deg, length=arrow_length)
plt.savefig(output_path(f"Plot_{run_name}.png"), dpi=300, bbox_inches='tight')

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Coherent superposition (per-target output only for a few fixed targets)
is_fixed = isinstance(school, FixedSchool)
result = coherent_backscatter(
    points, orientations, df_database, database_dir, Observation_point, verbose=is_fixed
)
if is_fixed:
    print('Incident angles (deg): ', result.incident_angles)
    print('Distances (m): ', result.distances)

freq = result.freq_hz / 1000  # kHz, for plotting
p_far = result.p_far
Total_p_TS = total_ts(p_far, result.distances[0])

plt.figure()
plt.plot(freq, Total_p_TS, color = [0, 0, 0], dashes = [3,0], linewidth = 1, label = 'Superposition')
if cfg.fembem is not None:
    plt.plot(f_bem/1000, TS_bem, color=[1,0,0], dashes = [2,2], label = 'FEM-BEM')

window_L = int(len(Total_p_TS) / 5)
# Ensure window_length is odd
if window_L % 2 == 0:
    window_L += 1
smoothed = savgol_filter(Total_p_TS, window_length=window_L, polyorder=3)

plt.xlabel(' Frequency (kHz)', fontsize = 12)
plt.ylabel(' $10log_{10}(p_{tot}^2)$ (dB)', fontsize = 12)
plt.legend(loc='upper right', fontsize=12, facecolor='white')

plt.savefig(output_path(f"Multitarget_{run_name}.png"), dpi=300, bbox_inches='tight')  # PNG
plt.savefig(output_path(f"Multitarget_{run_name}.jpg"), dpi=300, bbox_inches='tight')  # JPG

plt.show()
