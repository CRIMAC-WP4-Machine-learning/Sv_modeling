"""
Coordinate system: +X alongship, +Y athwartship, +Z downward (beam axis).

Target position angles (EK80-style direction-cosine convention):
  R                 : Slant range from transducer to target (m).
  angle_alongship   : arcsin(x/R), positive toward +X (degrees).
  angle_athwartship : arcsin(y/R), positive toward +Y (degrees).
  off_axis_angle    : Angle between target direction and +Z (degrees).

  x = R * np.sin(theta_al)
  y = R * np.cos(theta_al) * np.sin(theta_at)
  z = R * np.cos(theta_al) * np.cos(theta_at)
  
Fish orientation angles (spherical coordinates):
  theta : Azimuth in XY plane, measured from +X toward +Y (radians).
  phi   : Polar angle from +Z; phi=90° means horizontal (radians).

Fish incidence angle:
  Angle between fish longitudinal axis and acoustic direction.
  0° = end-on, 90° = broadside (for a symmetric fish model).

Run from anywhere, e.g. from the repository root:
    uv run python scripts/DistributeFish_IdealSpect_sv_beam.py

Figures are written to outputs/.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from sv_modeling.database import Extract_database_metadata
from sv_modeling.density import beam_shell_volume, estimate_density
from sv_modeling.paths import FEMBEM_DIR, TS_DATABASE_150HZ_DIR, output_path
from sv_modeling.plotting import plot_fish_school
from sv_modeling.school import orient_fish, place_fish_beam_shell
from sv_modeling.scattering import coherent_backscatter
from sv_modeling.signals import func_Gen_SweepSig_Explicit  # noqa: F401 (used in commented-out block)
from sv_modeling.spectrum import ESD_BKH  # noqa: F401 (used in commented-out block)

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
# Frequency interval (Hz)
# f1 = 10000.0
# f2 = 99000.0

# 1: Define an idealized flat spectrum ============================
# Frequency step for coherent simulation
# df = 1000.0

# freq_vec = np.arange(f1, f2 + df / 2, df)

# # Flat spectral weighting
# ESD_vec = np.ones_like(freq_vec)

# # Integral of the spectral weighting
# tips_0 = np.trapezoid(ESD_vec, freq_vec)


# 2: Realistic FM simulation =====================
# Transmit signal
# T_effectiveDuration=1*1.024*1E-3 # Sweep duration (sec)
# T_center=2*1.024*1E-3
# T_whole=4*1.024*1E-3     # Whole time signal with zeros (sec)

# Fs=512*512*16*2 # Sampling rate (Hz)
# F_start=f1  # Start frequency (Hz)
# F_end=f2    # End frequency (Hz)
# mexpo=25        # Fast ramping. For slow ramping select smaller value

# [t_vec,p_vec]=func_Gen_SweepSig_Explicit(T_whole,T_center,T_effectiveDuration,Fs,F_start,F_end,mexpo)
# dt = np.mean(np.diff(t_vec))
# Output = ESD_BKH(p_vec, dt)
# freq_vec = Output[:,0]
# ESD_vec = Output[:,1]
# df = 1 / (len(p_vec) * dt)



# FIG=plt.figure(figsize=(8,7)) 
# #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  
# ax = FIG.add_subplot(211)
# box = ax.get_position()
# #        ax.set_position([box.x0+0.0, box.y0+(0.01), box.width * 1 , box.height * 1.0])
# ax.set_position([box.x0, box.y0, box.width * 1.0 , box.height * 1.0])
# #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
# plt.plot(t_vec,p_vec)

# #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  
# ax = FIG.add_subplot(212)
# box = ax.get_position()
# #        ax.set_position([box.x0+0.0, box.y0+(0.01), box.width * 1 , box.height * 1.0])
# ax.set_position([box.x0, box.y0, box.width * 1.0 , box.height * 1.0])
# #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
# plt.plot(freq_vec, ESD_vec, color = [0,0,1])
# plt.show()

# #Energy of signal in time and frequency domains:
# E_time = np.sum(np.abs(p_vec)**2) * dt

# E_freq = np.sum(ESD_vec) * df

# print(f'E_time={E_time:.6e}')
# print(f'E_freq={E_freq:.6e}')
#====================================


#%% main part
Observation_point = np.array([0, 0, 0]) # Echosounder location

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Place N points (fish location) with no overlap within Acoustic beam 
# at given range with defined thickness layer 
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# Parameters
seed = None  # set an int for a reproducible school
rng = np.random.default_rng(seed)

N = 200
min_dist = 0.1  # m, minimum distance between fish

School_Mean_range = 50.0  # m
School_thikness = 1  # m

beam_half_angle_deg = 3.5
beam_half_angle = np.deg2rad(beam_half_angle_deg)

R = School_Mean_range
DR = School_thikness

# Generate targets uniformly in the circular conical shell
points = place_fish_beam_shell(N, R, DR, beam_half_angle_deg, min_dist, rng)

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ---------------------------------------------------------
# EK80-style split-beam angular coordinates
# ---------------------------------------------------------

# slant_range = np.linalg.norm(points, axis=1)
# angle_alongship = np.rad2deg(np.arcsin(np.clip(points[:, 0] / slant_range, -1, 1)))
# angle_athwartship = np.rad2deg(np.arcsin(np.clip(points[:, 1] / slant_range, -1, 1)))

# # True off-axis angle from the acoustic axis
# off_axis_angle = np.rad2deg(np.arccos(np.clip(points[:, 2] / slant_range, -1, 1)))

# # Verification
# print("Number of targets:", len(points))
# print("Minimum range:", slant_range.min())
# print("Maximum range:", slant_range.max())

# print("Alongship angle limits:", angle_alongship.min(), angle_alongship.max())
# print("Athwartship angle limits:", angle_athwartship.min(), angle_athwartship.max())
# print("Maximum off-axis angle:", off_axis_angle.max())


# #%% Plot simulated target positions

# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]

# slant_range = np.linalg.norm(points, axis=1)

# # EK80-style angular coordinates (degrees)
# angle_alongship = np.rad2deg(np.arcsin(np.clip(x / slant_range, -1, 1)))
# angle_athwartship = np.rad2deg(np.arcsin(np.clip(y / slant_range, -1, 1)))

# fig = plt.figure(figsize=(16, 5))
# # ---------------------------------------------------------
# # 1. 3D target distribution
# # ---------------------------------------------------------
# ax1 = fig.add_subplot(131, projection="3d")

# sc = ax1.scatter(x, y, z, c=z, cmap="viridis", s=15, alpha=0.8)

# ax1.set_xlabel("Alongship X (m)")
# ax1.set_ylabel("Athwartship Y (m)")
# ax1.set_zlabel("Depth Z (m)")
# ax1.set_title(f"3D target distribution (N={len(points)})")

# # Equal spatial scaling
# max_span = max(np.ptp(x), np.ptp(y), np.ptp(z))
# half_span = max_span / 2

# mid_x = (x.max() + x.min()) / 2
# mid_y = (y.max() + y.min()) / 2
# mid_z = (z.max() + z.min()) / 2

# ax1.set_xlim(mid_x - half_span, mid_x + half_span)
# ax1.set_ylim(mid_y - half_span, mid_y + half_span)
# ax1.set_zlim(mid_z + half_span, mid_z - half_span)

# ax1.set_box_aspect((1, 1, 1))

# fig.colorbar(sc, ax=ax1, shrink=0.65, label="Depth (m)")

# # ---------------------------------------------------------
# # 2. XY projection (looking down the beam)
# # ---------------------------------------------------------
# ax2 = fig.add_subplot(132)

# ax2.scatter(x, y, s=15, alpha=0.8)

# # Beam boundary at mean slant range
# beam_radius = R * np.sin(beam_half_angle)
# t = np.linspace(0, 2*np.pi, 360)

# ax2.plot(beam_radius * np.cos(t), beam_radius * np.sin(t), "r--", linewidth=1.5, label="Beam boundary")

# ax2.set_xlabel("Alongship X (m)")
# ax2.set_ylabel("Athwartship Y (m)")
# ax2.set_title("XY projection")
# ax2.set_aspect("equal")
# ax2.grid(True)
# ax2.legend()

# # ---------------------------------------------------------
# # 3. EK80 alongship / athwartship angle plot
# # ---------------------------------------------------------
# ax3 = fig.add_subplot(133)

# ax3.scatter(angle_alongship,  angle_athwartship, s=15, alpha=0.8)

# # Exact circular-cone boundary in split-beam coordinates
# t = np.linspace(0, 2*np.pi, 360)
# angle_boundary_al = np.rad2deg(np.arcsin(np.sin(beam_half_angle) * np.cos(t)))
# angle_boundary_at = np.rad2deg(np.arcsin(np.sin(beam_half_angle) * np.sin(t)))

# ax3.plot(angle_boundary_al, angle_boundary_at, "r--", linewidth=1.5, label="Beam boundary")

# ax3.set_xlabel("Alongship angle (deg)")
# ax3.set_ylabel("Athwartship angle (deg)")
# ax3.set_title("Target positions in beam")
# ax3.set_aspect("equal")
# ax3.grid(True)
# ax3.legend()

# plt.tight_layout()
# plt.show()
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


# II. Orientation of fish
# Ranges (deg). Theta: azimuth in XY plane from +X. Phi: angle from +Z.
theta_range_deg = (-0.001, 0.001)
phi_range_deg = (89.999, 90.001)

# "normal" or "uniform" within the ranges
orientations, theta, phi = orient_fish(N, theta_range_deg, phi_range_deg, rng, distribution="normal")


#%% Fish aggregate volume and number density
# ---------------------------------------------------------------
# Fish are randomly distributed inside a circular conical shell
# defined by the echosounder beam and a slant-range interval.
#
# R       : Mean slant range (m)
# DR      : Range thickness (m)
# alpha   : Beam half-opening angle (radians)
# r_min   : R - DR/2
# r_max   : R + DR/2
#
# The volume is obtained by integrating the spherical volume
# element dV = r² sin(theta) dr dtheta dphi over:
#   r     : [r_min, r_max]
#   theta : [0, alpha]
#   phi   : [0, 2*pi]
#
# Exact volume:
#   V = (2*pi/3) * (1 - cos(alpha)) * (r_max³ - r_min³)
#
# Narrow-beam, thin-shell approximation:
#   V ≈ pi * R² * alpha² * DR
#
# Fish number density:
#   rho_N = N / V   [fish/m³]
#
# Note: V is the beam-sampled volume, not necessarily the volume
# of the entire fish school if it extends beyond the beam.
# ---------------------------------------------------------------

# Exact beam-sampled volume (m³); approximate=True gives the narrow-beam form
V = beam_shell_volume(R, DR, beam_half_angle_deg)

# Fish number density (fish/m³)
rho_N = len(points) / V

print(f"Number of fish       : {len(points)}")
print(f"Aggregate volume     : {V:.3f} m³")
print(f"Fish number density  : {rho_N:.3f} fish/m³")


#%% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#                       Test case for few fish at defined locations:  
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
# points = []
# p = np.array([0, 0, - School_Mean_range])
# points.append(p)

# p = np.array([0, 0, - School_Mean_range -  0.1])
# points.append(p)

# p = np.array([0, 0, - School_Mean_range - 0.25])
# points.append(p)

# p = np.array([0, -0.25, - School_Mean_range - 0.25])
# points.append(p)

# points = np.array(points)
# # print(points)


# # Orientation: ---------------------------------------------
# PHI = np.array([90.0, 90.0, 90.0, 90.0]) * np.pi/180
# THETA = np.array([0.0, 0.0, 0.0, 0.0]) * np.pi/180

# # Convert spherical to Cartesian coordinates
# x = np.sin(PHI) * np.cos(THETA)
# y = np.sin(PHI) * np.sin(THETA)
# z = np.cos(PHI)

# orientations = np.stack((x, y, z), axis=1)  # shape (N, 3)
# orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]  # normalize

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Frequency interval (kHz, matching freq from database)
# Frequency interval (kHz)
# f1 = 1
# f2 = 100

N = len(points)

result = coherent_backscatter(points, orientations, df_database, database_dir, Observation_point)
freq_vec = result.freq_hz
p_far = result.p_far
sigma_bs_fish = result.sigma_bs  # sigma_bs(f) for every fish, shape (N, F)


#%% Number density estimation for selected frequency band (flat spectral weighting)

f1 = 38000.0
f2 = 70000.0

estimate = estimate_density(result, R, V, (f1, f2))
svf = estimate.svf  # coherent s_v(f) over all frequencies

print(f"Coherent density : {estimate.coherent:.4f} fish/m³")
print(f"Incoherent density: {estimate.incoherent:.4f} fish/m³")
print(f"True density     : {estimate.true:.4f} fish/m³")
print(f"Coherent/True    : {estimate.coherent / estimate.true:.4f}")


#%% ====================================================


# Plot the fish school of N fish
fig = plt.figure(figsize=(12, 5))


# 1. Theta histogram
ax1 = fig.add_subplot(231)
pos = ax1.get_position()
ax1.set_position([pos.x0-0.05, pos.y0 + 0.05, pos.width * 0.8, pos.height * 0.5])
ax1.hist(np.degrees(theta), bins=30, density=False, alpha=0.6, color="skyblue", edgecolor="black")
ax1.set_xlabel(r"$\theta$ (deg)", fontsize=12)
ax1.set_ylabel("Number of fish", fontsize=12)
ax1.set_title("Azimuth distribution")

# 2. Phi histogram
ax2 = fig.add_subplot(232)
pos = ax2.get_position()
ax2.set_position([pos.x0-0.05, pos.y0 + 0.05, pos.width * 0.8, pos.height * 0.5])
ax2.hist(np.degrees(phi), bins=30, density=False, alpha=0.6, color="skyblue", edgecolor="black")
ax2.set_xlabel(r"$\phi$ (deg)", fontsize=12)
ax2.set_ylabel("Number of fish", fontsize=12)
ax2.set_title("Polar angle distribution")

# 3. Fish school in 3D
ax3 = fig.add_subplot(233, projection="3d")
pos = ax3.get_position()
ax3.set_position([pos.x0-0.15, pos.y0-0.5, pos.width*2.6, pos.height*2.8])
plot_fish_school(points, orientations, ax3, length=0.3, margin=1.1, view=(12, -55), invert_z=True)
ax3.set_xlabel("X (m)", fontsize=12)
ax3.set_ylabel("Y (m)", fontsize=12)
ax3.set_zlabel("Z (m)", fontsize=12)

ax3.tick_params(axis="both", labelsize=10)
ax3.tick_params(axis="z", labelsize=10)

# 3. sv(f)
ax4 = fig.add_subplot(234)
pos = ax4.get_position()
ax4.set_position([pos.x0-0.04, pos.y0 - 0.00, pos.width * 2.1, pos.height * 0.8])
ax4.plot(freq_vec/1000,svf, color =[0,0,0])
ax4.set_xlabel("Frequency (kHz)", fontsize=12)
ax4.set_ylabel(r'$s_v\;(\mathrm{m}^2\,\mathrm{m}^{-3})$', fontsize=12)


ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# plt.tight_layout()

fig.suptitle(rf"Frequency interval: $[{f1/1e3:.1f},\,{f2/1e3:.1f}]$ kHz", fontsize=14)

plt.savefig(output_path("sv_beam_summary.png"), dpi=300, bbox_inches="tight")
plt.show()
