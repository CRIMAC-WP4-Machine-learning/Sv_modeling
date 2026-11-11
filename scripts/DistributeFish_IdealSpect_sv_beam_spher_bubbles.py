
"""
Simulate coherent broadband backscattering from spherical gas bubbles.

Bubble depths follow a normal distribution around Mean_depth.
Complex f_bs(f) spectra are read from spherical CSV databases.
"""

#%% Imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sv_modeling.paths import TS_DATABASE_SPHERE_DIR, output_path
from sv_modeling.plotting import plot_bubble_cloud


#%% 1. Simulation parameters
Gas = "CO2"

N = 10000
seed = 42
rng = np.random.default_rng(seed)

Mean_depth = 55.0
Depth_std = 1.0

beam_half_angle_deg = 3.5
beam_half_angle = np.deg2rad(beam_half_angle_deg)

R_bubble_min = 0.0005
R_bubble_max = 0.0020

c_w = 1500.0
Observation_point = np.array([0.0, 0.0, 0.0])

# Database coverage: approximately 99.7% of a normal distribution
required_min = Mean_depth - 3 * Depth_std
required_max = Mean_depth + 3 * Depth_std


#%% 2. Find CSV databases covering the requested depth interval
database_dir = Path(TS_DATABASE_SPHERE_DIR)

if not database_dir.is_dir():
    raise FileNotFoundError(f"Database directory does not exist: {database_dir}")

available_files = []

for filepath in database_dir.glob("*.csv"):
    match = re.fullmatch(rf"{re.escape(Gas)}_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)m", filepath.stem)
    if match:
        d_min, d_max = map(float, match.groups())
        available_files.append((d_min, d_max, filepath))

available_files.sort(key=lambda item: item[0])

if not available_files:
    raise FileNotFoundError(f"No {Gas} CSV databases found in {database_dir}")

# Select the first database covering the lower bound, then add adjacent files if needed
selected_files = []
covered_until = required_min
tolerance = 1e-6

while covered_until < required_max - tolerance:
    candidates = [item for item in available_files if item[0] <= covered_until + tolerance and item[1] > covered_until]
    if not candidates:
        raise ValueError(f"No continuous {Gas} database coverage for {required_min:g}–{required_max:g} m")

    chosen = max(candidates, key=lambda item: item[1])
    if chosen[2] not in selected_files:
        selected_files.append(chosen[2])
    covered_until = chosen[1]

print("Selected databases:")
for filepath in selected_files:
    print(" ", filepath.name)


#%% 3. Load and combine databases
frames = []

for filepath in selected_files:
    df_part = pd.read_csv(filepath)
    frames.append(df_part)

df = pd.concat(frames, ignore_index=True)

required_columns = {"depth_m", "frequency_Hz", "radius_m", "f_bs"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing database columns: {required_columns - set(df.columns)}")

df["f_bs"] = df["f_bs"].astype(complex)

# Remove any duplicate depth-radius-frequency combinations at file boundaries
df = df.drop_duplicates(subset=["depth_m", "radius_m", "frequency_Hz"], keep="first")

depth_db = np.sort(df["depth_m"].unique())
radius_db = np.sort(df["radius_m"].unique())
freq_vec = np.sort(df["frequency_Hz"].unique())

expected_rows = len(depth_db) * len(radius_db) * len(freq_vec)

if len(df) != expected_rows:
    raise ValueError("Combined database is not a complete depth-radius-frequency grid")

df = df.sort_values(["depth_m", "radius_m", "frequency_Hz"])
f_bs_db = df["f_bs"].to_numpy().reshape(len(depth_db), len(radius_db), len(freq_vec))

print(f"Database depth range: {depth_db.min():.2f}–{depth_db.max():.2f} m")
print(f"Database radii: {len(radius_db)}")
print(f"Frequency range: {freq_vec.min()/1e3:.1f}–{freq_vec.max()/1e3:.1f} kHz")
print(f"Database shape: {f_bs_db.shape}")

if depth_db.min() > required_min or depth_db.max() < required_max:
    raise ValueError("Actual CSV depth values do not cover the requested ±3σ interval")


#%% 4. Generate normally distributed bubble depths
# Rejection sampling avoids clipping bubbles onto the database boundaries
depths = np.empty(N)
remaining = np.arange(N)

while remaining.size:
    samples = rng.normal(Mean_depth, Depth_std, remaining.size)
    valid = (samples >= depth_db.min()) & (samples <= depth_db.max())
    depths[remaining[valid]] = samples[valid]
    remaining = remaining[~valid]


#%% 5. Generate 3D bubble positions within the acoustic beam
azimuth = rng.uniform(0, 2*np.pi, N)
cos_alpha = rng.uniform(np.cos(beam_half_angle), 1.0, N)
alpha = np.arccos(cos_alpha)

x = depths * np.tan(alpha) * np.cos(azimuth)
y = depths * np.tan(alpha) * np.sin(azimuth)
z = depths

points = np.column_stack((x, y, z))
ranges = np.linalg.norm(points - Observation_point, axis=1)


#%% 6. Assign bubble radii
available_radii = radius_db[(radius_db >= R_bubble_min) & (radius_db <= R_bubble_max)]

if available_radii.size == 0:
    raise ValueError("No database radii within the requested interval")

bubble_radii = rng.choice(available_radii, size=N)


#%% 7. Select complex f_bs(f) for each bubble
# Nearest-neighbor depth lookup; exact radius lookup
depth_idx = np.abs(depth_db[:, None] - depths[None, :]).argmin(axis=0)
radius_idx = np.searchsorted(radius_db, bubble_radii)

f_bs_bubbles = f_bs_db[depth_idx, radius_idx, :]

print(f"Number of bubbles: {N}")
print(f"Mean simulated depth: {depths.mean():.3f} m")
print(f"Bubble spectra shape: {f_bs_bubbles.shape}")


#%% 8. Coherent broadband backscattering
k = 2 * np.pi * freq_vec / c_w

# Two-way propagation phase and spreading
phase = np.exp(-2j * ranges[:, None] * k[None, :])
p_bubbles = f_bs_bubbles * phase / ranges[:, None]**2

p_far = np.sum(p_bubbles, axis=0)

power_coherent = np.abs(p_far)**2
power_incoherent = np.sum(np.abs(p_bubbles)**2, axis=0)


#%% 9. Plot bubble distribution and spectra
fig = plt.figure(figsize=(16, 5))

# 3D bubble cloud
ax1 = fig.add_subplot(131, projection="3d")
plot_bubble_cloud(points, bubble_radii, source=Observation_point,
                  source_direction=(0, 0, 1), beam_half_angle_deg=beam_half_angle_deg,
                  ax=ax1, margin=1.1, view=(15, -55), invert_z=True, size_scale=2)

# Depth distribution
ax2 = fig.add_subplot(132)
ax2.hist(depths, bins=25, edgecolor="black", alpha=0.7)
ax2.axvline(Mean_depth, color="red", linestyle="--", label="Mean depth")
ax2.set_xlabel("Depth (m)")
ax2.set_ylabel("Number of bubbles")
ax2.set_title(f"Normal depth distribution (σ={Depth_std:g} m)")
ax2.legend()

# Broadband backscattering spectra
ax3 = fig.add_subplot(133)
ax3.plot(freq_vec/1e3, 10*np.log10(np.maximum(power_coherent, 1e-300)), label="Coherent")
ax3.plot(freq_vec/1e3, 10*np.log10(np.maximum(power_incoherent, 1e-300)), "--", label="Incoherent")
ax3.set_xlabel("Frequency (kHz)")
ax3.set_ylabel("Relative level (dB)")
ax3.set_title("Combined bubble backscattering")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig(output_path("bubble_backscatter_summary.png"), dpi=300, bbox_inches="tight")
plt.show()
