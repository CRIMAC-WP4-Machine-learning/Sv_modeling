
"""
Simulate coherent broadband backscattering from spherical gas bubbles.

Bubbles are randomly distributed within a conical beam shell
between slant ranges R-DR/2 and R+DR/2.

Complex f_bs(f) spectra are read from spherical CSV databases.
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

from sv_modeling.paths import TS_DATABASE_SPHERE_DIR, output_path
from sv_modeling.plotting import plot_bubble_cloud
from sv_modeling.density import  estimate_density_sphericalbubbles, beam_shell_volume
from sv_modeling.school import place_fish_beam_shell
from sv_modeling.school import _truncated_normal
from sv_modeling.scattering import coherent_backscatter_sphericalbubbles

#%% 1. Simulation parameters 

#======================================================
Observation_point = np.array([0.0, 0.0, 0.0])

Gas = "CO2"

N = 1000
# seed = 42
# rng = np.random.default_rng(seed)
rng = np.random.default_rng()

min_dist = 0.01  # Minimum separation between bubbles (m)

Mean_depth = 55.0
# Depth_std = 1.0
R = Mean_depth
DR = 1 # Layer thickness

beam_half_angle_deg = 3.5
beam_half_angle = np.deg2rad(beam_half_angle_deg)

c_w = 1500.0

R_bubble_min = 0.0005       # m
R_bubble_max = 0.0020       # m
R_bubble_resolution = 0.0001  # m
#======================================================

# Generate targets uniformly in the circular conical shell
points = place_fish_beam_shell(N, R, DR, beam_half_angle_deg, min_dist, rng)

bubble_radii = _truncated_normal(R_bubble_min, R_bubble_max, N, rng)
bubble_radii = (R_bubble_min + np.round( (bubble_radii - R_bubble_min) / R_bubble_resolution) * R_bubble_resolution)
bubble_radii = np.clip(bubble_radii, R_bubble_min, R_bubble_max) # N bubbles sizes with normal distribution



# Exact beam-sampled volume (m³); approximate=True gives the narrow-beam form
V = beam_shell_volume(R, DR, beam_half_angle_deg)

# Bubble number density (bubbles/m³)
rho_N = len(points) / V

print(f"Number of bubbles : {len(points)}")
print(f"Sampled volume    : {V:.3f} m³")
print(f"Bubble density    : {rho_N:.3f} bubbles/m³")

#%%

#%% 2. Calculate coherent backscattering

freq_hz, p_far, sigma_bs, distances = coherent_backscatter_sphericalbubbles(
    points=points,
    bubble_radii=bubble_radii,
    database_dir=TS_DATABASE_SPHERE_DIR,
    gas=Gas,
    observation_point=Observation_point,
    sound_speed=c_w
)

#%% 3. Estimate bubble density

f1 = 45000
f2 = 90000

estimate = estimate_density_sphericalbubbles(
    freq_hz, p_far, sigma_bs, distances, R, V, (f1, f2)
)

svf = estimate.svf

print(f"Density from Coh. Sv : {estimate.coherent:.4f} bubbles/m³")
print(f"Density from Incoh. Sv: {estimate.incoherent:.4f} bubbles/m³")
print(f"True density      : {estimate.true:.4f} bubbles/m³")
print(f"Coherent/True     : {estimate.coherent / estimate.true:.4f}")

#%% 4. Calculate incoherent volume backscattering

svf_incoherent = R**4 / V * np.sum(
    sigma_bs / distances[:, None]**4, axis=0
)

fig = plt.figure(figsize=(16, 5))

# 3D bubble cloud
ax1 = fig.add_subplot(131, projection="3d")
plot_bubble_cloud(points, bubble_radii, source=Observation_point,
                  source_direction=(0, 0, 1), beam_half_angle_deg=beam_half_angle_deg,
                  ax=ax1, margin=1.1, view=(15, -55), invert_z=True, size_scale=2)

# Bubble radius distribution
ax2 = fig.add_subplot(132)
bins = np.arange(1000*R_bubble_min, 1000*(R_bubble_max+R_bubble_resolution), 1000*R_bubble_resolution)
ax2.hist(bubble_radii * 1e3, bins=bins, edgecolor="black", alpha=0.7)
ax2.set_xlabel("Bubble radius (mm)")
ax2.set_ylabel("Number of bubbles")
ax2.set_title(f"Bubble size distribution (N={N})")
ax2.grid(True, alpha=0.3)

# Volume backscattering strength
ax3 = fig.add_subplot(133)
ax3.plot(freq_hz/1e3, 10*np.log10(np.maximum(svf, 1e-300)), label="Coherent")
ax3.plot(freq_hz/1e3, 10*np.log10(np.maximum(svf_incoherent, 1e-300)), "--", label="Incoherent")
ax3.set_xlabel("Frequency (kHz)")
ax3.set_ylabel(r"$S_v(f)$ (dB re 1 m$^{-1}$)")
ax3.set_title("Volume backscattering strength")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig(output_path("bubble_backscatter_summary.png"), dpi=300, bbox_inches="tight")
plt.show()