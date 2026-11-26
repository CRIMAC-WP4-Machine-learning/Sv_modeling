
"""
Simulate coherent broadband backscattering from spherical gas bubbles.

Bubbles are randomly distributed within a conical beam shell
between slant ranges R-DR/2 and R+DR/2.

Complex f_bs(f) spectra are read from spherical CSV databases.

Run from anywhere, e.g. from the repository root:
    uv run python scripts/DistributeFish_IdealSpect_sv_beam_spher_bubbles.py
    uv run python scripts/DistributeFish_IdealSpect_sv_beam_spher_bubbles.py --config path/to/other.yaml

Parameters are read from configs/DistributeFish_IdealSpect_sv_beam_spher_bubbles.yaml unless --config is given.
Figures are written to outputs/.
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

from sv_modeling.config import BeamShellSchool, config_from_cli
from sv_modeling.paths import TS_DATABASE_SPHERE_DIR, output_path
from sv_modeling.plotting import plot_bubble_cloud
from sv_modeling.density import  estimate_density_sphericalbubbles, beam_shell_volume
from sv_modeling.school import place_fish_beam_shell
from sv_modeling.school import _truncated_normal
from sv_modeling.scattering import coherent_backscatter_sphericalbubbles

#%% 1. Simulation parameters 

#======================================================
# Configuration (configs/DistributeFish_IdealSpect_sv_beam_spher_bubbles.yaml or --config)
cfg = config_from_cli("DistributeFish_IdealSpect_sv_beam_spher_bubbles.yaml", description=__doc__)
cfg.require("echosounder.beam_direction", "echosounder.beam_half_angle_deg", "bubbles", "analysis")
school = cfg.school_as(BeamShellSchool)
bub = cfg.bubbles

Observation_point = np.array(cfg.echosounder.position_m, dtype=float)

Gas = bub.gas

N = school.n_targets
rng = np.random.default_rng(cfg.seed)  # seed: null in the config gives a new cloud each run

min_dist = school.min_dist_m  # Minimum separation between bubbles (m)

R = school.mean_range_m  # Mean slant range (m)
DR = school.thickness_m  # Slant-range shell thickness (m)

transducer_depth = bub.transducer_depth_m  # Depth below sea surface (m)

beam_half_angle_deg = cfg.echosounder.beam_half_angle_deg
beam_half_angle = np.deg2rad(beam_half_angle_deg)

c_w = bub.sound_speed_m_s

R_bubble_min = bub.radius_min_m  # m
R_bubble_max = bub.radius_max_m  # m
R_bubble_resolution = bub.radius_resolution_m  # m
#======================================================

# Generate targets uniformly in the circular conical shell
points = place_fish_beam_shell(N, R, DR, beam_half_angle_deg, min_dist, rng)

# Generate N bubble radii from a truncated normal distribution
bubble_radii = _truncated_normal(
    R_bubble_min, R_bubble_max, N, rng
)

# Quantize radii to the database resolution
radius_indices = np.rint(
    (bubble_radii - R_bubble_min) / R_bubble_resolution
).astype(int)

n_radius_steps = round(
    (R_bubble_max - R_bubble_min) / R_bubble_resolution
)

radius_indices = np.clip(radius_indices, 0, n_radius_steps)

bubble_radii = (
    R_bubble_min + radius_indices * R_bubble_resolution
)



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
    sound_speed=c_w,
    transducer_depth=transducer_depth,
)

#%% 3. Estimate bubble density

f1, f2 = cfg.analysis.band_hz

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
                  source_direction=cfg.echosounder.beam_direction, beam_half_angle_deg=beam_half_angle_deg,
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