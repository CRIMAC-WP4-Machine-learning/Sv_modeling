"""
Coherent superposition of backscatter from a random fish school (N fish in a
spheroid at ~50 m depth). Optionally compare with FEM/BEM results.

Run from anywhere, e.g. from the repository root:
    uv run python scripts/DistributeFish_sv.py

Figures are written to outputs/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from sv_modeling.database import (
    Extract_database_metadata,
    Find_closestfile_in_database,
    func_get_frq_TS_fbs_Dict,
)
from sv_modeling.fembem import func_FEM_BEM
from sv_modeling.geometry import Get_Angle
from sv_modeling.paths import FEMBEM_DIR, TS_DATABASE_150HZ_DIR, output_path
from sv_modeling.plotting import plot_fish_school
from sv_modeling.signals import func_Gen_SweepSig_Explicit
from sv_modeling.spectrum import ESD_BKH

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
T_effectiveDuration=2*1.024*1E-3 # Sweep duration (sec)
T_center=10*1.024*1E-3
T_whole=12*1.024*1E-3     # Whole time signal with zeros (sec)

Fs=512*512*16*2 # Sampling rate (Hz)
F_start=160000  # Start frequency (Hz)
F_end=260000    # End frequency (Hz)
mexpo=25        # Fast ramping. For slow ramping select smaller value

[t_vec,p_vec]=func_Gen_SweepSig_Explicit(T_whole,T_center,T_effectiveDuration,Fs,F_start,F_end,mexpo)
dt = np.mean(np.diff(t_vec))
Output = ESD_BKH(p_vec, dt)
freq_vec = Output[:,0]
ESD_vec = Output[:,1]
df = 1 / (len(p_vec) * dt)



FIG=plt.figure(figsize=(8,7)) 
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  
ax = FIG.add_subplot(211)
box = ax.get_position()
#        ax.set_position([box.x0+0.0, box.y0+(0.01), box.width * 1 , box.height * 1.0])
ax.set_position([box.x0, box.y0, box.width * 1.0 , box.height * 1.0])
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
plt.plot(t_vec,p_vec)

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  
ax = FIG.add_subplot(212)
box = ax.get_position()
#        ax.set_position([box.x0+0.0, box.y0+(0.01), box.width * 1 , box.height * 1.0])
ax.set_position([box.x0, box.y0, box.width * 1.0 , box.height * 1.0])
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
plt.plot(freq_vec, ESD_vec, color = [0,0,1])
plt.show()

#Energy of signal in time and frequency domains:
E_time = np.sum(np.abs(p_vec)**2) * dt

E_freq = np.sum(ESD_vec) * df

print(f'E_time={E_time:.6e}')
print(f'E_freq={E_freq:.6e}')
#%% ##====================================================


FEM_BEM_files = ['FEM_BEM_II_FourTargets_TS_a1cm_b4mm_rho7p34_90deg_D1_0p1m_D2_0p25m_D3_z0p25m_x0p25m.txt']

[f_bem, TS_bem] = func_FEM_BEM(FEM_BEM_dir, FEM_BEM_files[0])
#====================================
#%% Test functions

# # For a given point:
# # Incident_Angle_deg:
# # Angle between "u" the unit vector of point "p1" and vector connecting "p0" to "p1":
# p0 = np.array([0.0, 0.0, 5.0]) 
# p1 = np.array([-1.0, 0.0, 0.0])
# u_p1 = np.array([1.0, 0.0, 0.0])  # orientation at p1
# Incident_Angle_deg = Get_Angle(p0, p1, u_p1)

# # Get the TS(f) and fbs(f) for fish with length and tilt angle at given depth:
# Depth = 50 
# Length = 0.30
# IncAngl = 88

# # Find the file in database (modeled .csv files) closest to the size of fish at "point_ii" with "orientation_ii"
# target_dict = Find_closestfile_in_database(df_database, Depth, Length, IncAngl)
# print('target_dict: ', target_dict)
# L_fish = Length
# [freq_scaled, TS_scaled, scaled_f_bs] = func_get_frq_TS_fbs_Dict(database_dir, target_dict, L_fish)
# # plt.plot(freq_scaled, 20*np.log10(np.abs(scaled_f_bs)), color = [1, 0, 0], dashes = [3,2], linewidth = 2)
# plt.plot(freq_scaled, TS_scaled, color = [1, 0, 0], dashes = [3,2], linewidth = 2)
# plt.show()


#%% main part
Observation_point = np.array([0, 0, 0]) # Echosounder location

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Example: place N points (fish location) with no overlap in a prolate or oblate spheroid
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
N = 200
a, b = 2.0, 0.6  # spheroid axes
points = []
min_dist = 0.1
Average_school_Depth = 50 # m

# "Theta" is the angle of projected vector on XY plane and X axis
theta_range = np.array([-0.01, 0.01])*np.pi/180

# "Phi" is the angle of vector and Z axis
phi_range = np.array([89.99, 90.01])*np.pi/180


# I. Create N points(x, y, z) with min_dist to avoid overlap:
while len(points) < N:
    x = np.random.uniform(-a, a) 
    y = np.random.uniform(-a, a)
    z = np.random.uniform(-b, b) - Average_school_Depth
    if (x**2/a**2 + y**2/a**2 + (z + Average_school_Depth)**2/b**2) <= 1:
        p = np.array([x, y, z])
        if all(np.linalg.norm(p - q) > min_dist for q in points):
            points.append(p)

points = np.array(points)
# print(points)



# II. Orientaion of fish - Random unit vectors: 
# "Theta" is the angle of projected vector on XY plane and X axis
# "Phi" is the angle of vector and Z axis

# RANDOM distribution:   -----------------------------------------------
# Generate arrays of theta and phi: 
theta = np.random.uniform(theta_range[0], theta_range[1], size=len(points))
phi = np.random.uniform(phi_range[0], phi_range[1], size=len(points))

# # NORMAL distribution:   -----------------------------------------------
# Parameters
mu = np.mean(theta_range)       # mean
sigma = 0.25*(theta_range[1]-theta_range[0])     # standard deviation

theta = np.random.normal(mu, sigma, N)

# Plot histogram
count, bins, ignored = plt.hist(180*theta/np.pi, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
plt.xlabel('$\\theta$',fontsize = 12)
plt.show()

# Parameters
mu = np.mean(phi_range)       # mean
sigma = 0.25*(phi_range[1]-phi_range[0])     # standard deviation

phi = np.random.normal(mu, sigma, N)

# Plot histogram
count, bins, ignored = plt.hist(180*phi/np.pi, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
plt.xlabel(r'$\phi$',fontsize = 12)
plt.show()

# Convert spherical to Cartesian coordinates
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

orientations = np.stack((x, y, z), axis=1)  # shape (N, 3)
orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]  # normalize

# print(orientations[0])


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#                       Test case for two fish:  
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
# points = []
# p = np.array([0, 0, - Average_school_Depth])
# points.append(p)

# p = np.array([0, 0, - Average_school_Depth -  0.1])
# points.append(p)

# p = np.array([0, 0, - Average_school_Depth - 0.25])
# points.append(p)

# p = np.array([0, -0.25, - Average_school_Depth - 0.25])
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
# # ====================================================


# Plot the fish school of N fish
plot_fish_school(points, orientations)
plt.savefig(output_path("Plot_"+os.path.splitext(FEM_BEM_files[0])[0]+".png"), dpi=300, bbox_inches='tight')  # PNG
plt.show() 

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# print('len(points): ', len(points))
p_far = 0
for ii in range(0, len(points)):
    point_ii = points[ii]
    orientations_ii = orientations[ii]

    Incident_Angle_ii = Get_Angle(Observation_point, point_ii, orientations_ii)

    # print('Incident_Angle_ii: ', Incident_Angle_ii)

    Length = 0.0 # m.  This can be changed to include distribution of ranges

    # Find the file in database (modeled .csv files) closest to the size of fish at "point_ii" with "orientation_ii"
    target_dict = Find_closestfile_in_database(df_database, np.abs(point_ii[2]), Length, Incident_Angle_ii)
    
    L_fish = Length # m 
    [freq, TS_ii, f_bs_ii] = func_get_frq_TS_fbs_Dict(database_dir, target_dict, L_fish)
    # print(freq.shape)
    Distance = np.linalg.norm(Observation_point-point_ii)
    # print('Distance: ', Distance)
    p_far_ii = (f_bs_ii/(Distance*Distance)) * np.exp(-1j*2*np.pi*(1000*freq/1500)*(2*Distance))  # k = 2*pi*f/c
    p_far = p_far + p_far_ii
    # plt.plot(freq, 20*np.log10(np.abs(f_bs_ii)), color = [1, 0, 0], dashes = [3,2], linewidth = 2)


point_ii = points[0]
Distance = np.linalg.norm(Observation_point-point_ii)
Total_p_TS = 20*np.log10(np.abs(Distance*Distance * p_far))
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
    # title='Legend Title', # optional title
    # title_fontsize=14,    # title font size
    # frameon=True,         # draw a frame around the legend
    # shadow=True,          # add shadow
    facecolor='white',    # legend background color
    # edgecolor='black'     # legend border color
)

# target_dict = Find_closestfile_in_database(df_database, Depth, Length, IncAngl)
# print('target_dict:>>>>>>> ',target_dict)

# L_fish = 0.15
# [freq_scaled, TS_scaled, scaled_f_bs] = func_get_frq_TS_fbs_Dict(database_dir, target_dict, L_fish)

# plt.plot(freq_scaled, TS_scaled, color = [0, 0, 0], dashes = [3,0], linewidth = 2)
# plt.plot(freq_scaled, 20*np.log10(np.abs(scaled_f_bs)), color = [1, 0, 0], dashes = [3,2], linewidth = 2)
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
