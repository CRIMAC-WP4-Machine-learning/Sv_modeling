'''
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
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob
import pandas as pd
import re
from scipy.signal import savgol_filter

# from src.inversion.liquid_ps import inverse_frequency_response

#%% funcs -----------------------------------
def Get_Angle(_p0, _p1, _u_p1):
    ''' 
    Returns the angle between a vector connecting p1 to p0 and the vector u_p1 originating from p1
    Inputs:
    _p0 = np.array([x0, y0, z0])  observation point (coordinate of echosounder)
    _p1 = np.array([x1, y1, z1])  location of fish (coordinate of fish)
    _u_p1 = np.array([1.0, 1.0, 0.0])  #  (orientation at p1, unit vector) Orientation of the fish

    Outputs:
    
    Example:
    # ======================================================
    # Angle between "u" the unit vector of point "p1" and vector connecting "p0" to "p1":
    print('points[0], orientations[0]: ', points[0], orientations[0])

    p0 = np.array([0.0, 0.0, 5.0]) 
    p1 = np.array([0.0, 0.0, 0.0])
    u_p1 = np.array([1.0, 0.0, 0.0])  # orientation at p1
    Get_Angle(p0, p1, u_p1)

    print('np.linalg.norm(p1-p0): ', np.linalg.norm(p1-p0))
    '''
    # _p0 = np.array([0.0, 0.0, 5.0]) 
    # _p1 = np.array([0.0, 0.0, 0.0])
    # _u_p1 = np.array([1.0, 1.0, 0.0])  # orientation at p1
    

    _u_p1_norm = np.linalg.norm(_u_p1)  # unit vector (orientation at p1)


    v = _p0 - _p1
    v_norm = np.linalg.norm(v)

    # Ensure v is not zero-length
    if v_norm > 0:
        cos_theta = np.dot(_u_p1, v) / (v_norm*_u_p1_norm)
        # Clip for numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        # print(f"Angle: {theta_deg:.2f} degrees")
    else:
        print("Warning: p0 and p1 are the same point.")
    
    Incident_Angle_deg = 90 - np.abs(90-theta_deg)
    return Incident_Angle_deg

def Extract_database_metadata(_Dir):
    '''
    Reads all .csv files in _Dir and returns the metadata as dataframe including file names, a, b, depth, Incident angle, 
    and lengths from filenames
    '''
    csv_files = glob.glob(os.path.join(_Dir, '*.csv'))

    # Regex pattern to extract parameters from filename
    pattern = r"a_(?P<a>[0-9.]+)_b_(?P<b>[0-9.]+).*?_IncAngle_(?P<IncAngle>[0-9.]+)_depth_(?P<depth>[0-9.]+)_length_(?P<length>[0-9.]+)"

    # Collect data from filenames
    records = []

    for file in csv_files:
        match = re.search(pattern, file)
        if match:
            record = {
                'filename': os.path.basename(file),
                'a': float(match.group('a')),
                'b': float(match.group('b')),
                'IncAngle': float(match.group('IncAngle')),
                'depth': float(match.group('depth')),
                'length': float(match.group('length'))
            }
            records.append(record)

    # Create DataFrame
    _df = pd.DataFrame(records)

    # View it
    # print(_df.head())

    return _df

def Find_closestfile_in_database(_df_database, _Depth, _Length, _IncAngl):
    """
    Reads the CSV file for the corresponding fish length, depth, and incident angle.
    Output:
        Returns the file in database (modeled .csv files) closest to the size of fish at "point_ii" with "orientation_ii".
    """
    # Make a copy to avoid modifying the original DataFrame
    df_db = _df_database.copy()

    # Step 1: Closest depth
    df_db['depth_diff'] = np.abs(df_db['depth'] - _Depth)
    min_depth_diff = df_db['depth_diff'].min()
    depth_filtered = df_db[df_db['depth_diff'] == min_depth_diff].copy()

    # Step 2: Closest length >= _Length
    larger_or_equal = depth_filtered[depth_filtered['length'] >= _Length]
    if not larger_or_equal.empty:
        selected_length = larger_or_equal['length'].min()
        length_filtered = larger_or_equal[larger_or_equal['length'] == selected_length].copy()
    else:
        # Fallback: pick overall closest length
        depth_filtered['length_diff'] = np.abs(depth_filtered['length'] - _Length)
        min_length_diff = depth_filtered['length_diff'].min()
        length_filtered = depth_filtered[depth_filtered['length_diff'] == min_length_diff].copy()

    # Step 3: Closest incident angle
    length_filtered['angle_diff'] = np.abs(length_filtered['IncAngle'] - _IncAngl)
    min_angle_diff = length_filtered['angle_diff'].min()
    final_selection = length_filtered[length_filtered['angle_diff'] == min_angle_diff]

    # print('final_selection: ', final_selection)

    # Take the first row if multiple matches
    df_closest_row = final_selection.iloc[0]
    dict_closest_row = df_closest_row.to_dict()

    return dict_closest_row


# def func_frq_TS_from_Dict(_Dir, _Inp_Dict, _Lfish, _f_vec):
#     # print('_Inp_Dict :', _Inp_Dict)
#     # print(os.path.join(_Dir, _Inp_Dict['filename']))
#     df_csv = pd.read_csv(os.path.join(_Dir, _Inp_Dict['filename']))
#     print(df_csv.columns)


#     L0 = _Inp_Dict['length']
    
#     # print(' type(df_csv): ', type(df_csv))
#     freq_vec0 = df_csv['Freq_kHz']
#     TS_vec0 = df_csv['TS']
#     Fbs_vec0 = df_csv['f_bs']

#     # Ensure f_bs is complex. That is if f_bs is stored as string (e.g., "1+2j"), convert it to complex
#     if not np.iscomplexobj(Fbs_vec0):
#         Fbs_vec0 = Fbs_vec0.astype(complex)

#     ScaleFactor=(L0/_Lfish)**(1/3) # !!!!! Req0/Req = L0/_Lfish: Note that since in the current version, 
#                             #           "b" the minor axis of prolate spheroid is indepndent of L
#     plt.plot(freq_vec0, TS_vec0)
#     print('ScaleFactor: >>>>>>>>>>>>>>>>>>>>>>', ScaleFactor)
    
#     def Func_rescale_TS(_t_vec, _Sig, _factor):
#         _t_vec = np.asarray(_t_vec)   # Convert pandas Series or list to ndarray
#         _t_vec = np.insert(_t_vec, 0, 0.0)        # Insert 0.0 at the beginning

#         _Sig = np.asarray(_Sig)  # Convert pandas Series or list to ndarray
#         _Sig = np.insert(_Sig, 0, 1E-200)        # Insert 0.0 at the beginning

#         Scaled_t = _factor * _t_vec
#         Sig_interpolated = np.interp(_f_vec, Scaled_t, _Sig)
#         ScaledSig = Sig_interpolated + 20 * np.log10(1 / _factor)

#         return _f_vec, ScaledSig
    
#     def Func_rescale_Fbs(_t_vec, _Sig, _factor):
#         _t_vec = np.asarray(_t_vec)   # Convert pandas Series or list to ndarray
#         _t_vec = np.insert(_t_vec, 0, 0.0)        # Insert 0.0 at the beginning
#         _Sig = np.asarray(_Sig)
#         _Sig = np.insert(_Sig, 0, 1E-200)        # Insert 0.0 at the beginning

#         Scaled_t = _factor * _t_vec
#         Sig_interpolated = np.interp(_f_vec, Scaled_t, _Sig)
#         ScaledSig = Sig_interpolated  * 1/_factor

#         return ScaledSig

    
#     [freq_vec, TS_vec] = Func_rescale_TS(freq_vec0, TS_vec0, ScaleFactor)
#     Real_f_bs = Func_rescale_Fbs(freq_vec0, np.real(Fbs_vec0), ScaleFactor)
#     Imag_f_bs = Func_rescale_Fbs(freq_vec0, np.imag(Fbs_vec0), ScaleFactor)
#     scaled_f_bs = Real_f_bs + 1j * Imag_f_bs

#     return freq_vec, TS_vec, scaled_f_bs

def func_get_frq_TS_fbs_Dict(_Dir, _Inp_Dict, _Lfish):
    '''
    _Dir: directory of modeled TS and fbs as function of frequency
    _Inp_Dict: is the output of function "Find_closestfile_in_database" which contains filename in _Dir which is 
               closest to _Lfish at a given Depth and tilt angle 
    '''
    df_csv = pd.read_csv(os.path.join(_Dir, _Inp_Dict['filename']))
    # TS_file = 'ts_vs_freq_loop_herring_a_0.01000_b_0.00400_f1_0_f2_100_rhos_7.34_IncAngle_90_depth_50_length_0.0_iterRef_LU.csv'
    # df_csv = pd.read_csv(os.path.join(_Dir, TS_file))
    # print(df_csv.columns)

    # print(' type(df_csv): ', type(df_csv))
    freq_vec0 = df_csv['Freq_kHz']
    TS_vec0 = df_csv['TS']
    Fbs_vec0 = df_csv['f_bs']
    
    # Ensure f_bs is complex. That is if f_bs is stored as string (e.g., "1+2j"), convert it to complex
    if not np.iscomplexobj(Fbs_vec0):
        Fbs_vec0 = Fbs_vec0.astype(complex)

    return freq_vec0, TS_vec0, Fbs_vec0


def plot_fish_school(_points, _orientations, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    ax.quiver(_points[:, 0], _points[:, 1], _points[:, 2],
            _orientations[:, 0], _orientations[:, 1], _orientations[:, 2],
            length=0.3, normalize=True)

    # Equal aspect ratio
    max_range = np.array([
        _points[:, 0].max() - _points[:, 0].min(),
        _points[:, 1].max() - _points[:, 1].min(),
        _points[:, 2].max() - _points[:, 2].min()
    ]).max() / 2.0

    mid_x = (_points[:, 0].max() + _points[:, 0].min()) * 0.5
    mid_y = (_points[:, 1].max() + _points[:, 1].min()) * 0.5
    mid_z = (_points[:, 2].max() + _points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - 1.1*max_range, mid_x + 1.1*max_range)
    ax.set_ylim(mid_y - 1.1*max_range, mid_y + 1.1*max_range)
    ax.set_zlim(mid_z - 1.1*max_range, mid_z + 1.1*max_range)
    ax.view_init(elev=25, azim=-60)
    ax.invert_zaxis()
    # plt.show()    

#%% Initialize Directories and Databases:

# 1. Get the directory where the script is located
ProjectDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ProjectDIR)

print('ProjectDIR: ',ProjectDIR)
# sys.path.insert(0, ProjectDIR)

# 2. Calculate the parent directory (navigates two directories up.)
ParentDIR = os.path.abspath(os.path.join(os.getcwd()))
print('Parent DIR: >>> ', ParentDIR)

# Database Directory containing csv files with TS(f) and F_bs(f):
database_dir = os.path.join(ParentDIR, 'model_results','model_backscatter_150Hz/')
print('database_dir: ',database_dir)

FEM_BEM_dir = os.path.join(ParentDIR, 'FEM_BEM/')
print('FEM_BEM_dir: ',FEM_BEM_dir)
# Create database info from csv files in database_dir. df_database has filename, target (fish) length, incident angle, depth,
#  prolate spheroid dimensions of swimblader "a, b", 
df_database = Extract_database_metadata(database_dir)

from src.Func_FEMBEM import func_FEM_BEM
from src.FUNC_ESD_Bkh import ESD_BKH
from src.FUNC_Gen_SweepSig_Explicit import func_Gen_SweepSig_Explicit

#%%====================================================
# Frequency interval (Hz)
f1 = 10000.0
f2 = 99000.0

# 1: Define an idealized flat spectrum ============================
# Frequency step for coherent simulation
df = 100.0

freq_vec = np.arange(f1, f2 + df / 2, df)

# Flat spectral weighting
ESD_vec = np.ones_like(freq_vec)

# Integral of the spectral weighting
tips_0 = np.trapezoid(ESD_vec, freq_vec)


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
N = 200
points = []
min_dist = 0.1

School_Mean_range = 50.0
School_thikness = 1

beam_half_angle = np.deg2rad(3.5)

R = School_Mean_range
DR = School_thikness

r_min = R - DR / 2
r_max = R + DR / 2

rng = np.random.default_rng()

# Generate targets uniformly in the circular conical shell
while len(points) < N:

    # Uniform sampling in volume
    r = np.cbrt(rng.uniform(r_min**3, r_max**3))

    # Polar angle from acoustic axis (+Z)
    cos_gamma = rng.uniform(np.cos(beam_half_angle), 1.0)
    gamma = np.arccos(cos_gamma)

    # Azimuth in XY plane, measured from +X toward +Y
    azimuth = rng.uniform(0, 2 * np.pi)

    # Cartesian coordinates
    x = r * np.sin(gamma) * np.cos(azimuth)
    y = r * np.sin(gamma) * np.sin(azimuth)
    z = r * np.cos(gamma)

    p = np.array([x, y, z])

    # Minimum distance between targets
    if all(np.linalg.norm(p - q) >= min_dist for q in points):
        points.append(p)

points = np.asarray(points)

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


# II. Orientaion of fish - Random unit vectors: 
# "Theta" is the angle of projected vector on XY plane and X axis
# "Phi" is the angle of vector and Z axis

# RANDOM distribution:   -----------------------------------------------
# "Theta" is the angle of projected vector on XY plane and X axis
theta_range = np.array([-0.001, 0.001])*np.pi/180

# "Phi" is the angle of vector and Z axis
phi_range = np.array([89.999, 90.001])*np.pi/180

# Generate arrays of theta and phi: 
theta = np.random.uniform(theta_range[0], theta_range[1], size=len(points))
phi = np.random.uniform(phi_range[0], phi_range[1], size=len(points))

# # NORMAL distribution:   -----------------------------------------------
# Parameters
mu = np.mean(theta_range)       # mean
sigma = 0.25*(theta_range[1]-theta_range[0])     # standard deviation

theta = np.random.normal(mu, sigma, N)

# # Plot histogram
# count, bins, ignored = plt.hist(180*theta/np.pi, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
# plt.xlabel('$\\theta$',fontsize = 12)
# plt.show()

# Parameters
mu = np.mean(phi_range)       # mean
sigma = 0.25*(phi_range[1]-phi_range[0])     # standard deviation

phi = np.random.normal(mu, sigma, N)

# # Plot histogram
# count, bins, ignored = plt.hist(180*phi/np.pi, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
# plt.xlabel(r'$\phi$',fontsize = 12)
# plt.show()

# Convert spherical to Cartesian coordinates
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

orientations = np.stack((x, y, z), axis=1)  # shape (N, 3)
orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]  # normalize


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

r_min = R - DR / 2
r_max = R + DR / 2
alpha = beam_half_angle

# Exact beam-sampled volume (m³)
V = (2 * np.pi / 3) * (1 - np.cos(alpha)) * (r_max**3 - r_min**3)

# Approximate volume for narrow beam and thin range interval
V_approx = np.pi * R**2 * alpha**2 * DR

# Fish number density (fish/m³)
rho_N = len(points) / V

print(f"Number of fish       : {len(points)}")
print(f"Aggregate volume     : {V:.3f} m³")
print(f"Approximate volume   : {V_approx:.3f} m³")
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
#%% ====================================================


# Plot the fish school of N fish
fig = plt.figure(figsize=(12, 5))

# 1. Theta histogram
ax1 = fig.add_subplot(131)
ax1.hist(np.degrees(theta), bins=30, density=True, alpha=0.6, color="skyblue", edgecolor="black")
ax1.set_xlabel(r"$\theta$ (deg)", fontsize=12)
ax1.set_ylabel("Probability density", fontsize=12)
ax1.set_title("Azimuth distribution")

# 2. Phi histogram
ax2 = fig.add_subplot(132)
ax2.hist(np.degrees(phi), bins=30, density=True, alpha=0.6, color="skyblue", edgecolor="black")
ax2.set_xlabel(r"$\phi$ (deg)", fontsize=12)
ax2.set_ylabel("Probability density", fontsize=12)
ax2.set_title("Polar angle distribution")

# 3. Fish school in 3D
ax3 = fig.add_subplot(133, projection="3d")
plot_fish_school(points, orientations, ax3)

plt.tight_layout()

plt.show() 

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Frequency interval (kHz, matching freq from database)
# Frequency interval (kHz)
# f1 = 1
# f2 = 100

N = len(points)

sigma_bs_P_fish = np.zeros(N)
p_far = np.zeros(len(freq_vec), dtype=complex)
R0 = 1

for ii in range(0, len(points)):
    point_ii = points[ii]
    orientations_ii = orientations[ii]

    Incident_Angle_ii = Get_Angle(
        Observation_point, point_ii, orientations_ii
    )

    Length = 0.0

    target_dict = Find_closestfile_in_database(
        df_database,
        np.abs(point_ii[2]),
        Length,
        Incident_Angle_ii
    )

    L_fish = Length

    [freq, TS_ii, f_bs_ii] = func_get_frq_TS_fbs_Dict(database_dir, target_dict, L_fish)


    # ---------------------------------------------------
    # Flat-spectrum averaged sigma_bs for fish ii
    # ---------------------------------------------------

    freq = np.asarray(freq, dtype=float)  # kHz
    f_bs_ii = np.asarray(f_bs_ii, dtype=complex)
    freq_Hz = freq * 1000

    # Interpolate complex scattering amplitude onto the 100 Hz grid
    f_bs_interp = (
        np.interp(freq_vec, freq_Hz, f_bs_ii.real)
        + 1j * np.interp(freq_vec, freq_Hz, f_bs_ii.imag)
    )

    # Flat-spectrum mean scattering cross section
    sigma_bs_ii = np.abs(f_bs_interp)**2

    sigma_bs_P_fish[ii] = (
        np.trapezoid(sigma_bs_ii, freq_vec)
        / (f2 - f1)
    )

    # Coherent scattering contribution
    Distance = np.linalg.norm(Observation_point - point_ii)

    p_far_ii = (
        f_bs_interp / Distance**2
        * np.exp(-1j * 4*np.pi * freq_vec * Distance / 1500)
    )

    p_far = p_far + p_far_ii

#%% Calculate number density using idealized flat spectrum

sigma_bs_P_mean = np.mean(sigma_bs_P_fish)

tips_GV_coh_normalized = (
    R0**2
    * np.trapezoid(np.abs(p_far)**2, freq_vec)
    / (f2 - f1)
)

R = School_Mean_range
V_GV = V

n_b_est_acoustic = (
    tips_GV_coh_normalized * R**4
    / (V_GV * R0**2 * sigma_bs_P_mean)
)

n_b_true = N / V_GV

R_i = np.linalg.norm(points - Observation_point, axis=1)

tips_GV_incoh_normalized = (
    R0**2 * np.sum(sigma_bs_P_fish / R_i**4)
)

n_b_incoh = (
    tips_GV_incoh_normalized * R**4
    / (V_GV * R0**2 * sigma_bs_P_mean)
)

print(f"Coherent density acoustics    : {n_b_est_acoustic:.4f} fish/m³")
print(f"Incoherent density : {n_b_incoh:.4f} fish/m³")
print(f"True density       : {n_b_true:.4f} fish/m³")
print(f"Coherent/Incoherent: {tips_GV_coh_normalized/tips_GV_incoh_normalized:.4f}")