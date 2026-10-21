'''
To run, be in Prol_Spheroid_herring directory. for example 
run cd /root/projects/Prol_Spheroid_herring/ 
Then use:
python -m src.distribution.DistributeFish_2_wFEMBEM
to run the script
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

    print('final_selection: ', final_selection)

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


def plot_fish_school(_points, _orientations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(_points[:, 0], _points[:, 1], _points[:, 2],
            _orientations[:, 0], _orientations[:, 1], _orientations[:, 2],
            length=0.2, normalize=True)

    # Equal aspect ratio
    max_range = np.array([
        _points[:, 0].max() - _points[:, 0].min(),
        _points[:, 1].max() - _points[:, 1].min(),
        _points[:, 2].max() - _points[:, 2].min()
    ]).max() / 2.0

    mid_x = (_points[:, 0].max() + _points[:, 0].min()) * 0.5
    mid_y = (_points[:, 1].max() + _points[:, 1].min()) * 0.5
    mid_z = (_points[:, 2].max() + _points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

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

print(f'E_time={E_time:.6e}, E_freq={E_freq:.6e}')
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

# N = 200
# a, b = 2.0, 0.6  # spheroid axes
# points = []
# min_dist = 0.1
# Average_school_Depth = 50 # m

# # "Theta" is the angle of projected vector on XY plane and X axis
# theta_range = np.array([-0.01, 0.01])*np.pi/180

# # "Phi" is the angle of vector and Z axis
# phi_range = np.array([89.99, 90.01])*np.pi/180


# # I. Create N points(x, y, z) with min_dist to avoid overlap:
# while len(points) < N:
#     x = np.random.uniform(-a, a) 
#     y = np.random.uniform(-a, a)
#     z = np.random.uniform(-b, b) - Average_school_Depth
#     if (x**2/a**2 + y**2/a**2 + (z + Average_school_Depth)**2/b**2) <= 1:
#         p = np.array([x, y, z])
#         if all(np.linalg.norm(p - q) > min_dist for q in points):
#             points.append(p)

# points = np.array(points)
# # print(points)



# # II. Orientaion of fish - Random unit vectors: 
# # "Theta" is the angle of projected vector on XY plane and X axis
# # "Phi" is the angle of vector and Z axis

# # RANDOM distribution:   -----------------------------------------------
# # Generate arrays of theta and phi: 
# theta = np.random.uniform(theta_range[0], theta_range[1], size=len(points))
# phi = np.random.uniform(phi_range[0], phi_range[1], size=len(points))

# # # NORMAL distribution:   -----------------------------------------------
# # Parameters
# mu = np.mean(theta_range)       # mean
# sigma = 0.25*(theta_range[1]-theta_range[0])     # standard deviation

# theta = np.random.normal(mu, sigma, N)

# # Plot histogram
# count, bins, ignored = plt.hist(180*theta/np.pi, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
# plt.xlabel('$\\theta$',fontsize = 12)
# plt.show()

# # Parameters
# mu = np.mean(phi_range)       # mean
# sigma = 0.25*(phi_range[1]-phi_range[0])     # standard deviation

# phi = np.random.normal(mu, sigma, N)

# # Plot histogram
# count, bins, ignored = plt.hist(180*phi/np.pi, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
# plt.xlabel(r'$\phi$',fontsize = 12)
# plt.show()

# # Convert spherical to Cartesian coordinates
# x = np.sin(phi) * np.cos(theta)
# y = np.sin(phi) * np.sin(theta)
# z = np.cos(phi)

# orientations = np.stack((x, y, z), axis=1)  # shape (N, 3)
# orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]  # normalize

# print(orientations[0])


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#                       Test case for two fish:  
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
points = []
Average_school_Depth = 50 # m
p = np.array([0, 0, - Average_school_Depth])
points.append(p)

p = np.array([0, 0, - Average_school_Depth -  0.1])
points.append(p)

p = np.array([0, 0, - Average_school_Depth - 0.25])
points.append(p)

p = np.array([0, -0.25, - Average_school_Depth - 0.25])
points.append(p)

points = np.array(points)
# print(points)


# Orientation: ---------------------------------------------
PHI = np.array([90.0, 90.0, 90.0, 90.0]) * np.pi/180
THETA = np.array([0.0, 0.0, 0.0, 0.0]) * np.pi/180

# Convert spherical to Cartesian coordinates
x = np.sin(PHI) * np.cos(THETA)
y = np.sin(PHI) * np.sin(THETA)
z = np.cos(PHI)

orientations = np.stack((x, y, z), axis=1)  # shape (N, 3)
orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]  # normalize
# # ====================================================


# Plot the fish school of N fish
plot_fish_school(points, orientations)
plt.savefig("Plot_"+os.path.splitext(FEM_BEM_files[0])[0]+".png", dpi=300, bbox_inches='tight')  # PNG
plt.show() 

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# print('len(points): ', len(points))
p_far = 0
for ii in range(0, len(points)):
    point_ii = points[ii]
    orientations_ii = orientations[ii]

    Incident_Angle_ii = Get_Angle(Observation_point, point_ii, orientations_ii)

    print('Incident_Angle_ii: ', Incident_Angle_ii)

    Length = 0.0 # m.  This can be changed to include distribution of ranges

    # Find the file in database (modeled .csv files) closest to the size of fish at "point_ii" with "orientation_ii"
    target_dict = Find_closestfile_in_database(df_database, np.abs(point_ii[2]), Length, Incident_Angle_ii)
    
    L_fish = Length # m 
    [freq, TS_ii, f_bs_ii] = func_get_frq_TS_fbs_Dict(database_dir, target_dict, L_fish)
    print(freq.shape)
    Distance = np.linalg.norm(Observation_point-point_ii)
    print('Distance: ', Distance)
    p_far_ii = (f_bs_ii/(Distance*Distance)) * np.exp(-1j*2*np.pi*(1000*freq/1500)*(2*Distance))  # k = 2*pi*f/c
    p_far = p_far + p_far_ii
    # plt.plot(freq, 20*np.log10(np.abs(f_bs_ii)), color = [1, 0, 0], dashes = [3,2], linewidth = 2)


point_ii = points[0]
Distance = np.linalg.norm(Observation_point-point_ii)
Total_p_TS = 20*np.log10(np.abs(Distance*Distance * p_far))
plt.plot(freq, Total_p_TS, color = [0, 0, 0], dashes = [3,0], linewidth = 1, label = 'Superposition')

plt.plot(f_bem/1000, TS_bem, color=[1,0,0], dashes = [2,2], label = 'FEM-BEM')

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
plt.savefig("Multitarget_"+os.path.splitext(FEM_BEM_files[0])[0]+".png", dpi=300, bbox_inches='tight')  # PNG
plt.savefig("Multitarget_"+os.path.splitext(FEM_BEM_files[0])[0]+".jpg", dpi=300, bbox_inches='tight')  # JPG

plt.show()

# delta_hz = np.mean(freq[1:].values - freq[:-1].values) * 1000
# time, ifft = inverse_frequency_response(Total_p_TS.values, delta_hz)
# plt.plot(time[1:] * 1500 * 0.5, ifft[1:], linewidth = 1, label = 'iFFT')
# plt.xlabel('Distance [m]', fontsize=12)
# plt.legend()
# plt.xlim([0.0, 1.0])
# plt.show()
