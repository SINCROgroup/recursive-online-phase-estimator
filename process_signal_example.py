import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import csv
from mpl_toolkits.mplot3d import Axes3D
from RecursiveOnlinePhaseEstimator import RecursiveOnlinePhaseEstimator

# Parameters to set
discarded_time = 5
listening_time = 8
min_duration_first_quasiperiod = 0
look_behind_pcent = 5
look_ahead_pcent = 15
time_const_lowpass_filter_phase = 0
time_const_lowpass_filter_estimand_pos = 0
time_step = 0.01
is_use_baseline = True
time_step_baseline = 0.01

# File path
file_path_estimand = r"example/data/Human_motion/spiral_mc_1.csv"





directory = os.path.dirname(file_path_estimand)
base_name = os.path.basename(file_path_estimand)
name_moto = re.sub(r'_[a-zA-Z]+_\d+\.csv$', '', base_name)
file_path_baseline = os.path.join(directory, f"{name_moto}_baseline.csv")
file_path_real_phase = os.path.join(directory, f"real_{base_name}")

# Column names
col_names_pos_estimand = ['TX.3', 'TY.3', 'TZ.3']
col_names_pos_baseline = ['x', 'y', 'z']
col_names_ref_frame_estimand_points = [['TX', 'TY', 'TZ'], ['TX.2', 'TY.2', 'TZ.2'], ['TX.1', 'TY.1', 'TZ.1']]
col_names_ref_frame_baseline_points = [['p1_x','p1_y','p1_z'],['p2_x','p2_y','p2_z'],['p3_x','p3_y','p3_z']]

# Helpers
def extract_points_from_df(df, col_names_points):
    points = []
    for i in range(len(col_names_points)):
        points.append(np.array(df[col_names_points[i]].dropna().iloc[0]))
    return points

# Load data
df_estimand = pd.read_csv(file_path_estimand, skiprows=[0, 1, 2] + list(range(4, 30)), low_memory=False)
df_estimand_pos = df_estimand[col_names_pos_estimand].copy()
df_estimand_pos.ffill(inplace=True)
estimand_pos_signal = np.array(df_estimand_pos)
time_signal = np.arange(0, time_step * len(df_estimand_pos), time_step)

if is_use_baseline:
    ref_frame_estimand_points = extract_points_from_df(df_estimand, col_names_ref_frame_estimand_points)
    df_baseline = pd.read_csv(file_path_baseline)
    baseline_pos_loop = np.array(df_baseline[col_names_pos_baseline])
    ref_frame_baseline_points = extract_points_from_df(df_baseline, col_names_ref_frame_baseline_points)
else:
    baseline_pos_loop = None
    ref_frame_estimand_points = []
    ref_frame_baseline_points = []

# Initialize ROPE estimator
n_dims_estimand_pos = estimand_pos_signal.shape[1]
phase_estimator = RecursiveOnlinePhaseEstimator(
    n_dims_estimand_pos=n_dims_estimand_pos,
    listening_time=listening_time,
    discarded_time=discarded_time,
    min_duration_first_pseudoperiod=min_duration_first_quasiperiod,
    look_behind_pcent=look_behind_pcent,
    look_ahead_pcent=look_ahead_pcent,
    time_const_lowpass_filter_pos=time_const_lowpass_filter_estimand_pos,
    time_const_lowpass_filter_phase=time_const_lowpass_filter_phase,
    is_use_baseline=is_use_baseline,
    baseline_pos_loop=baseline_pos_loop,
    time_step_baseline=time_step_baseline,
    ref_frame_estimand_points=ref_frame_estimand_points,
    ref_frame_baseline_points=ref_frame_baseline_points,
    is_use_elapsed_time=False,
)

# Estimate phase online
n_time_instants = estimand_pos_signal.shape[0]
phase_estimand_online = np.full(n_time_instants, None)
for i_t in range(n_time_instants - 1):
    phase_estimand_online[i_t] = phase_estimator.update_estimator(estimand_pos_signal[i_t, :], time_signal[i_t])
initial_phase_estimand_online  = phase_estimand_online [int((listening_time + discarded_time) / time_step) + 1]   # + 1 necessary because, e.g., if listening_time + discarded_time = 4, we want to start at time = idx = 5

# Benchmark phase
with open(file_path_real_phase, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    vector_real_period_index = [int(row[0]) for row in reader]
real_phase = np.zeros(vector_real_period_index[-1])
for i in range(len(vector_real_period_index) - 1):
    start = vector_real_period_index[i]
    end = vector_real_period_index[i + 1]
    real_phase[start:end] = np.linspace(0, 2 * np.pi, end - start, endpoint=False)
real_phase = np.unwrap(real_phase)
real_phase += initial_phase_estimand_online - real_phase[int((listening_time + discarded_time) / time_step) + 1]
real_phase = np.mod(real_phase, 2 * np.pi)

# Configure plot settings
fig = plt.figure(figsize=(10, 9))
plt.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman'})
cmap = plt.get_cmap("Dark2")
colors = cmap.colors

# Define time window for plotting
loop_index_to_plot = 4
n_loops_to_plot = 3
idx_start = np.argmax(time_signal >= phase_estimator.delimiter_time_instants[loop_index_to_plot - 1])
idx_end = np.argmax(time_signal > phase_estimator.delimiter_time_instants[loop_index_to_plot + n_loops_to_plot - 1])

# Compute phase estimation error
real_phase = np.array(real_phase, dtype=float)
estimated_phase = np.array(phase_estimand_online, dtype=float)
phase_error = np.abs(np.angle(np.exp(1j * (real_phase[idx_start:idx_end] - estimated_phase[idx_start:idx_end]))))


# --- Subplot 1: Real vs estimated phase ---
ax1 = fig.add_subplot(311)
ax1.plot(time_signal[idx_start:idx_end],
         real_phase[idx_start:idx_end],
         color=colors[4], linewidth=2, linestyle='--', label='Benchmark')
ax1.plot(time_signal[idx_start:idx_end],
         estimated_phase[idx_start:idx_end],
         color=colors[2], linewidth=2, linestyle='-', label='ROPE')
ax1.set_ylabel('Phase [rad]')
ax1.legend()
ax1.grid(True)

# --- Subplot 2: Phase error ---
ax2 = fig.add_subplot(312)
ax2.plot(time_signal[idx_start:idx_end],
         phase_error,
         color=colors[2], linewidth=2, linestyle='-', label='Phase error')
ax2.set_ylabel('Error [rad]')
ax2.legend()
ax2.grid(True)

# --- Subplot 3: 3D trajectory of selected loop ---
ax3 = fig.add_subplot(313, projection='3d')
idx_3d_start = np.argmax(time_signal >= phase_estimator.delimiter_time_instants[loop_index_to_plot - 1])
idx_3d_end = np.argmax(time_signal > phase_estimator.delimiter_time_instants[loop_index_to_plot])
trajectory = estimand_pos_signal[idx_3d_start:idx_3d_end]

ax3.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
         color="C0", linewidth=2, label=f'Trajectory - Loop {loop_index_to_plot}')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_zticklabels([])
ax3.legend()

# --- Final layout ---
plt.tight_layout()
plt.show()
