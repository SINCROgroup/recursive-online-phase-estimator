import warnings
import numpy as np
from scipy.signal import detrend, find_peaks
import matplotlib.pyplot as plt
from LowPassFilters import LowPassFilterPhase


##################################################
# Estimator class
##################################################

class OnlineMultidimPhaseEstimator:
    def __init__(self,
                 n_dims_estimand_pos: int,
                 listening_time,
                 discarded_time                 = 0,
                 min_duration_first_quasiperiod = 1,
                 look_behind_pcent              = 0,
                 look_ahead_pcent               = 10,
                 time_constant_lowpass_filter   = None,
                 is_use_baseline                = False,
                 baseline_pos_loop              = None,
                 time_step_baseline             = 0.01,
                 ref_frame_point_1              = None,
                 ref_frame_point_2              = None,
                 ref_frame_point_3              = None):

        self.is_first_loop_estimated = False
        assert look_ahead_pcent + look_behind_pcent <= 100, "look_ahead_pcent + look_behind_pcent must not exceed 100"

        # Initialization from arguments
        self.n_dims                       = n_dims_estimand_pos
        self.discarded_time               = discarded_time       # [s] discarded at the beginning before estimation
        self.listening_time               = listening_time       # [s] waits this time before estimating first loop must contain 2 quasiperiods
        self.look_ahead_pcent             = look_ahead_pcent     # % of last completed loop before last nearest point on which estimate the new phase
        self.look_behind_pcent            = look_behind_pcent    # % of last completed loop after last nearest point on which estimate the new phase
        self.time_constant_lowpass_filter = time_constant_lowpass_filter
        self.is_use_baseline              = is_use_baseline
        self.min_duration_quasiperiod     = min_duration_first_quasiperiod # [s]
        self.time_step_baseline           = time_step_baseline

        if is_use_baseline:
            assert n_dims_estimand_pos == 3,      "Tethered mode can be used only with n_dim = 3"
            assert baseline_pos_loop is not None, "Tethered mode was required but baseline_pos_loop was not provided"
            assert ref_frame_point_1 is not None, "Tethered mode was required but ref_frame_point_1 was not provided"
            assert ref_frame_point_2 is not None, "Tethered mode was required but ref_frame_point_2 was not provided"
            assert ref_frame_point_3 is not None, "Tethered mode was required but ref_frame_point_3 was not provided"

            self.baseline_pos_loop = baseline_pos_loop.copy()
            self.ref_frame_point_1 = ref_frame_point_1.copy()
            self.ref_frame_point_2 = ref_frame_point_2.copy()
            self.ref_frame_point_3 = ref_frame_point_3.copy()

        # Attributes not tunable by caller
        self.phase_jump_for_loop_detection = np.pi

        # Initial values
        self.active_mode             = "sleeping"
        self.phase_offset            = 0

        self.pos_signal              = []
        self.vel_signal              = []
        self.local_time_signal       = []
        self.delimiter_time_instants = []
        self.local_phase_signal      = []  # local:  without offset
        self.global_phase_signal     = []  # global: with offset
        self.new_loop                = []
        self.idx_curr_phase_in_latest_loop = 0


    def update_pos_vel(self, curr_pos) -> None:
        self.pos_signal.append(curr_pos)
        if len(self.pos_signal) >= 2:
            curr_step_time = self.local_time_signal[-1] - self.local_time_signal[-2]
            self.vel_signal.append((self.pos_signal[-1] - self.pos_signal[-2]) / curr_step_time)
        else:                          self.vel_signal.append(np.zeros(self.n_dims))

    def get_kinematics(self, idx:int) -> np.ndarray:
        return np.concatenate((self.pos_signal[idx], self.vel_signal[idx]))


    def update_look_ranges(self) -> None:
        self.look_ahead_range  = int(len(self.latest_pos_loop) * self.look_ahead_pcent  / 100)
        self.look_behind_range = int(len(self.latest_pos_loop) * self.look_behind_pcent / 100)


    # This is the main cycle perfomerd at each time instant
    def update_estimator(self, curr_pos, curr_time) -> float:

        if not self.local_time_signal:  self.initial_time = curr_time  # initialize initial_time
        self.local_time_signal.append(curr_time - self.initial_time)

        # Update active mode
        if self.active_mode == "sleeping" and self.local_time_signal[-1] >= self.discarded_time:
            self.active_mode = "listening"
            self.idx_time_start_listening = len(self.local_time_signal)-1
        if self.active_mode == "listening" and self.local_time_signal[-1] > self.discarded_time + self.listening_time:
            self.active_mode = "estimating"
            # self.idx_time_start_estimating = len(self.local_time_signal)
 
        # Update estimator according to active mode
        match self.active_mode:
            case "sleeping":
                return None

            case "listening":
                self.update_pos_vel(curr_pos)
                return None

            case "estimating":
                self.update_pos_vel(curr_pos)

                if not self.is_first_loop_estimated:
                    self.latest_pos_loop = compute_loop_with_autocorrelation(
                        pos_signal=self.pos_signal,
                        vel_signal=self.vel_signal,
                        local_time_vec=self.local_time_signal,
                        min_duration_quasiperiod=self.min_duration_quasiperiod)
                    self.update_look_ranges()

                    self.delimiter_time_instants.append(float(self.local_time_signal[self.idx_time_start_listening] + self.initial_time))
                    self.delimiter_time_instants.append(float(self.local_time_signal[len(self.latest_pos_loop) + self.idx_time_start_listening] + self.initial_time))
                    self.is_first_loop_estimated = True

                    if self.is_use_baseline:  self.compute_phase_offset()
    
                    # Compute phases in first loop
                    local_phases_first_loop  = np.linspace(0, 2 * np.pi, len(self.latest_pos_loop))
                    global_phases_first_loop = np.mod(local_phases_first_loop + self.phase_offset, 2 * np.pi)
                    self.local_phase_signal  = self.local_phase_signal + local_phases_first_loop.tolist()
                    self.global_phase_signal = self.global_phase_signal + (global_phases_first_loop + self.phase_offset).tolist()

                    # Estimate phases between first loop and current time
                    idx_first_instant_of_second_loop = len(self.latest_pos_loop)
                    idx_prev_instant = len(self.pos_signal) - 2
                    self.lowpass_filter_phase = LowPassFilterPhase(init_state=self.local_phase_signal[-1],
                                                                 time_step=self.local_time_signal[idx_first_instant_of_second_loop] - self.local_time_signal[idx_first_instant_of_second_loop-1],
                                                                 time_const=self.time_constant_lowpass_filter,
                                                                 wrap_interv="0_to_2pi")
                    for idx_scanning in range(idx_first_instant_of_second_loop, idx_prev_instant+1):   # recall: range excludes end point
                        curr_time_step = self.local_time_signal[idx_scanning] - self.local_time_signal[idx_scanning-1]
                        self.compute_phase(self.get_kinematics(idx_scanning), curr_time_step)
                        if idx_scanning > idx_first_instant_of_second_loop:  # at first instant of second loop don't check, because new loop is already loaded and besides new_loop is empty
                            self.possibly_update_latest_loop(self.local_time_signal[idx_scanning + self.idx_time_start_listening] + self.initial_time)
                        self.new_loop.append(self.get_kinematics(idx_scanning))

                # Estimate phase at current time
                curr_time_step = self.local_time_signal[-1] - self.local_time_signal[-2]
                self.compute_phase(self.get_kinematics(-1), curr_time_step)
                self.possibly_update_latest_loop(curr_time)
                self.new_loop.append(self.get_kinematics(-1))
                return self.global_phase_signal[-1]


    def possibly_update_latest_loop(self, curr_time):
            if self.local_phase_signal[-1] - self.local_phase_signal[-2] < - self.phase_jump_for_loop_detection:   # a quasiperiodicity window ended
                self.delimiter_time_instants.append(float(curr_time))
                self.latest_pos_loop = np.vstack(self.new_loop)
                self.update_look_ranges()
                self.new_loop = []         # reinitialize new_loop
                self.idx_curr_phase_in_latest_loop = 0


    def compute_phase(self, curr_kinematics, curr_time_step):
        len_latest_loop = len(self.latest_pos_loop)
        if self.idx_curr_phase_in_latest_loop + self.look_ahead_range <= len_latest_loop:
            #   loop: [- - - - - - single part - - - - -]
            idxs_loop_for_search = np.arange( max(0, self.idx_curr_phase_in_latest_loop - self.look_behind_range),
                                              self.idx_curr_phase_in_latest_loop + self.look_ahead_range)
            loop_for_search = self.latest_pos_loop[idxs_loop_for_search]
        else:
            #   loop: [part_2 - - - - - - - - part_1]
            idxs_end_part_2 = self.idx_curr_phase_in_latest_loop + self.look_ahead_range - len_latest_loop
            idxs_part_1 = np.arange(max(idxs_end_part_2, self.idx_curr_phase_in_latest_loop - self.look_behind_range), len_latest_loop)
            idxs_part_2 = np.arange(0, idxs_end_part_2)
            idxs_loop_for_search = np.concatenate((idxs_part_1, idxs_part_2))
            loop_for_search = self.latest_pos_loop[idxs_loop_for_search]

        idx_min_distance = compute_idx_min_distance(pos_signal = loop_for_search[:, 0:self.n_dims].copy(),
                                                    vel_signal = loop_for_search[:, self.n_dims:].copy(),
                                                    curr_pos   = curr_kinematics[0:self.n_dims],
                                                    curr_vel   = curr_kinematics[self.n_dims:])

        self.idx_curr_phase_in_latest_loop = idxs_loop_for_search[idx_min_distance]
        self.local_phase_signal.append((2 * np.pi * self.idx_curr_phase_in_latest_loop) / len_latest_loop)
        if not self.time_constant_lowpass_filter in {None, 0, -1}:
            self.lowpass_filter_phase.change_time_step(curr_time_step)
            self.local_phase_signal[-1] = self.lowpass_filter_phase.update_state(self.local_phase_signal[-1])
        self.global_phase_signal.append(np.mod(self.local_phase_signal[-1] + self.phase_offset, 2 * np.pi))

    
    def compute_phase_offset(self) -> None:
        x_axis = (self.ref_frame_point_2 - self.ref_frame_point_1) / np.linalg.norm(self.ref_frame_point_2 - self.ref_frame_point_1) # x-axix directed from p1 to p2
        z_vector = self.ref_frame_point_3 - self.ref_frame_point_2 
        z_axis = z_vector - np.dot(z_vector, x_axis) * x_axis # z-axix defined by orthogonalizing the vector from p2 to p3 with respect to the x-axis
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis) # y-axix obtained via the cross product of z and x (x,y,z are mutually perpendicular and normalized)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis])   # TODO check 
        
        rotated_loop = self.latest_pos_loop[:, 0:3] @ rotation_matrix.T # it converts the coordinates from the local frame to the global one.

        centroid = np.mean(rotated_loop, axis=0)
        rotated_centered_loop = rotated_loop - centroid

        scale_factors = np.std(self.baseline_pos_loop, axis=0) / np.std(rotated_centered_loop, axis=0)
        scale_factors[np.isnan(scale_factors)] = 1 
        scaled_rotated_centered_loop = rotated_centered_loop * scale_factors
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
       
        ax.plot(self.baseline_pos_loop[:,0], self.baseline_pos_loop[:,1], self.baseline_pos_loop[:,2], label="baseline")
        ax.plot(scaled_rotated_centered_loop[:,0], scaled_rotated_centered_loop[:,1], scaled_rotated_centered_loop[:,2], label="period")
        ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
        plt.legend()
        plt.show()
        curr_pos_ = scaled_rotated_centered_loop[0, :]
        curr_step_time = self.local_time_signal[-1] - self.local_time_signal[-2]
        curr_vel_ = (scaled_rotated_centered_loop[1, :] - scaled_rotated_centered_loop[0, :]) / curr_step_time
        baseline_vel_loop = np.gradient(self.baseline_pos_loop, np.arange(0, len(self.baseline_pos_loop) * self.time_step_baseline, self.time_step_baseline), axis=0)
        index = compute_idx_min_distance(pos_signal = self.baseline_pos_loop.copy(),
                                         vel_signal = baseline_vel_loop.copy(),
                                         curr_pos   = curr_pos_,
                                         curr_vel   = curr_vel_)
        self.phase_offset = (2 * np.pi * index) / len(self.baseline_pos_loop)


##################################################
# Helper functions
##################################################

threshold_acceptable_peaks_wrt_maximum_pcent = 20  # Acceptance range of autocorrelation peaks defined as a percentage of the maximum autocorrelation value.


def compute_loop_with_autocorrelation(pos_signal, vel_signal, local_time_vec, min_duration_quasiperiod) -> np.ndarray:
    n_dim = len(pos_signal[0])
    pos_signal_stacked = np.vstack(pos_signal)  # time flows vertically
    vel_signal_stacked = np.vstack(vel_signal)

    pos_signal_unstacked = np.full(n_dim, None)
    autocorr_vecs_per_dim = np.full(n_dim, None)
    for i in range(n_dim):
        pos_signal_unstacked[i] = pos_signal_stacked[:, i]
        autocorr_vecs_per_dim[i] = compute_autocorr_vec(detrend(pos_signal_unstacked[i]))
    autocorr_vec_tot = np.sum(np.array(autocorr_vecs_per_dim), axis=0)

    idx_min_length = np.argmax(np.array(local_time_vec) > min_duration_quasiperiod)  # argmax finds the first True
    autocorr_vec_tot[:idx_min_length] = autocorr_vec_tot[idx_min_length]

    autocorr_vec_tot = autocorr_vec_tot / np.max(np.abs(autocorr_vec_tot))

    peaks, _ = find_peaks(autocorr_vec_tot)
    peaks_values = autocorr_vec_tot[peaks[np.where(peaks > idx_min_length)]]
    assert max(peaks_values) > 0, "No valid first loop was found, try increasing the listening time."
    lower_bound_acceptable_peaks_values = max(peaks_values) - (max(peaks_values) * threshold_acceptable_peaks_wrt_maximum_pcent / 100)
    idxs_possible_period = np.where(peaks_values > lower_bound_acceptable_peaks_values)[0]  # indexing necessary because np.where returns a tuple containing an array
    idx_start_loop = 0
    idx_end_loop = peaks[idxs_possible_period][0]

    pos_loop = pos_signal_stacked[idx_start_loop:idx_end_loop, :]
    vel_loop = vel_signal_stacked[idx_start_loop:idx_end_loop, :]
    return np.column_stack((pos_loop, vel_loop))


def compute_autocorr_vec(signal: np.ndarray) -> np.ndarray:
    max_lag = len(signal) // 2
    autocorr_vec = np.zeros(max_lag + 1)

    for lag in range(1, len(autocorr_vec)):
        var = np.var(signal[0:2 * lag])
        if var == 0:
            autocorr_vec[lag] = 0
        else:
            mean = np.mean(signal[0:2 * lag])
            sum_ = 0
            for t in range(lag):
                sum_ += (signal[t] - mean) * (signal[t + lag] - mean)
            autocorr_vec[lag] = sum_ / (lag * var)
    return autocorr_vec


def compute_idx_min_distance(pos_signal, vel_signal, curr_pos, curr_vel) -> int:
    distances_pos = np.sqrt(np.sum((pos_signal - curr_pos) ** 2, axis=1))
    distances_vel = np.sqrt(np.sum((vel_signal - curr_vel) ** 2, axis=1))
    distances_pos = distances_pos / max(distances_pos, default=1)  # avoids dividing by zero
    distances_vel = distances_vel / max(distances_vel, default=1)
    return np.argmin(distances_pos + distances_vel)