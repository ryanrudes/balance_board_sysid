from calibration import estimate_orientation_vqf


import argparse
import warnings

from scipy.integrate import cumulative_trapezoid, cumulative_simpson, trapezoid, simpson
from numpy import ndarray

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

def check_extension(value: str, expected_extension: str):
    if not value.endswith(expected_extension):
        raise argparse.ArgumentTypeError(f"File must end in extension '{expected_extension}'")
    return value

GYRO_X_COL = " GyroX"
GYRO_Y_COL = " GyroY"
GYRO_Z_COL = " GyroZ"
ACCEL_X_COL = " AccelerationX"
ACCEL_Y_COL = " AccelerationY"
ACCEL_Z_COL = " AccelerationZ"
MAG_X_COL = " MagneticFieldX"
MAG_Y_COL = " MagneticFieldY"
MAG_Z_COL = " MagneticFieldZ"
TIME_COL = "SamplingTime"

parser = argparse.ArgumentParser()
parser.add_argument("--filepath",
                    type=lambda s: check_extension(s, ".csv"),
                    default="data/more_intermittent_disturbances.csv",
                    help="Data file (must be .csv)")

padding_group = parser.add_mutually_exclusive_group()

padding_group.add_argument("--padding-samples", "-pn",
                           type=int,
                           help="Number of samples to trim from beginning and end of data. Default: 100.")

padding_group.add_argument("--padding-time", "-pt",
                           type=float,
                           help="Time padding to trim from the beginning and end of data [s]. Overrides --padding-samples.")

parser.add_argument(
    "--advanced-model",
    action="store_true",
    help="Use extended regression model including yaw and bias terms."
)
parser.add_argument(
    "--friction-model",
    action="store_true",
    help="Include Coulomb-like friction terms (tanh of roll/pitch rates) in the regression."
)
parser.add_argument(
    "--geometry-model",
    action="store_true",
    help="Use yaw-aware elliptical rocker geometry model with principal stiffness/damping."
)

args = parser.parse_args()

def read_data(filepath: str) -> tuple[ndarray, ndarray]:
    """
    Reads gyro velocity and time data from a CSV file.
    
    Returns:
        data : (6, N) array of gyro velocities and accelerations [rad/s, m/s^2]
        time : (N,) time array [s]
    """
    data = pd.read_csv(filepath)
    
    wx = data[GYRO_X_COL].to_numpy()
    wy = data[GYRO_Y_COL].to_numpy()
    wz = data[GYRO_Z_COL].to_numpy()
    
    ax = data[ACCEL_X_COL].to_numpy()
    ay = data[ACCEL_Y_COL].to_numpy()
    az = data[ACCEL_Z_COL].to_numpy()
    
    mx = data[MAG_X_COL].to_numpy()
    my = data[MAG_Y_COL].to_numpy()
    mz = data[MAG_Z_COL].to_numpy()
    
    measurements = np.vstack((wx, wy, wz, ax, ay, az, mx, my, mz))
    
    time = data[TIME_COL].to_numpy()
    time = time - time[0]  # Normalize time to start at 0
    
    return measurements, time

def trim_data(data: ndarray, time: ndarray, padding_samples: int | None, padding_time: float | None) -> tuple[ndarray, ndarray]:
    """
    Trims padding time from start and end of data.
    
    Returns:
        data : trimmed data
        time : trimmed time
    """
    if padding_samples is not None:
        start_index = padding_samples
        end_index = -padding_samples
    else:
        start_index = np.searchsorted(time, padding_time, side="left")
        end_index = np.searchsorted(time, time[-1] - padding_time, side="right")
    
    data_trimmed = data[:, start_index:end_index]
    time_trimmed = time[start_index:end_index]

    # Re-normalize time to start at 0
    time_trimmed = time_trimmed - time_trimmed[0]
    
    return data_trimmed, time_trimmed

def shade_true_intervals(ax, flags, time, color="orange", alpha=0.2, y_min=None, y_max=None):
    """
    Shade x-intervals where flags[i] is True.
    Intervals are [i, i+1) for each integer i. Consecutive True values are merged.

    Parameters:
        ax: matplotlib Axes to draw on
        flags: sequence of booleans, index 0..N-1
        time: array of time values corresponding to flags
        color: fill color
        alpha: opacity of the shading
        y_min, y_max: optional y-range to limit the shading; if None, use ax.get_ylim()
    """
    n = len(flags)
    if n == 0:
        return

    # Determine vertical bounds for the shading
    if y_min is None or y_max is None:
        y_min, y_max = ax.get_ylim()

    # Find contiguous runs of True and shade them with axvspan
    i = 0
    while i < n:
        if flags[i]:
            j = i + 1
            while j < n and flags[j]:
                j += 1
            # Shade the merged interval [i, j)
            ax.axvspan(time[i], time[j - 1], ymin=0, ymax=1, facecolor=color, alpha=alpha)
            i = j
        else:
            i += 1

def find_true_intervals(rest_detected: ndarray, time: ndarray) -> list[tuple[int, int]]:
    true_indices = np.nonzero(rest_detected)[0]
    
    if true_indices.size == 0:
        return []
    
    # Find the indices where the difference between consecutive indices is greater than 1
    diff = np.diff(true_indices)
    # Find the positions in `true_indices` where a gap occurs (diff > 1)
    gap_positions = np.where(diff > 1)[0]

    # The starts of runs are the first true index and any index immediately after a gap
    starts = np.insert(true_indices[gap_positions + 1], 0, true_indices[0])

    # The ends of runs are any true_index at a gap position (end of run), plus the final true index
    ends = np.append(true_indices[gap_positions], true_indices[-1])
    
    # Return the list of tuples here
    intervals = list(zip(starts, ends))
    
    return intervals

def trim_intervals(intervals: list[tuple[int, int]], time: ndarray, conservativity: float) -> list[tuple[int, int]]:
    start_times = np.array([time[start] for start, end in intervals])
    end_times = np.array([time[end] for start, end in intervals])
    
    durations = end_times - start_times
    
    # Conservatively trim the middle 80% of each interval
    conservativity = 0.5
    trim_amounts = durations * (1 - conservativity) / 2
    
    intervals = [(np.searchsorted(time, start + trim, side="left"), np.searchsorted(time, end - trim, side="right"))
                for start, end, trim in zip(start_times, end_times, trim_amounts)]
        
    return intervals

def get_intervals_between(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    
    between_intervals = []
    
    # Start from the beginning of the data to the start of the first interval
    if intervals[0][0] > 0:
        between_intervals.append((0, intervals[0][0]))
    
    # Between each pair of intervals
    for i in range(len(intervals) - 1):
        end_current = intervals[i][1]
        start_next = intervals[i + 1][0]
        if start_next > end_current:
            between_intervals.append((end_current, start_next))
    
    # From the end of the last interval to the end of the data
    # (Assuming we know the total length; here we just leave it open-ended)
    
    return between_intervals

def extract_free_motion_intervals(intervals: list[tuple[int, int]], angvel: ndarray, time: ndarray, impact_margin: float) -> list[tuple[int, int]]:
    free_motion_intervals = []
    
    for start, end in intervals:
        # Find the impact peak inside the dynamic interval
        omega_mag = np.linalg.norm(angvel[start:end, :], axis=1)
        impact_rel = np.argmax(omega_mag)
        impact_idx = start + impact_rel

        # Add a small margin after the impact to be safe
        dt = np.median(np.diff(time))
        fs = 1.0 / dt
        margin_samples = int(impact_margin * fs)

        start_free = impact_idx + margin_samples
        end_free   = end

        if start_free < end_free:
            free_motion_intervals.append((start_free, end_free))
    
    return free_motion_intervals

def moving_average(x: ndarray, window: int = 11) -> ndarray:
    """Simple moving-average smoothing along axis 0.

    Parameters:
        x      : (N, d) array
        window : odd integer window length; if too large, it will be clamped.

    Returns:
        smoothed x with same shape.
    """
    if window <= 1 or x.shape[0] < 3:
        return x

    # Clamp window to a reasonable value and ensure it's odd
    window = min(window, x.shape[0] if x.shape[0] % 2 == 1 else x.shape[0] - 1)
    if window < 3:
        return x

    kernel = np.ones(window, dtype=float) / window
    # Apply convolution column-wise
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=x)

def project_psd(M: ndarray, eps: float = 0.0) -> ndarray:
    """Project a symmetric 2x2 matrix onto the positive semidefinite cone.

    Eigenvalues below `eps` are clamped up to `eps`.
    """
    # Ensure symmetry numerically
    M_sym = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M_sym)
    vals_clamped = np.maximum(vals, eps)
    return (vecs * vals_clamped) @ vecs.T

def build_segment_regression(
    angvel: ndarray,
    rpy: ndarray,
    time: ndarray,
    speed_threshold: float = 0.05,
    advanced: bool = False,
    friction: bool = False,
) -> tuple[ndarray, ndarray]:
    """Build regression data (F, Y) for a single free-motion segment (roll & pitch only, optionally advanced/friction).

    We model the small-angle upright dynamics for q = [roll, pitch] as

        ddot(q) + D_eff * dot(q) + K_eff * q = 0,

    which we rewrite as

        -ddot(q) = [dot(q), q] @ [D_row, K_row]^T

    for each row. This function returns the stacked regressors and targets for this
    single segment; multiple segments can be concatenated before solving.

    Parameters:
        angvel : (N, 3) array of angular velocities [rad/s]
        rpy    : (N, 3) array of roll, pitch, yaw angles [rad]
        time   : (N,) array of time [s] (monotonic within the segment)
        speed_threshold : drop samples where angular speed is below this to
                          avoid near-static, noise-dominated data.
        advanced : If True, include yaw and bias terms in the regression matrix.
        friction : If True, include smooth Coulomb-like friction terms based on
                   roll and pitch rates.

    Returns:
        F_seg : (M, n_features) array of regressors.
                The first four columns are always [roll_rate, pitch_rate, roll, pitch].
                If `friction` is True, two additional columns are added corresponding
                to tanh-regularized roll and pitch rate (Coulomb-like friction).
                If `advanced` is True, three additional columns [yaw_rate, yaw, 1]
                are appended.
        Y_seg : (M, 2) array with columns [-ddot(roll), -ddot(pitch)]
    """
    if angvel.shape[0] != rpy.shape[0] or angvel.shape[0] != time.shape[0]:
        raise ValueError("angvel, rpy, and time must have the same length")

    # Determine number of feature columns based on options
    n_base = 4  # [roll_rate, pitch_rate, roll, pitch]
    n_fric = 2 if friction else 0  # Coulomb-like terms for roll & pitch
    n_adv = 3 if advanced else 0   # [yaw_rate, yaw, bias]
    n_features = n_base + n_fric + n_adv

    if angvel.shape[0] < 5:
        # Too short to get a meaningful derivative
        return np.empty((0, n_features)), np.empty((0, 2))

    # Use only roll & pitch rates (assume small angles so body rates ≈ Euler rates)
    omega_rp = angvel[:, :2]  # shape (N, 2)
    q_rp = rpy[:, :2]         # shape (N, 2)

    # Drop very low-speed samples to avoid purely static points
    speed = np.linalg.norm(omega_rp, axis=1)
    mask_speed = speed > speed_threshold

    if np.count_nonzero(mask_speed) < 5:
        # Not enough excited samples to be useful
        return np.empty((0, n_features)), np.empty((0, 2))

    omega_rp = omega_rp[mask_speed]
    q_rp = q_rp[mask_speed]
    time_used = time[mask_speed]

    # For advanced model, also keep yaw and yaw rate (approx. from body rates)
    if advanced:
        yaw = rpy[:, 2][mask_speed]           # (N_used,)
        yaw_rate = angvel[:, 2][mask_speed]   # (N_used,)

    # Smooth the angular rates before differentiating to reduce noise
    omega_smooth = moving_average(omega_rp, window=11)

    # Estimate angular accelerations via numerical differentiation
    # qddot ≈ d/dt (omega_smooth)
    qddot = np.gradient(omega_smooth, time_used, axis=0)  # shape (N_used, 2)

    # Build regression matrix and target:
    #   -ddot(q) = [dot(q), q] @ [D_row, K_row]^T
    F_parts = [np.hstack((omega_smooth, q_rp))]  # base: (N_used, 4)

    if friction:
        # Smooth Coulomb-like friction terms using tanh of rate / eps
        eps = 0.05  # [rad/s], sets transition scale between linear and saturated
        fric_terms = np.tanh(omega_smooth / eps)  # (N_used, 2)
        F_parts.append(fric_terms)

    if advanced:
        bias = np.ones_like(yaw)[..., None]   # (N_used, 1)
        # Append yaw_rate, yaw, and bias
        F_parts.append(
            np.hstack((yaw_rate.reshape(-1, 1),
                       yaw.reshape(-1, 1),
                       bias))
        )

    F_seg = np.hstack(F_parts)
    Y_seg = -qddot                           # (N_used, 2)

    return F_seg, Y_seg

def system_identification(F: ndarray, Y: ndarray) -> tuple[ndarray, ndarray, dict[str, dict[str, float]]]:
    """Solve for 2-DOF effective stiffness and damping from global regression data,
    and compute basic fit quality metrics.

    Parameters:
        F : (N, n_features) stacked regressors. The first four columns must be
            [roll_rate, pitch_rate, roll, pitch]. Any additional columns (if present)
            are extra regressors (e.g., yaw, bias) used to improve the fit but not
            included in K_eff/D_eff.
        Y : (N, 2) stacked targets [-ddot(roll), -ddot(pitch)]

    Returns:
        K_eff : (2, 2) effective stiffness matrix (J^{-1} K) for roll & pitch
        D_eff : (2, 2) effective damping matrix (J^{-1} D) for roll & pitch
    """
    if F.shape[0] != Y.shape[0]:
        raise ValueError("F and Y must have the same number of rows")

    if F.shape[0] < 10:
        raise ValueError("Not enough samples in regression data for system identification")

    # Small ridge regularization to stabilize the fit
    lam = 1e-3
    n_features = F.shape[1]
    n_base = 4  # [roll_rate, pitch_rate, roll, pitch]
    if n_features < n_base:
        raise ValueError("F must have at least 4 columns: [roll_rate, pitch_rate, roll, pitch]")

    I = np.eye(n_features)

    # Pre-build the regularized design matrix and zero targets for the penalty
    F_reg = np.vstack((F, np.sqrt(lam) * I))
    zeros = np.zeros(n_features)

    D_eff = np.zeros((2, 2))
    K_eff = np.zeros((2, 2))
    extra_coeffs = np.zeros((2, max(0, n_features - n_base)))

    # Solve two independent least-squares problems, one per axis (roll, pitch)
    for i in range(2):
        y = Y[:, i]
        y_reg = np.concatenate((y, zeros))
        coeffs, *_ = np.linalg.lstsq(F_reg, y_reg, rcond=None)
        # First two coefficients multiply [roll_rate, pitch_rate] -> damping row
        # Next two coefficients multiply [roll, pitch] -> stiffness row
        D_eff[i, :] = coeffs[:2]
        K_eff[i, :] = coeffs[2:4]
        if n_features > n_base:
            extra_coeffs[i, :] = coeffs[n_base:]

    # Enforce symmetry as a reasonable physical assumption
    D_eff = 0.5 * (D_eff + D_eff.T)
    K_eff = 0.5 * (K_eff + K_eff.T)

    # Project damping (and stiffness) to be positive semidefinite for physical plausibility
    D_eff = project_psd(D_eff, eps=0.0)
    K_eff = project_psd(K_eff, eps=0.0)

    # Compute basic fit metrics (RMS error, relative RMS, R^2) for roll and pitch
    metrics: dict[str, dict[str, float]] = {}
    axis_names = ["roll", "pitch"]

    for i, name in enumerate(axis_names):
        # Use the symmetrized matrices to form the effective base coefficients
        base_coeffs = np.hstack((D_eff[i, :], K_eff[i, :]))  # length 4
        if extra_coeffs.shape[1] > 0:
            coeffs_sym_full = np.concatenate((base_coeffs, extra_coeffs[i, :]))
        else:
            coeffs_sym_full = base_coeffs

        y_true = Y[:, i]
        y_pred = F @ coeffs_sym_full
        residuals = y_true - y_pred

        mse = float(np.mean(residuals**2))
        rms = float(np.sqrt(mse))

        # Relative RMS: normalize by standard deviation of the target if non-zero
        var_y = float(np.var(y_true))
        rel_rms = float(rms / np.sqrt(var_y)) if var_y > 0.0 else float("nan")

        # Coefficient of determination R^2
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

        metrics[name] = {"rms": rms, "rel_rms": rel_rms, "r2": r2}

    return K_eff, D_eff, metrics

def geometry_system_identification(
    angvel: ndarray,
    rpy: ndarray,
    time: ndarray,
    free_intervals: list[tuple[int, int]],
    speed_threshold: float = 0.05,
    friction: bool = False,
) -> tuple[ndarray, ndarray, dict[str, float], dict[str, dict[str, float]]]:
    """Identify principal stiffness/damping for an elliptical rocker using a yaw-aware model.

    We assume the board has two principal rocker directions in the board frame with

        K_b = diag(k1, k2),  D_b = diag(d1, d2),

    and that in the world roll/pitch frame the stiffness and damping rotate with yaw psi:

        K_world(psi) = R(psi) K_b R(psi)^T
        D_world(psi) = R(psi) D_b R(psi)^T

    where R(psi) is the 2D rotation matrix built from yaw psi. This yields, in world
    coordinates,

        ddot(q) + D_world(psi) dot(q) + K_world(psi) q = 0

    with q = [roll, pitch]. We build a regression that is linear in the unknown
    principal parameters [d1, d2, k1, k2] (and optionally Coulomb-like friction
    coefficients) and solve it using ridge-regularized least squares.

    Parameters:
        angvel : (N, 3) array of angular velocities [rad/s]
        rpy    : (N, 3) array of roll, pitch, yaw angles [rad]
        time   : (N,) time array [s]
        free_intervals : list of (start, end) index pairs for free-motion segments
        speed_threshold : threshold on roll/pitch rate magnitude for including samples
        friction : if True, include smooth Coulomb-like friction terms.

    Returns:
        K_b : (2, 2) diagonal matrix of principal stiffnesses in the board frame
        D_b : (2, 2) diagonal matrix of principal damping in the board frame
        friction_params : dict with friction coefficients per axis (if friction=True)
        metrics : dict of fit quality metrics per axis (RMS, rel_RMS, R^2)
    """
    # Containers for global regression
    Z_list: list[ndarray] = []
    y_list: list[ndarray] = []
    axis_flags_list: list[ndarray] = []  # 0 for roll rows, 1 for pitch rows

    # Friction parameter count
    n_params = 4 + (2 if friction else 0)  # [d1, d2, k1, k2, (Cf_roll, Cf_pitch)]

    for start, end in free_intervals:
        ang_seg = angvel[start:end, :]
        rpy_seg = rpy[start:end, :]
        t_seg = time[start:end]

        if ang_seg.shape[0] < 5:
            continue

        omega_rp = ang_seg[:, :2]        # roll & pitch rates
        q_rp = rpy_seg[:, :2]           # roll & pitch angles
        yaw = rpy_seg[:, 2]             # yaw

        # Drop low-speed samples (in roll/pitch) to avoid noise-dominated points
        speed = np.linalg.norm(omega_rp, axis=1)
        mask_speed = speed > speed_threshold
        if np.count_nonzero(mask_speed) < 5:
            continue

        omega_rp = omega_rp[mask_speed]
        q_rp = q_rp[mask_speed]
        yaw = yaw[mask_speed]
        t_used = t_seg[mask_speed]

        # Smooth roll/pitch rates before differentiating
        omega_smooth = moving_average(omega_rp, window=11)

        # Approximate angular accelerations of roll & pitch
        qddot = np.gradient(omega_smooth, t_used, axis=0)  # (Ns, 2)

        x = q_rp[:, 0]
        y_ = q_rp[:, 1]
        xd = omega_smooth[:, 0]
        yd = omega_smooth[:, 1]
        xdd = qddot[:, 0]
        ydd = qddot[:, 1]

        c = np.cos(yaw)
        s = np.sin(yaw)

        # Precompute coefficients for d1, d2, k1, k2 for roll equation
        # xdd = -d1 (c^2 xd + c s yd) - d2 (s^2 xd - c s yd)
        #       -k1 (c^2 x + c s y)   - k2 (s^2 x - c s y)  - friction_terms
        a1_roll = c * c * xd + c * s * yd
        a2_roll = s * s * xd - c * s * yd
        b1_roll = c * c * x + c * s * y_
        b2_roll = s * s * x - c * s * y_

        # Precompute coefficients for pitch equation
        # ydd = -d1 (-c s xd + s^2 yd) - d2 (c s xd + c^2 yd)
        #       -k1 (-c s x + s^2 y)   - k2 (c s x + c^2 y) - friction_terms
        a1_pitch = -c * s * xd + s * s * yd
        a2_pitch =  c * s * xd + c * c * yd
        b1_pitch = -c * s * x + s * s * y_
        b2_pitch =  c * s * x + c * c * y_

        if friction:
            eps = 0.05  # transition scale for tanh
            fric_roll = np.tanh(xd / eps)
            fric_pitch = np.tanh(yd / eps)

            Z_roll_seg = np.column_stack(
                (a1_roll, a2_roll, b1_roll, b2_roll, fric_roll, np.zeros_like(fric_roll))
            )
            Z_pitch_seg = np.column_stack(
                (a1_pitch, a2_pitch, b1_pitch, b2_pitch, np.zeros_like(fric_pitch), fric_pitch)
            )
        else:
            Z_roll_seg = np.column_stack((a1_roll, a2_roll, b1_roll, b2_roll))
            Z_pitch_seg = np.column_stack((a1_pitch, a2_pitch, b1_pitch, b2_pitch))

        # Targets: -ddot(roll), -ddot(pitch)
        y_roll_seg = -xdd
        y_pitch_seg = -ydd

        # Stack roll and pitch rows for this segment
        Z_seg = np.vstack((Z_roll_seg, Z_pitch_seg))         # (2*Ns, n_params)
        y_seg = np.concatenate((y_roll_seg, y_pitch_seg))    # (2*Ns,)
        axis_flags_seg = np.concatenate(
            (np.zeros_like(y_roll_seg, dtype=int), np.ones_like(y_pitch_seg, dtype=int))
        )

        Z_list.append(Z_seg)
        y_list.append(y_seg)
        axis_flags_list.append(axis_flags_seg)

    if not Z_list:
        raise ValueError("No usable samples for geometry-aware system identification.")

    Z = np.vstack(Z_list)
    y = np.concatenate(y_list)
    axis_flags = np.concatenate(axis_flags_list)

    # Ridge-regularized least squares
    lam = 1e-3
    I = np.eye(n_params)
    Z_reg = np.vstack((Z, np.sqrt(lam) * I))
    y_reg = np.concatenate((y, np.zeros(n_params)))

    theta, *_ = np.linalg.lstsq(Z_reg, y_reg, rcond=None)

    # Raw principal parameters from the unconstrained fit
    d1_raw, d2_raw, k1_raw, k2_raw = [float(v) for v in theta[:4]]

    if friction:
        Cf_roll = float(theta[4])
        Cf_pitch = float(theta[5])
    else:
        Cf_roll = 0.0
        Cf_pitch = 0.0

    # Enforce physical plausibility:
    #   * Stiffnesses must be non-negative.
    #   * Damping should not be exactly zero in any principal direction.
    #     If a damping coefficient comes out negative or zero, we replace it
    #     with a small fraction of the mean positive damping (if any), or a
    #     small default floor if all are non-positive.
    k_vals_raw = np.array([k1_raw, k2_raw], dtype=float)
    k_vals = np.maximum(k_vals_raw, 0.0)

    d_vals_raw = np.array([d1_raw, d2_raw], dtype=float)
    if np.all(d_vals_raw <= 0.0):
        # Both principal damping coefficients were non-positive; fall back to a small default.
        d_floor = 0.01
        d_vals = np.full_like(d_vals_raw, d_floor)
    else:
        d_vals = d_vals_raw.copy()
        pos_mean = float(np.mean(d_vals_raw[d_vals_raw > 0.0]))
        # Replace non-positive entries with 10% of the mean positive damping.
        d_vals[d_vals <= 0.0] = 0.1 * pos_mean

    K_b = np.diag(k_vals)
    D_b = np.diag(d_vals)

    friction_params: dict[str, float] = {}
    if friction:
        friction_params = {"roll": Cf_roll, "pitch": Cf_pitch}

    # Compute fit metrics per axis using the full parameter vector
    y_pred = Z @ theta
    residuals = y - y_pred

    metrics: dict[str, dict[str, float]] = {}
    for axis_name, axis_idx in (("roll", 0), ("pitch", 1)):
        mask = axis_flags == axis_idx
        if not np.any(mask):
            metrics[axis_name] = {"rms": float("nan"), "rel_rms": float("nan"), "r2": float("nan")}
            continue

        y_true_axis = y[mask]
        y_pred_axis = y_pred[mask]
        res_axis = y_true_axis - y_pred_axis

        mse = float(np.mean(res_axis**2))
        rms = float(np.sqrt(mse))

        var_y = float(np.var(y_true_axis))
        rel_rms = float(rms / np.sqrt(var_y)) if var_y > 0.0 else float("nan")

        ss_res = float(np.sum(res_axis**2))
        ss_tot = float(np.sum((y_true_axis - np.mean(y_true_axis))**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

        metrics[axis_name] = {"rms": rms, "rel_rms": rel_rms, "r2": r2}

    return K_b, D_b, friction_params, metrics

def plot(roll, pitch, yaw, time, rest_detected):
    fig, ax = plt.subplots(figsize=(10, 6))

    shade_true_intervals(ax, rest_detected, time, color="orange", alpha=0.3, y_min=min(roll.min(), pitch.min(), yaw.min()), y_max=max(roll.max(), pitch.max(), yaw.max()))

    # Shade the trimmed intervals in a darker color
    intervals = find_true_intervals(rest_detected, time)
    intervals = trim_intervals(intervals, time, conservativity=0.5)
    for start, end in intervals:
        print("Roll: %.5f (%.5f), Pitch: %.5f (%.5f), Yaw: %.5f (%.5f)" % (roll[start:end].mean(), roll[start:end].std(), pitch[start:end].mean(), pitch[start:end].std(), yaw[start:end].mean(), yaw[start:end].std()))
        ax.axvspan(time[start], time[end], ymin=0, ymax=1, facecolor="red", alpha=0.6)
        
    ax.plot(time, roll, "--", color="red", label="Gyro X angle (roll)")
    ax.plot(time, pitch, "--", color="green", label="Gyro Y angle (pitch)")
    #ax.plot(time, yaw, "--", color="blue", label="Gyro Z angle (yaw)")

    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")

    plt.legend()

    plt.xlim(0, time[-1])
    plt.ylim(-np.pi / 32, np.pi / 32)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title("IMU Data")
    plt.show()
    
data, time = read_data(args.filepath)

should_trim = (args.padding_samples is not None and args.padding_samples > 0) or \
              (args.padding_time is not None and args.padding_time > 0.0)

if should_trim:
    data, time = trim_data(data, time, args.padding_samples, args.padding_time)
else:
    warnings.warn("No padding specified; using full data.")
    
wx, wy, wz, ax, ay, az, mx, my, mz = data
quat, rpy, angvel, rest_detected = estimate_orientation_vqf(wx, wy, wz, ax, ay, az, mx, my, mz, time)

static_intervals = find_true_intervals(rest_detected, time)
dynamic_intervals = get_intervals_between(static_intervals)
free_intervals = extract_free_motion_intervals(dynamic_intervals, angvel, time, impact_margin=0.2)

plot(rpy[:,0], rpy[:,1], rpy[:,2], time, rest_detected)

# Reject time intervals in which the max angular velocity norm is below threshold
free_intervals = [interval for interval in free_intervals if np.max(np.linalg.norm(angvel[interval[0]:interval[1]], axis=1)) > 0.15]

# Plot each free-motion interval
for start, end in free_intervals:
    print(f"Free-motion interval: {time[start]:.3f}s to {time[end]:.3f}s ({end - start} samples)")
    plot(rpy[start:end,0], rpy[start:end,1], rpy[start:end,2], time[start:end] - time[start], rest_detected[start:end])

# Identification logic: branch on --geometry-model
if args.geometry_model:
    try:        
        K_b, D_b, friction_params, metrics = geometry_system_identification(
            angvel,
            rpy,
            time,
            free_intervals,
            speed_threshold=0.05,
            friction=args.friction_model,
        )
        print(f"Using geometry-aware model with {len(free_intervals)} free-motion segments.")
        print("Estimated principal K (board frame):\n", K_b)
        print("Estimated principal D (board frame):\n", D_b)
        if friction_params:
            print("Estimated friction coefficients:", friction_params)
        print("Fit quality (per axis):")
        for axis, m in metrics.items():
            print(
                f"  {axis:5s}: RMS = {m['rms']:.4e}, "
                f"rel_RMS = {m['rel_rms']:.3f}, R^2 = {m['r2']:.3f}"
            )
    except Exception as e:
        warnings.warn(f"Geometry-aware identification failed: {e}")
else:
    # Collect regression data across all free-motion segments (original method)
    F_list: list[ndarray] = []
    Y_list: list[ndarray] = []

    for start, end in free_intervals:
        # Orientation angles for possible inspection/plotting
        roll = rpy[start:end, 0]
        pitch = rpy[start:end, 1]
        yaw = rpy[start:end, 2]
        time_segment = time[start:end] - time[start]
        
        #plot(roll, pitch, yaw, time_segment, rest_detected[start:end])

        # Build regression for this segment (roll & pitch only, optionally advanced/friction)
        F_seg, Y_seg = build_segment_regression(
            angvel[start:end, :],
            rpy[start:end, :],
            time_segment,
            advanced=args.advanced_model,
            friction=args.friction_model,
        )

        if F_seg.size == 0 or Y_seg.size == 0:
            continue

        F_list.append(F_seg)
        Y_list.append(Y_seg)

    if F_list:
        F_global = np.vstack(F_list)
        Y_global = np.vstack(Y_list)

        print(f"Using {F_global.shape[0]} samples from {len(F_list)} free-motion segments for identification.")

        K, D, metrics = system_identification(F_global, Y_global)
        print("Estimated K (effective stiffness, roll & pitch):\n", K)
        print("Estimated D (effective damping, roll & pitch):\n", D)
        print("Fit quality (per axis):")
        for axis, m in metrics.items():
            print(
                f"  {axis:5s}: RMS = {m['rms']:.4e}, "
                f"rel_RMS = {m['rel_rms']:.3f}, R^2 = {m['r2']:.3f}"
            )
    else:
        warnings.warn("No usable free-motion samples for system identification.")