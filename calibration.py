import numpy as np
from vqf import offlineVQF  # pip install vqf


def quat_to_euler_zyx(quat):
    """
    Convert quaternions [w, x, y, z] to ZYX Euler angles:
    roll (x), pitch (y), yaw (z) in radians.
    Returns (N, 3) -> [roll, pitch, yaw].
    """
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp_clamped = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp_clamped)

    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack([roll, pitch, yaw])


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def estimate_orientation_vqf(
    wx, wy, wz,
    ax, ay, az,
    mx, my, mz,
    t,
    use_magnetometer=True,
    zero_initial=True,
):
    """
    Estimate orientation from IMU data using the VQF offline filter.

    Parameters
    ----------
    wx, wy, wz : 1D arrays
        Gyroscope [rad/s] in body frame.
    ax, ay, az : 1D arrays
        Accelerometer [m/s^2] in body frame.
    mx, my, mz : 1D arrays
        Magnetometer [arbitrary units] in body frame.
        Ignored if use_magnetometer=False (or all-NaN).
    t : 1D array
        Timestamps [s], roughly uniform.

    use_magnetometer : bool
        If True and mag is valid, uses full 9D solution; otherwise 6D.
    zero_initial : bool
        If True, re-reference orientation so the first sample has
        roll = pitch = yaw = 0 (i.e. all quaternions are made
        relative to the initial orientation).

    Returns
    -------
    quat : (N, 4) ndarray
        Quaternions [w, x, y, z]. If zero_initial=True, these are
        already made relative to the initial orientation.
    rpy : (N, 3) ndarray
        Euler angles [roll, pitch, yaw] in radians, matching `quat`.
    """

    # ---- Ensure numpy arrays ----
    wx = np.asarray(wx, dtype=float)
    wy = np.asarray(wy, dtype=float)
    wz = np.asarray(wz, dtype=float)
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)
    mx = np.asarray(mx, dtype=float)
    my = np.asarray(my, dtype=float)
    mz = np.asarray(mz, dtype=float)
    t  = np.asarray(t,  dtype=float)

    n = wx.size
    if not (wy.size == wz.size == ax.size == ay.size == az.size ==
            mx.size == my.size == mz.size == t.size == n):
        raise ValueError("All input arrays must have the same length.")

    # Stack into (N, 3) arrays as expected by VQF
    gyr = np.column_stack([wx, wy, wz])
    acc = np.column_stack([ax, ay, az])

    # Decide if we actually use mag
    if use_magnetometer and np.isfinite(mx).any():
        mag = np.column_stack([mx, my, mz])
    else:
        mag = None

    # Sampling time Ts
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Timestamps t must be strictly increasing and distinct.")
    Ts = float(np.median(dt))  # seconds

    # Call offlineVQF
    out = offlineVQF(gyr, acc, mag, Ts, params=None)

    # Pick quaternion: 9D if available (with mag), otherwise 6D (acc+gyro)
    if mag is not None and "quat9D" in out:
        quat = np.array(out["quat9D"], dtype=float)
    else:
        quat = np.array(out["quat6D"], dtype=float)

    # Normalize to be safe
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    
    # gyro bias (rad/s), same frame as input gyro
    bias = np.array(out["bias"], dtype=float)  # shape (N,3)

    # bias-corrected angular velocity
    omega_corr = gyr - bias  # still in body frame, rad/s

    # Optionally re-reference to initial orientation
    if zero_initial:
        q0 = quat[0]
        # inverse of unit quaternion = conjugate
        q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        quat_rel = np.array([quat_multiply(q0_inv, qi) for qi in quat])
        quat = quat_rel

    # Euler angles
    rpy = quat_to_euler_zyx(quat)

    return quat, rpy, omega_corr, out["restDetected"]