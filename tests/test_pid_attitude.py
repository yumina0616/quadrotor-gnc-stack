import numpy as np
from control import pid_attitude as pid
from dynamics import quaternion as quat

def test_attitude_error_is_zero_for_same_attitude():
    q = quat.normalize(np.array([1, 0.2, 0.3, 0.4]))
    assert np.allclose(pid.attitude_error(q, q), np.zeros(3))

def test_attitude_error_small_angle_approximation():
    roll, pitch, yaw = 0.05, -0.03, 0.02
    q_current = np.array([1.0,0,0,0])
    q_target = quat.from_euler_zyx(roll, pitch, yaw)

    error = pid.attitude_error(q_current, q_target)

    assert np.allclose(error, [roll, pitch, yaw], atol=1e-2)