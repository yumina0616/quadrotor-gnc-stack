import numpy as np
from control import pid_attitude as pid
from dynamics import quaternion as quat
from control.pid_attitude import AttitudeController

def test_attitude_error_is_zero_for_same_attitude():
    q = quat.normalize(np.array([1, 0.2, 0.3, 0.4]))
    assert np.allclose(pid.attitude_error(q, q), np.zeros(3))

def test_attitude_error_small_angle_approximation():
    roll, pitch, yaw = 0.05, -0.03, 0.02
    q_current = np.array([1.0,0,0,0])
    q_target = quat.from_euler_zyx(roll, pitch, yaw)

    error = pid.attitude_error(q_current, q_target)

    assert np.allclose(error, [roll, pitch, yaw], atol=1e-2)

def make_controller(integral_limit = 0.5):
    kp = np.array([2.0, 0.3, 4.0])
    ki = np.array([1.2, 2.3, 3.4])
    kd = np.array([0.5, 1.0, 2.2])
    return AttitudeController(kp, ki, kd, integral_limit=integral_limit)

def test_zero_error_gives_zero_torque():
    ctrl = make_controller()
    q = quat.normalize(np.array([1, 0.2, 0.3, 0.4]))
    torque = ctrl.compute_torque(q, q, np.zeros(3), dt=0.01)
    assert np.allclose(torque, np.array([0, 0, 0]))

def test_integral_accumulates_between_calls():
    ctrl = make_controller()
    q_current = quat.normalize(np.array([1, 0.2, 0.3, 0.4]))
    q_target = quat.normalize(np.array([1, 0.5, 0.7, 0.1]))
    omega = np.zeros(3)

    torque1 = ctrl.compute_torque(q_current, q_target, omega, dt=0.01)
    torque2 = ctrl.compute_torque(q_current, q_target, omega, dt=0.01)

    e = pid.attitude_error(q_current, q_target)
    expected_diff = ctrl.ki * e * 0.01
    assert np.allclose(torque2 - torque1, expected_diff)

def test_anti_windup_clamps_integral():
    ctrl = make_controller(integral_limit=0.05)
    q_current = quat.normalize(np.array([1, 0.2, 0.3, 0.4]))
    q_target = quat.normalize(np.array([1, 4,5,6]))

    for _ in range(1000):
        ctrl.compute_torque(q_current, q_target, omega=np.zeros(3), dt=0.01)

    assert np.all(np.abs(ctrl._integral) <= 0.05)

def test_reset_clears_integral():
    ctrl = make_controller(integral_limit=0.05)
    q_current = quat.normalize(np.array([1, 0.2, 0.3, 0.4]))
    q_target = quat.normalize(np.array([1, 0.5, 0.7, 0.1]))
    ctrl.compute_torque(q_current, q_target, omega=np.zeros(3), dt=0.01)

    ctrl.reset()
    
    assert np.allclose(ctrl._integral, np.array([0,0,0]))