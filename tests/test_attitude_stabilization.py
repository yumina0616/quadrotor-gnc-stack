import numpy as np
from scenarios.attitude_stabilization import (
    simulate_attitude_stabilization,
    make_controller,
    apply_motor_mixing,
    params,
)
from control.pid_attitude import attitude_error
from dynamics import quaternion as quat

def test_attitude_converges_within_duration():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01)
    q_final = result["states"][-1][6:10]
    q_target = np.array([1.0, 0, 0, 0])
    err = attitude_error(q_final, q_target)
    assert np.linalg.norm(err) < np.radians(1.0)

def test_quaternion_stays_unit_norm_throughout():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01)
    q_result = [s[6:10] for s in result["states"]]
    norms = np.linalg.norm(q_result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_zero_initial_offset_stays_near_zero():
    initial = (0.0, 0.0, 0.0)
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01)
    q_target = np.array([1.0, 0, 0, 0])
    errors = [np.linalg.norm(attitude_error(s[6:10], q_target)) for s in result["states"]]
    assert np.allclose(errors, 0, atol=1e-3)

def test_motor_mixing_quaternion_stays_valid():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01, use_motor_mixing=True)
    q_result = [s[6:10] for s in result["states"]]
    norms = np.linalg.norm(q_result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_apply_motor_mixing_preserves_wrench_when_unsaturated():
    thrust = params.mass * params.gravity
    torque = np.zeros(3)
    achieved_thrust, achieved_torque = apply_motor_mixing(thrust, torque, params)
    assert np.isclose(achieved_thrust, thrust)
    assert np.allclose(achieved_torque, torque)

def test_apply_motor_mixing_clips_when_saturated():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    ctrl = make_controller()
    torque = ctrl.compute_torque(
        quat.from_euler_zyx(*initial), np.array([1.0, 0, 0, 0]), np.zeros(3), 0.01
    )
    thrust = params.mass * params.gravity
    achieved_thrust, achieved_torque = apply_motor_mixing(thrust, torque, params)
    assert not np.allclose(achieved_torque, torque)