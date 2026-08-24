import numpy as np
from scenarios.attitude_stabilization import (
    simulate_attitude_stabilization,
    make_controller,
    apply_motor_mixing,
    params,
)
from control.pid_attitude import attitude_error
from dynamics import quaternion as quat
from scenarios.sensor_models import Gyroscope
from scenarios.disturbances import WindDisturbance

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

def test_zero_noise_gyroscope_matches_ideal_case():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    result_ideal = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01)
    result_gyro = simulate_attitude_stabilization(
        initial, duration=3.0, dt=0.01,
        gyroscope=Gyroscope(noise_std=0.0, bias_random_walk_std=0.0),
    )
    assert np.allclose(result_ideal["states"], result_gyro["states"])

def test_gyroscope_quaternion_stays_valid():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    gyro = Gyroscope(noise_std=0.01, bias_random_walk_std=0.001, rng=np.random.default_rng(1))
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01, gyroscope=gyro)
    q_result = [s[6:10] for s in result["states"]]
    norms = np.linalg.norm(q_result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_gyroscope_still_converges_within_duration():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    gyro = Gyroscope(noise_std=0.01, bias_random_walk_std=0.001, rng=np.random.default_rng(1))
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01, gyroscope=gyro)
    q_final = result["states"][-1][6:10]
    q_target = np.array([1.0, 0, 0, 0])
    err = attitude_error(q_final, q_target)
    assert np.linalg.norm(err) < np.radians(1.0)

def test_gyroscope_reproducible_with_same_seed():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    result_a = simulate_attitude_stabilization(
        initial, duration=1.0, dt=0.01,
        gyroscope=Gyroscope(0.01, 0.001, rng=np.random.default_rng(7)),
    )
    result_b = simulate_attitude_stabilization(
        initial, duration=1.0, dt=0.01,
        gyroscope=Gyroscope(0.01, 0.001, rng=np.random.default_rng(7)),
    )
    assert np.array_equal(result_a["states"], result_b["states"])

def test_zero_wind_matches_ideal_case():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    result_ideal = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01)
    result_wind = simulate_attitude_stabilization(
        initial, duration=3.0, dt=0.01,
        wind=WindDisturbance(steady_force=np.zeros(3), gust_std=0.0),
    )
    assert np.allclose(result_ideal["states"], result_wind["states"])

def test_wind_quaternion_stays_valid():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    wind = WindDisturbance(steady_force=np.array([0.5, 0, 0]), gust_std=0.1, rng=np.random.default_rng(1))
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01, wind=wind)
    q_result = [s[6:10] for s in result["states"]]
    norms = np.linalg.norm(q_result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_wind_reproducible_with_same_seed():
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    result_a = simulate_attitude_stabilization(
        initial, duration=1.0, dt=0.01,
        wind=WindDisturbance(np.array([0.5, 0, 0]), 0.1, rng=np.random.default_rng(7)),
    )
    result_b = simulate_attitude_stabilization(
        initial, duration=1.0, dt=0.01,
        wind=WindDisturbance(np.array([0.5, 0, 0]), 0.1, rng=np.random.default_rng(7)),
    )
    assert np.array_equal(result_a["states"], result_b["states"])

def test_attitude_still_converges_with_wind():
    """자세 방정식은 위치/속도/바람과 결합되지 않으므로, 바람이 있어도 자세 수렴은 그대로여야 한다."""
    initial = (np.radians(20), np.radians(-15), np.radians(30))
    wind = WindDisturbance(steady_force=np.array([0.5, 0, 0]), gust_std=0.1, rng=np.random.default_rng(3))
    result = simulate_attitude_stabilization(initial, duration=3.0, dt=0.01, wind=wind)
    q_final = result["states"][-1][6:10]
    q_target = np.array([1.0, 0, 0, 0])
    err = attitude_error(q_final, q_target)
    assert np.linalg.norm(err) < np.radians(1.0)

def test_steady_wind_causes_position_drift():
    initial = (0.0, 0.0, 0.0)
    result_ideal = simulate_attitude_stabilization(initial, duration=1.0, dt=0.01)
    result_wind = simulate_attitude_stabilization(
        initial, duration=1.0, dt=0.01,
        wind=WindDisturbance(steady_force=np.array([1.0, 0, 0]), gust_std=0.0),
    )
    pos_ideal_final = result_ideal["states"][-1][0:3]
    pos_wind_final = result_wind["states"][-1][0:3]
    assert not np.allclose(pos_ideal_final, pos_wind_final)