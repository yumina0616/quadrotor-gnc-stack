import numpy as np
from scenarios.sensor_models import Gyroscope, Accelerometer, true_specific_force
from config.vehicle_params import DEFAULT_PARAMS as params

def test_measure_reproducible_with_same_seed():
    true_omega = np.array([0.1, -0.2, 0.05])
    g1 = Gyroscope(noise_std=0.01, bias_random_walk_std=0.001, rng=np.random.default_rng(1))
    g2 = Gyroscope(noise_std=0.01, bias_random_walk_std=0.001, rng=np.random.default_rng(1))
    for _ in range(10):
        m1 = g1.measure(true_omega, dt=0.01)
        m2 = g2.measure(true_omega, dt=0.01)
        assert np.array_equal(m1, m2)

def test_zero_noise_and_bias_returns_true_omega_exactly():
    true_omega = np.array([0.3, -0.1, 0.2])
    g = Gyroscope(noise_std=0.0, bias_random_walk_std=0.0, rng=np.random.default_rng(0))
    for _ in range(20):
        measured = g.measure(true_omega, dt=0.01)
        assert np.array_equal(measured, true_omega)

def test_bias_stays_zero_without_random_walk():
    true_omega = np.zeros(3)
    g = Gyroscope(noise_std=0.05, bias_random_walk_std=0.0, rng=np.random.default_rng(3))
    for _ in range(50):
        g.measure(true_omega, dt=0.01)
    assert np.array_equal(g.bias, np.zeros(3))

def test_bias_drift_variance_independent_of_dt():
    """duration이 같으면 dt를 잘게 쪼개도 bias 드리프트 표준편차가 거의 같아야 한다."""
    bias_random_walk_std = 0.05
    duration = 1.0
    n_trials = 300

    def final_bias_std(dt, seed_offset):
        finals = []
        n_steps = int(duration / dt)
        for i in range(n_trials):
            g = Gyroscope(
                noise_std=0.0,
                bias_random_walk_std=bias_random_walk_std,
                rng=np.random.default_rng(seed_offset + i),
            )
            for _ in range(n_steps):
                g.measure(np.zeros(3), dt=dt)
            finals.append(g.bias[0])
        return np.std(finals)

    std_coarse = final_bias_std(dt=0.01, seed_offset=0)
    std_fine = final_bias_std(dt=0.001, seed_offset=10_000)

    expected = bias_random_walk_std * np.sqrt(duration)
    assert np.isclose(std_coarse, expected, rtol=0.3)
    assert np.isclose(std_fine, expected, rtol=0.3)

def test_true_specific_force_at_hover_equals_gravity_up():
    hover_thrust = params.mass * params.gravity
    f = true_specific_force(hover_thrust, params)
    assert np.allclose(f, [0, 0, -params.gravity])

def test_true_specific_force_scales_with_thrust():
    thrust = 12.0
    f = true_specific_force(thrust, params)
    assert np.allclose(f, [0, 0, -thrust / params.mass])

def test_accelerometer_measure_reproducible_with_same_seed():
    f_true = true_specific_force(params.mass * params.gravity, params)
    a1 = Accelerometer(noise_std=0.05, bias_random_walk_std=0.005, rng=np.random.default_rng(1))
    a2 = Accelerometer(noise_std=0.05, bias_random_walk_std=0.005, rng=np.random.default_rng(1))
    for _ in range(10):
        m1 = a1.measure(f_true, dt=0.01)
        m2 = a2.measure(f_true, dt=0.01)
        assert np.array_equal(m1, m2)

def test_accelerometer_zero_noise_and_bias_returns_true_value_exactly():
    f_true = true_specific_force(params.mass * params.gravity, params)
    a = Accelerometer(noise_std=0.0, bias_random_walk_std=0.0, rng=np.random.default_rng(0))
    for _ in range(20):
        measured = a.measure(f_true, dt=0.01)
        assert np.array_equal(measured, f_true)

def test_accelerometer_bias_stays_zero_without_random_walk():
    f_true = true_specific_force(params.mass * params.gravity, params)
    a = Accelerometer(noise_std=0.05, bias_random_walk_std=0.0, rng=np.random.default_rng(3))
    for _ in range(50):
        a.measure(f_true, dt=0.01)
    assert np.array_equal(a.bias, np.zeros(3))
