import numpy as np

from dynamics import quaternion as quat
from dynamics import rigid_body as rb
from config.vehicle_params import DEFAULT_PARAMS


def test_inertia_matrix():
    params = DEFAULT_PARAMS
    assert np.allclose(rb.inertia_matrix(params), np.diag([params.Ixx, params.Iyy, params.Izz]))

def test_hover_equilibrium_zero_derivative():
    params = DEFAULT_PARAMS
    state = np.zeros(13)
    state[6] = 1.0  # 항등 쿼터니언

    d_state = rb.state_derivative(
        state,
        thrust = params.mass * params.gravity,
        torque = np.zeros(3),
        params = params,
        )

    assert np.allclose(d_state[3:6], [0,0,0])
    assert np.allclose(d_state[10:13], [0,0,0])

def test_free_fall_acceleration():
    params = DEFAULT_PARAMS
    state = np.zeros(13)
    state[6] = 1.0

    d_state = rb.state_derivative(
        state,
        thrust=0.0,
        torque=np.zeros(3),
        params=params,
    )

    assert np.allclose(d_state[3:6], [0, 0, params.gravity])


def test_tilted_hover_thrust_decomposition():
    params = DEFAULT_PARAMS
    pitch = np.radians(30.0)
    attitude = quat.from_euler_zyx(0.0,pitch, 0.0)

    state = np.zeros(13)
    state[6:10] = attitude

    thrust = 10.0
    d_state = rb.state_derivative(state, thrust=thrust, torque = np.zeros(3), params = params)
    accel = d_state[3:6]

    expected_horizontal = -thrust * np.sin(pitch) / params.mass
    expected_vertical = -thrust * np.cos(pitch) / params.mass + params.gravity

    assert np.isclose(accel[0], expected_horizontal)
    assert np.isclose(accel[2], expected_vertical)

def test_gyroscopic_coupling_nonzero_when_omega_nonzero():
    params = DEFAULT_PARAMS
    state = np.zeros(13)
    state[6] = 1.0
    state[10:13] = [1.0, 2.0, -1.5] # 임의의 각속도

    hover_thrust = params.mass*params.gravity
    d_state = rb.state_derivative(
        state,
        thrust=hover_thrust,
        torque=np.zeros(3),
        params=params
    )
    assert not np.allclose(d_state[10:13], [0, 0, 0])

def test_quaternion_derivative_matches_day2_function():
    params = DEFAULT_PARAMS
    state = np.zeros(13)
    state[6] = 1.0
    state[10:13] = [1.0, 2.0, -1.5]

    hover_thrust = params.mass * params.gravity
    d_state = rb.state_derivative(state, thrust=hover_thrust, torque=np.zeros(3), params=params)

    expected_q_dot = quat.derivative(np.array([1.0, 0, 0, 0]), np.array([1.0, 2.0, -1.5]))
    assert np.allclose(d_state[6:10], expected_q_dot)

