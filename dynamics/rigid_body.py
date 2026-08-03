import numpy as np
from config.vehicle_params import QuadrotorParams
from dynamics import quaternion as quat

STATE_DIM = 13
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)

def inertia_matrix(params: QuadrotorParams) -> np.ndarray:
    return np.array(
        [
            [params.Ixx,0,0],
            [0,params.Iyy,0],
            [0,0,params.Izz]
        ]
    )

def state_derivative(
    state: np.ndarray, thrust: float, torque: np.ndarray, params: QuadrotorParams
) -> np.ndarray:
    """뉴턴-오일러 운동방정식.
    thrust: 총 추력 크기(N), body frame -z 방향으로 작용.
    torque: body frame 토크 벡터 [tau_x, tau_y, tau_z] (N*m).
    반환값: d(state)/dt, state와 같은 13개 벡터 레이아웃.
    """
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:]

    R = quat.to_rotation_matrix(q)
    accel_world = R@[0,0,-thrust]/params.mass+[0,0,params.gravity]

    q_dot = quat.derivative(q, omega)

    I = inertia_matrix(params)
    angular_momentum = I@omega
    gyroscopic_term = np.cross(omega, angular_momentum)
    omega_dot = np.linalg.solve(I, torque - gyroscopic_term)

    d_state = np.zeros(STATE_DIM)
    d_state[POS] = vel
    d_state[VEL] = accel_world
    d_state[QUAT] = q_dot
    d_state[OMEGA] = omega_dot
    return d_state

def rk4_step(
    state: np.ndarray, thrust: float, torque: np.ndarray, dt: float, params: QuadrotorParams
) -> np.ndarray:
    """RK4로 한 스텝(dt) 적분. thrust/torque는 이 스텝 동안 일정하다고 가정."""
    k1 = state_derivative(state, thrust, torque, params)
    k2 = state_derivative(state + dt/2 * k1, thrust, torque, params)
    k3 = state_derivative(state + dt/2 * k2, thrust, torque, params)
    k4 = state_derivative(state + dt * k3, thrust, torque, params)

    next_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    next_state[QUAT] = quat.normalize(next_state[QUAT])  # 재정규화 -> 누적 오차 방지
    return next_state