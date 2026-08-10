import numpy as np
from dynamics import quaternion as quat
from dynamics import rigid_body as rb
from config.vehicle_params import DEFAULT_PARAMS
from control.pid_attitude import AttitudeController

params = DEFAULT_PARAMS

def make_controller(integral_limit = 0.5):
    kp = np.array([2.0, 3.0, 15.0])
    ki = np.array([1.2, 2.3, 2.5])
    kd = np.array([0.5, 1.0, 2.2])
    return AttitudeController(kp, ki, kd, integral_limit=integral_limit)

def simulate_attitude_stabilization(
    initial_euler: tuple[float, float, float],
    target_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    duration: float = 3.0,
    dt: float = 0.01,
    controller=None
) -> dict:
    """자세 안정화 폐루프 시뮬레이션.(정지 상태, 자세만 기울어져 있는 초기 조건)

    initial_euler, target_euler: (roll, pitch, yaw), 라디안.
    반환값: 시간 이력과 state 이력을 담은 dict."""
    ctrl = controller if controller is not None else make_controller()

    state_current, state_target = np.zeros(13), np.zeros(13)
    state_current[6:10] = quat.from_euler_zyx(*initial_euler)
    state_target[6:10] = quat.from_euler_zyx(*target_euler)

    time_history = []
    state_history = []

    for i in range(int(duration/dt)):
        omega = state_current[10:13]
        torque = ctrl.compute_torque(state_current[6:10], state_target[6:10], omega, dt)
        thrust = params.mass * params.gravity

        state_current = rb.rk4_step(state_current, thrust, torque, dt, params)
        time_history.append(i*dt)
        state_history.append(state_current)

    return {
        "time": time_history,
        "states": state_history,
    }

    