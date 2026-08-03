import numpy as np
from config.vehicle_params import QuadrotorParams

def mixing_matrix(params: QuadrotorParams) -> np.ndarray:
    """A: wrench = A @ motor_thrusts, wrench=[T, tau_x, tau_y, tau_z]."""
    l = params.arm_length
    k = params.drag_torque_coeff

    return np.array(
        [
            [1,1,1,1],
            [0,-l,0,l],
            [l,0,-l,0],
            [-k,k,-k,k]
        ]
    )

def motors_to_wrench(motor_thrusts: np.ndarray, params: QuadrotorParams) -> np.ndarray:
    """정방향, 시뮬레이터용"""
    return mixing_matrix(params)@motor_thrusts

def wrench_to_motors(wrench: np.ndarray, params: QuadrotorParams) -> np.ndarray:
    """역방향/제어배분, 제어기용"""
    return np.linalg.solve(mixing_matrix(params), wrench)