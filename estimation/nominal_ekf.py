import numpy as np

class NominalEKF:
    """평행이동 상태(위치+속도)만 추정."""
    
    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        process_noise_std: np.ndarray,
    ):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = np.diag(process_noise_std**2)

    def predict(
        self,
        specific_force_body: np.ndarray,
        R_body_to_world: np.ndarray,
        dt: float,
        gravity: float,
    ) -> None:
        accel_world = R_body_to_world @ specific_force_body + np.array([0,0,gravity])
        A = np.block(
                [
                [np.eye(3), dt*np.eye(3)],
                [np.zeros((3,3)), np.eye(3)]
                ]
            )
        b = np.concatenate([np.zeros(3), accel_world * dt])
        self.x = A @ self.x +b
        self.P = A @ self.P @ np.transpose(A) + self.Q * dt