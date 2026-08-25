import numpy as np

class NominalEKF:
    """평행이동 상태(위치+속도)만 추정."""
    
    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        process_noise_std: np.ndarray,
        measurement_noise_std: np.ndarray
    ):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = np.diag(process_noise_std**2)
        self.R = np.diag(measurement_noise_std**2)

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


    def update(self, gps_measurement: np.ndarray | None) -> None:
        if gps_measurement is None : 
            return
        H = np.block([np.eye(3), np.zeros((3,3))])
        y = gps_measurement - H @ self.x    # innovation(residual)
        S = H @ self.P @ np.transpose(H) + self.R   # 전체(예측 + 측정) 불확실성
        K = self.P @ np.transpose(H) @ np.linalg.inv(S) # Kalman gain: P와 R 비율로 정해지는 가중치(얼마나 측정 쪽으로 끌어당길지)
        
        # next 예측값, 불확실성 업데이트
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
