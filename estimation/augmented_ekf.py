import numpy as np

class AugmentedEKF:
    """위치+속도+accel bias를 함께 추정.
    상태 x = [px,py,pz, vx,vy,vz, bx,by,bz]
    """

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
        bias_est = self.x[6:9]

        accel_world = R_body_to_world @ (specific_force_body - bias_est) + np.array([0,0,gravity])

        A = np.block([
            [np.eye(3), dt*np.eye(3), np.zeros((3,3))],
            [np.zeros((3,3)), np.eye(3), -dt*R_body_to_world],
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]
        ])

        b = np.concatenate([np.zeros(3), (R_body_to_world@specific_force_body+np.array([0,0,gravity]))*dt, np.zeros(3)])

        self.x = A @ self.x + b
        self.P = A@self.P@np.transpose(A) + self.Q*dt

    def update(
        self,
        gps_measurement: np.ndarray | None
    ) -> None:
        if gps_measurement is None:
            return

        H = np.block([np.eye(3), np.zeros((3,6))])
        y = gps_measurement - H@self.x
        S = H@self.P@np.transpose(H) + self.R
        K = self.P@np.transpose(H)@np.linalg.inv(S)

        self.x = self.x + K@y
        self.P = (np.eye(9)-K@H)@self.P