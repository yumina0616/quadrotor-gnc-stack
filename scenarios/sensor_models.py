import numpy as np


def true_specific_force(
    thrust: float,
    params,
    wind_force: np.ndarray | None = None,
    R_body_to_world: np.ndarray | None = None
    ) -> np.ndarray:
    """body frame에서 본 specific force"""
    if wind_force is None:
        wind_force = np.zeros(3)
    if R_body_to_world is None:
        R_body_to_world = np.eye(3)
    return np.array([0, 0, -thrust/params.mass] + R_body_to_world.T @ wind_force/params.mass)

class Gyroscope:
    """ 자이로스코프 측정 모델: 실제 각속도 + bias + 노이즈 가 섞인 측정값.

    노이즈와 bias 랜덤워크는 (x,y,z) 축마다 독립.
    """

    def __init__(
        self,
        noise_std: float,
        bias_random_walk_std: float,
        rng: np.random.Generator | None = None,
    ):

        self.noise_std = noise_std
        self.bias_random_walk_std = bias_random_walk_std
        self.bias = np.zeros(3)
        self.rng = rng if rng is not None else np.random.default_rng()

    def measure(self, true_omega: np.ndarray, dt: float) -> np.ndarray:
        """true_omega -> mesured_omega"""
        self.bias += self.rng.normal(0, self.bias_random_walk_std * np.sqrt(dt), size = 3)
        noise = self.rng.normal(0, self.noise_std, size = 3)

        return true_omega + self.bias + noise


class Accelerometer:
    """가속도계 측정 모델: specific force -> bias+노이즈가 섞이 측정값"""

    def __init__(
        self,
        noise_std: float,
        bias_random_walk_std: float,
        vibration_coefficient: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.noise_std = noise_std
        self.bias_random_walk_std = bias_random_walk_std
        self.bias = np.zeros(3)
        self.vibration_coefficient = vibration_coefficient
        self.rng = rng if rng is not None else np.random.default_rng()

    def measure(self, true_specific_force: np.ndarray, thrust: float, dt: float) -> np.ndarray:
        self.bias += self.rng.normal(0, self.bias_random_walk_std * np.sqrt(dt), size = 3)
        vibration_bias = np.array([0.0, 0.0, self.vibration_coefficient * thrust])
        noise = self.rng.normal(0, self.noise_std, size = 3)

        return true_specific_force + self.bias + vibration_bias + noise

class GPS:
    """GPS 위치 측정 모델: 진짜 위치 -> 노이즈 포함된 측정값, 또는 dropout일 때 None."""

    def __init__(
        self,
        noise_std: float,
        dropout_prob: float,
        rng: np.random.Generator | None = None,
    ):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.rng = rng if rng is not None else np.random.default_rng()

    def measure(self, true_position: np.ndarray) -> np.ndarray | None:
        if self.rng.random() < self.dropout_prob:
            return None
        noise = self.rng.normal(0, self.noise_std, size=3)
        return true_position + noise
        