import numpy as np

class WindDisturbance:
    """단순화된 풍외란 모델: world frame 기준 외력.
    steady_force(평균 바람) + gust(랜덤워크로 변동)"""

    def __init__(
        self,
        steady_force: np.ndarray,
        gust_std: float,
        rng: np.random.Generator | None = None,
    ):
        self.steady_force = steady_force
        self.gust_std = gust_std
        self.gust = np.zeros(3)
        self.rng = rng if rng is not None else np.random.default_rng()


    def force(self, dt: float) -> np.ndarray:
        self.gust += self.rng.normal(0, self.gust_std*np.sqrt(dt), size=3)

        return self.steady_force + self.gust
