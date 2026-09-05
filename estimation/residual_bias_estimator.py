import numpy as np

class ResidualBiasEstimator:
    """추력 -> z축 진동 상관 accel bias residual 선형 회귀 모델.
    """

    def __init__(self):
        self.coef_: float | None = None
        self.intercept_: float | None = None

    def fit(self, thrust_samples: np.ndarray, residual_z_samples: np.ndarray) -> None:
        self.coef_, self.intercept_ = np.polyfit(thrust_samples, residual_z_samples, deg=1)

    def predict(self, thrust: float) -> float:
        if self.coef_ is None:
            raise RuntimeError("Run fit() first.")
        return self.coef_ * thrust + self.intercept_