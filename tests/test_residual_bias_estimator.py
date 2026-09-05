import numpy as np
import pytest
from estimation.residual_bias_estimator import ResidualBiasEstimator


def test_fit_recovers_known_linear_coefficient():
    rng = np.random.default_rng(0)
    true_coef = 0.005
    true_intercept = 0.0
    thrust_samples = np.linspace(5.0, 15.0, 200)  # 호버 추력(~9.8N) 근처 범위
    noise = rng.normal(0, 0.01, size=200)
    residual_samples = true_coef * thrust_samples + true_intercept + noise

    estimator = ResidualBiasEstimator()
    estimator.fit(thrust_samples, residual_samples)

    assert np.isclose(estimator.coef_, true_coef, atol=0.001)
    assert np.isclose(estimator.intercept_, true_intercept, atol=0.01)


def test_predict_before_fit_raises():
    estimator = ResidualBiasEstimator()
    with pytest.raises(RuntimeError):
        estimator.predict(10.0)


def test_predict_matches_fitted_line():
    thrust_samples = np.linspace(5.0, 15.0, 50)
    residual_samples = 0.01 * thrust_samples + 0.2

    estimator = ResidualBiasEstimator()
    estimator.fit(thrust_samples, residual_samples)

    predicted = estimator.predict(10.0)
    assert np.isclose(predicted, 0.01 * 10.0 + 0.2, atol=1e-6)
