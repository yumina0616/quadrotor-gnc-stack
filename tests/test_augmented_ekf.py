import numpy as np
from estimation.augmented_ekf import AugmentedEKF

gravity = 9.81


def make_ekf(x0=None):
    return AugmentedEKF(
        x0=np.zeros(9) if x0 is None else x0,
        P0=np.diag([1.0] * 3 + [0.5] * 3 + [0.05] * 3),
        process_noise_std=np.array([0.01] * 3 + [0.1] * 3 + [0.001] * 3),
        measurement_noise_std=np.array([2.0, 2.0, 2.0]),
    )


def test_predict_hover_zero_bias_stays_at_origin():
    ekf = make_ekf()
    sf_hover = np.array([0.0, 0.0, -gravity])  # 추력이 중력을 상쇄
    for _ in range(100):
        ekf.predict(sf_hover, np.eye(3), dt=0.01, gravity=gravity)
    assert np.allclose(ekf.x[0:6], np.zeros(6), atol=1e-9)


def test_predict_covariance_grows_without_update():
    ekf = make_ekf()
    sf_hover = np.array([0.0, 0.0, -gravity])
    p0_diag = np.diag(ekf.P).copy()
    for _ in range(50):
        ekf.predict(sf_hover, np.eye(3), dt=0.01, gravity=gravity)
    assert np.all(np.diag(ekf.P) >= p0_diag)


def test_predict_bias_coupling_uses_rotation_matrix():
    """속도-bias 교차 블록이 -dt*I가 아니라 -dt*R_body_to_world여야 한다.
    body frame에서 x축 bias가 90도 회전(R: body x -> world y)을 거치면
    world frame에서는 y축 속도로 나타나야 한다."""
    x0 = np.zeros(9)
    x0[6:9] = [1.0, 0.0, 0.0]  # body frame bias
    ekf = make_ekf(x0)

    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # z축 기준 90도 회전
    sf_hover = np.array([0.0, 0.0, -gravity])
    ekf.predict(sf_hover, R, dt=0.01, gravity=gravity)

    assert np.allclose(ekf.x[3:6], [0.0, -0.01, 0.0], atol=1e-9)


def test_predict_free_fall_matches_analytic():
    ekf = make_ekf()
    for _ in range(1000):
        ekf.predict(np.zeros(3), np.eye(3), dt=0.001, gravity=gravity)
    assert np.isclose(ekf.x[5], gravity * 1.0)
    assert np.isclose(ekf.x[2], 0.5 * gravity * 1.0, atol=0.01)


def test_update_none_measurement_leaves_state_unchanged():
    ekf = make_ekf()
    ekf.predict(np.array([0.0, 0.0, -gravity]), np.eye(3), dt=0.01, gravity=gravity)
    x_before = ekf.x.copy()
    p_before = ekf.P.copy()
    ekf.update(None)
    assert np.array_equal(ekf.x, x_before)
    assert np.array_equal(ekf.P, p_before)


def test_update_reduces_position_covariance():
    ekf = make_ekf()
    ekf.predict(np.array([0.0, 0.0, -gravity]), np.eye(3), dt=0.01, gravity=gravity)
    p_pos_before = np.diag(ekf.P)[0:3].copy()
    ekf.update(np.zeros(3))
    p_pos_after = np.diag(ekf.P)[0:3]
    assert np.all(p_pos_after < p_pos_before)


def test_bias_estimate_converges_and_position_stays_bounded():
    """가속도계에 상수 bias가 있어도, GPS update가 반복되면 bias 추정치가
    실제 bias에 수렴하고 위치 추정 오차가 발산하지 않아야 한다 —
    이게 NominalEKF 대비 AugmentedEKF의 핵심 이점이다."""
    true_bias = np.array([0.1, 0.0, 0.0])
    sf_true = np.array([0.0, 0.0, -gravity])
    ekf = make_ekf()

    for i in range(2000):
        sf_measured = sf_true + true_bias
        ekf.predict(sf_measured, np.eye(3), dt=0.01, gravity=gravity)
        if i % 5 == 0:
            ekf.update(np.zeros(3))

    assert np.allclose(ekf.x[6:9], true_bias, atol=0.05)
    assert np.allclose(ekf.x[0:3], np.zeros(3), atol=0.1)
