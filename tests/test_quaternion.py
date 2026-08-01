import numpy as np

from dynamics import quaternion as q


def test_normalize_produces_unit_norm():
    test_q = [2.0, 0, 0, 0]
    result = np.linalg.norm(q.normalize(test_q))

    assert np.isclose(result, 1.0), f"norm이 1이 아님: {result}"


def test_identity_quaternion_gives_identity_rotation_matrix():
    # to_rotation_matrix([1,0,0,0]) 이 np.eye(3)과 같은지 확인 (np.allclose)
    assert np.allclose(q.to_rotation_matrix([1,0,0,0]), np.eye(3))


def test_rotation_matrix_is_orthogonal_with_det_one():
    # 임의의 단위 쿼터니언(예: normalize([0.7,0.2,0.5,-0.3]))에 대해
    # R = to_rotation_matrix(q) 를 구하고
    # R.T @ R 이 np.eye(3)과 같은지 (직교성), np.linalg.det(R) 이 1인지 확인
    test_q = q.normalize([0.7,0.2,0.5,-0.3])
    R = q.to_rotation_matrix(test_q)
    assert np.allclose(np.transpose(R)@R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_quaternion_multiply_identity():
    # quarternion_multiply([1,0,0,0], [1,0,0,0]) 이 [1,0,0,0]인지 확인
    assert np.allclose(q.quaternion_multiply([1,0,0,0], [1,0,0,0]), [1,0,0,0])


def test_derivative_is_zero_when_omega_is_zero():
    # 임의의 단위 쿼터니언에 대해 omega_body = [0,0,0]을 넣으면
    # derivative() 결과가 전부 0에 가까운지 확인
    assert np.allclose(
        q.derivative(q.normalize([0.2, 0.4, -0.3, 0.7]),
        [0,0,0]), [0,0,0,0]
    )


def test_euler_quaternion_roundtrip():
    # 여러 (roll, pitch, yaw) 조합 - 예: (0,0,0), (0.3,-0.2,0.5) 등 -
    # to_euler_zyx(from_euler_zyx(roll, pitch, yaw)) 가
    # 원래 (roll, pitch, yaw)와 np.allclose(atol=1e-9)로 같은지 확인
    

    for euler_set in [(0,0,0), (0.3, -0.2, 0.5)]:
        assert np.allclose(
                q.to_euler_zyx(q.from_euler_zyx(*euler_set)),
                euler_set,        
                atol=1e-9
            )
    
