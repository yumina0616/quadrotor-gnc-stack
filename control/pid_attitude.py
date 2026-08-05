import numpy as np
from dynamics import quaternion as quat

def attitude_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """현재 자세에서 목표 자세까지의 body-frame 오차 벡터(라디안 근사, 3성분).

    q_current, q_target: (w,x,y,z) 단위 쿼터니언.
    """

    q_error = quat.quaternion_multiply(quat.conjugate(q_current), q_target)

    if q_error[0]<0:
        q_error *= -1

    return 2 * q_error[1:4]
