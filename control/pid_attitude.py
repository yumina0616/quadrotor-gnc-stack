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


class AttitudeController:
    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray, integral_limit: float = 0.5):
        """kp, ki, kd 각각 roll, pitch, yaw 축별로 게인을 가짐
        integral_limit은 anti-windup 클램프 범위"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self._integral = np.zeros(3)    # 누적 오차 초기화

    def reset(self) -> None:
        """적분 누적값 초기화(시뮬레이션 다시 시작할 때 사용)"""
        self._integral = np.zeros(3)

    def compute_torque(
        self, q_current: np.ndarray, q_target: np.ndarray, omega: np.ndarray, dt: float
    ) -> np.ndarray:
        """목표 자세까지 필요한 토크 pid로 계산"""

        e = attitude_error(q_current, q_target)
        self._integral += e*dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        torque = self.kp*e + self.ki*self._integral - self.kd*omega

        return torque
