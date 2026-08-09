# Day 6 — PID 조립: derivative kick 회피와 anti-windup

관련 코드: control/pid_attitude.py (AttitudeController)

## 키워드
- Derivative kick (오차를 직접 미분할 때 목표값 급변 시 스파이크 발생)
- Rate feedback (d(error)/dt ≈ -omega, D항을 -kd*omega로 계산)
- Integral windup (물리적 한계로 오차가 안 줄 때 적분값이 과도하게 누적)
- Anti-windup / 클램핑 (np.clip으로 적분값 범위 제한)