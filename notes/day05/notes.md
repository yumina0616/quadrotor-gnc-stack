# Day 5 — PID 제어 기초와 쿼터니언 자세오차

관련 코드: control/pid_attitude.py, dynamics/quaternion.py (conjugate)

## 키워드
- Feedback control
- PID
- 오일러각 오차의 문제 (짐벌락, 각도 순환/wrap-around)
- 쿼터니언 자세오차 (q_current⁻¹ ⊗ q_target)
- Quaternion conjugate (단위 쿼터니언의 역원)
- 소각도 근사 + 부호 처리 (짧은 길로 회전)