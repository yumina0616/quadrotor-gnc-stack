# Day 03 — 뉴턴-오일러 운동방정식 (병진 + 회전)

관련 코드: config/vehicle_params.py, dynamics/rigid_body.py

## 키워드
- State vector (13개로 구성: position(3), velocity(3), quaternion(4), angular velocity(3))
- Newton's 2nd law (translational dynamics, body->world thrust rotation)
- Inertia tensor (diagonal, principal axes)
- Euler's rotation equation
- Gyroscopic coupling