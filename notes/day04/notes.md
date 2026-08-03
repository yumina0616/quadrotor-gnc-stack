# Day 04 — 모터 믹싱과 RK4 수치적분

관련 코드: `config/vehicle_params.py` (필드 추가), `dynamics/mixing.py` (신규),
`dynamics/rigid_body.py` (`rk4_step` 함수 추가)

## 키워드
- Motor mixing ('+' 배치, mixing matrix)
- Control allocation (역방향 믹싱, np.linalg.solve)
- Numerical integration (왜 필요한가 — closed-form 해 없음)
- Euler method의 한계 (1차 정확도)
- RK4 (Runge-Kutta 4차, k1~k4 가중평균)
- 적분 후 쿼터니언 재정규화