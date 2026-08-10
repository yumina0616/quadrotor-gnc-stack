# Day 7 - 폐루프 시뮬레이션, 게인 튜닝

관련 코드: scenarios/attitude_stabilization.py, reports/gain_tuning_comparision.py

## 키워드
- 폐루프(closed_loop) 시뮬레이션: state -> controller -> torque -> rk4_step -> next state 순서로 모듈 최종 연결.
- 이상적 actuator 모델로 시험 (모터 믹싱/포화 없이 torque를 기체에 직접 반영. 실제 모터 믹싱은 추후 진행 예정)
- PID 게인 튜닝: simulate_attitude_stabilization으로 state 확인 후 PID 게인 조정. ki/kp 비율, windup 오버슈트 해결. (추후 PID 게인 모델 학습 진행해볼 예정)
- 게인 튜닝 구체적 수치: pitch축의 kp가 다른 축에 비해 작아(roll 2.0, yaw 4.0) 오버슈트가 8.5°까지 발생. -> kp를 3.0으로 수정 후 1.4°로 줄었음을 확인. yaw 축의 ki/kp 비율이 0.85로 큼 -> kp를 5.0로 올리고 ki를 2.5로 낮추어 오버슈트 6.5°→4.0°로 줄었음을 확인.