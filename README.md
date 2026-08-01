# Quadrotor GNC Portfolio

Requirements-based automated V&V harness for quadrotor attitude control, combining
classical estimation (EKF / augmented-state EKF) with a physics-informed state
estimator, and an LLM-assisted (but not LLM-decided) verification harness.

## Status
🚧 In progress — Week 1 of 4 (dynamics + PID baseline)

## Structure
```
dynamics/       6DOF rigid-body dynamics model
estimation/     EKF, augmented-state EKF, physics-informed estimator
control/        PID attitude controller
requirements/   Requirement definitions (ID / threshold / verification method)
scenarios/      Disturbance & sensor-fault scenarios, Monte Carlo configs
tests/          Unit tests
reports/        Generated V&V reports
config/         Configuration files
docs/           Design docs (this repo's docs/private/ is git-ignored — personal notes)
```

## Development process
Each implementation step follows a fixed daily loop: study the underlying theory
first (keywords, then explanations, kept as private working notes), implement
that piece by hand, verify it with unit tests, then commit. Commit history is
structured to reflect this (`feat(day-N)` for the implementation, `test(day-N)`
for its tests), so the log itself traces the build order: theory understood
before code, code verified before it's considered done.

## Design principle: LLM stays out of the verdict
The verification harness uses an LLM agent only to (1) propose test-scenario
candidates, (2) summarize failure logs, and (3) draft report text. All numeric
computation and pass/fail verdicts against requirements are handled by
deterministic code — never by the agent.

## Limitations / Future Work
- Real-time C/C++ embedded implementation (currently Python simulation only)
- Hardware-in-the-Loop (HIL) testing
- Full CI/CD pipeline and code coverage tooling
- DO-178C-style requirements traceability
- Broader safety analysis (sensor faults, single-motor degradation, hazard linkage)
- Additional baselines (adaptive EKF / UKF, hybrid residual estimator)
