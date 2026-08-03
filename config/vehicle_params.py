from dataclasses import dataclass

@dataclass(frozen=True)
class QuadrotorParams:

    mass: float = 1.0

    Ixx: float = 0.0122
    Iyy: float = 0.0122
    Izz: float = 0.0224

    gravity: float = 9.81

    arm_length: float = 0.20
    drag_torque_coeff: float = 0.016
    max_motor_thrust: float = 6.0

DEFAULT_PARAMS = QuadrotorParams()