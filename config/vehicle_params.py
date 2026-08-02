from dataclasses import dataclass

@dataclass(frozen=True)
class QuadrotorParams:
    mass: float = 1.0
    Ixx: float = 0.0122
    Iyy: float = 0.0122
    Izz: float = 0.0224
    gravity: float = 9.81

DEFAULT_PARAMS = QuadrotorParams()