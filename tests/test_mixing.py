import numpy as np
from dynamics import mixing
from config.vehicle_params import DEFAULT_PARAMS

params = DEFAULT_PARAMS

def test_symmetric_thrust_gives_zero_torque():
    thrust = np.ones(4) * 2.5

    assert np.allclose(
        mixing.motors_to_wrench(thrust, params),
        [4*2.5,0,0,0]
    )

def test_roundtrip():
    thrust = [1,2,3,4]
    
    assert np.allclose(
        mixing.wrench_to_motors(
            mixing.motors_to_wrench(thrust, params),
            params
        ),
        thrust
    )