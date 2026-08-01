import numpy as np

def normalize(q: np.ndarray) -> np.ndarray:
    """단위 쿼터니언 정규화"""
    return q / np.linalg.norm(q)


def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    return np.array(
            [
            [1-2*(y**2+z**2),   2*(x*y-z*w),     2*(x*z+y*w)],
            [2*(x*y+z*w),     1-2*(x**2+z**2),   2*(y*z-x*w)],
            [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x**2+y**2)]
            ]
        )


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw = a[0]
    ax = a[1]
    ay = a[2]
    az = a[3]

    bw = b[0]
    bx = b[1]
    by = b[2]
    bz = b[3]

    return np.array(
        [
        aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+bw*ax+ay*bz-az*by,
        aw*by+bw*ay-ax*bz+az*bx,
        aw*bz+bw*az+ax*by-ay*bx
        ]
    )

def derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """쿼터니언 기구학 방정식"""
    omega_quat = np.concatenate(([0], omega_body))
    return 1/2 * quaternion_multiply(q, omega_quat)


def from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """roll/pitch/yaw -> 쿼터니언 변환. ZYX 순서"""
    qx = [np.cos(roll/2), np.sin(roll/2), 0, 0]
    qy = [np.cos(pitch/2), 0, np.sin(pitch/2), 0]
    qz = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]

    return quaternion_multiply(qz, quaternion_multiply(qy, qx))


def to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """쿼터니언 -> [roll, pitch, yaw] 변환"""
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    
    return np.array([roll, pitch, yaw])