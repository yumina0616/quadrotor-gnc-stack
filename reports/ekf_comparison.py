import numpy as np
import matplotlib.pyplot as plt
from scenarios.attitude_stabilization import simulate_attitude_stabilization
from scenarios.disturbances import WindDisturbance
from scenarios.sensor_models import Accelerometer, GPS
from estimation.nominal_ekf import NominalEKF
from estimation.augmented_ekf import AugmentedEKF


def run_comparison(save_path="reports/ekf_comparison.png"):
    rng = np.random.default_rng(0)
    wind = WindDisturbance(steady_force=np.array([0.5, 0, 0]), gust_std=0.05, rng=rng)

    rng = np.random.default_rng(1)
    accel = Accelerometer(noise_std=0.03, bias_random_walk_std=0.002, rng=rng)
    accel.bias = np.array([0.15, -0.1, 0.05])

    rng = np.random.default_rng(2)
    gps = GPS(noise_std=1.5, dropout_prob=0.05, rng=rng)


    measurement_noise_std = np.array([2.0, 2.0, 2.0])

    nominal_ekf = NominalEKF(
        x0=np.zeros(6),
        P0=np.diag([1.0] * 3 + [0.5] * 3),
        process_noise_std=np.array([0.01] * 3 + [0.1] * 3),
        measurement_noise_std=measurement_noise_std,
    )

    augmented_ekf = AugmentedEKF(
        x0=np.zeros(9),
        P0=np.diag([1.0] * 3 + [0.5] * 3 + [0.05] * 3),
        process_noise_std=np.array([0.01] * 3 + [0.1] * 3 + [0.001] * 3),
        measurement_noise_std=measurement_noise_std,
    )




    result = simulate_attitude_stabilization(initial_euler=(0,0,0), duration=10.0, dt=0.01, wind=wind, accelerometer=accel, gps=gps, nominal_ekf=nominal_ekf, augmented_ekf=augmented_ekf)

    true_pos = np.array([s[0:3] for s in result["states"]])
    nominal_pos = np.array([x[0:3] for x in result["nominal_ekf_x"]])
    augmented_pos = np.array([x[0:3] for x in result["augmented_ekf_x"]])

    error_nominal = np.linalg.norm(true_pos - nominal_pos, axis=1)
    error_augmented = np.linalg.norm(true_pos - augmented_pos, axis=1)

    rmse_nominal = np.sqrt(np.mean(error_nominal**2))
    rmse_augmented = np.sqrt(np.mean(error_augmented**2))
    print(f"nominal RMSE: {rmse_nominal:.3f} m, augmented RMSE: {rmse_augmented:.3f} m")


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["time"], error_nominal, label="nominal EKF")
    ax.plot(result["time"], error_augmented, label="augmented EKF")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Nominal vs Augmented EKF — position error over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"saved: {save_path}")

if __name__ == "__main__":
    run_comparison()