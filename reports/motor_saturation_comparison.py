"""Actuator saturation before/after 비교 그래프 생성"""
import numpy as np
import matplotlib.pyplot as plt
from scenarios.attitude_stabilization import simulate_attitude_stabilization
from dynamics import quaternion as quat

INITIAL_EULER = (np.radians(20), np.radians(-15), np.radians(30))


def euler_history(states):
    """state 이력(N,13) -> degree 단위 [roll,pitch,yaw] 이력(N,3)."""
    return np.degrees(np.array([quat.to_euler_zyx(s[6:10]) for s in states]))


def plot_comparison(save_path="reports/motor_saturation.png"):
    result_ideal = simulate_attitude_stabilization(INITIAL_EULER, use_motor_mixing=False)
    result_mixing = simulate_attitude_stabilization(INITIAL_EULER, use_motor_mixing=True)

    times_ideal = np.array(result_ideal["time"])
    times_mixing = np.array(result_mixing["time"])
    eulers_ideal = euler_history(result_ideal["states"])
    eulers_mixing = euler_history(result_mixing["states"])

    labels = ["roll", "pitch", "yaw"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, (ax_ideal, ax_mixing) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax_ideal.plot(times_ideal, eulers_ideal[:, i], label=label, color=color)
        ax_mixing.plot(times_mixing, eulers_mixing[:, i], label=label, color=color)

    for ax, title in [(ax_ideal, "Ideal actuator"), (ax_mixing, "Motor mixing + saturation")]:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("time [s]")
        ax.set_title(title)
        ax.legend()

    ax_ideal.set_ylabel("angle [deg]")
    fig.suptitle("Attitude stabilization — ideal actuator vs motor mixing + saturation")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    plot_comparison()
