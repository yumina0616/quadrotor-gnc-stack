"""Day7 게인 튜닝 before/after 비교 그래프 생성"""
import numpy as np
import matplotlib.pyplot as plt
from control.pid_attitude import AttitudeController
from scenarios.attitude_stabilization import simulate_attitude_stabilization
from dynamics import quaternion as quat

INITIAL_EULER = (np.radians(20), np.radians(-15), np.radians(30))


def make_before_controller():
    """튜닝 전 게인 — pitch/yaw의 ki가 kp 대비 과도하게 커서 오버슈트가 심했던 상태."""
    kp = np.array([2.0, 0.3, 4.0])
    ki = np.array([1.2, 2.3, 3.4])
    kd = np.array([0.5, 1.0, 2.2])
    return AttitudeController(kp, ki, kd)


def make_after_controller():
    """튜닝 후 게인 — pitch kp 0.3→3.0, yaw kp 4.0→5.0/ki 3.4→2.5로 조정."""
    kp = np.array([2.0, 3.0, 5.0])
    ki = np.array([1.2, 2.3, 2.5])
    kd = np.array([0.5, 1.0, 2.2])
    return AttitudeController(kp, ki, kd)


def euler_history(states):
    """state 이력(N,13) -> degree 단위 [roll,pitch,yaw] 이력(N,3)."""
    return np.degrees(np.array([quat.to_euler_zyx(s[6:10]) for s in states]))


def plot_comparison(save_path="reports/day07_gain_tuning.png"):
    result_before = simulate_attitude_stabilization(INITIAL_EULER, controller=make_before_controller())
    result_after = simulate_attitude_stabilization(INITIAL_EULER, controller=make_after_controller())

    times_before = np.array(result_before["time"])
    times_after = np.array(result_after["time"])
    eulers_before = euler_history(result_before["states"])
    eulers_after = euler_history(result_after["states"])

    labels = ["roll", "pitch", "yaw"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax_before.plot(times_before, eulers_before[:, i], label=label, color=color)
        ax_after.plot(times_after, eulers_after[:, i], label=label, color=color)

    for ax, title in [(ax_before, "Before tuning"), (ax_after, "After tuning")]:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("time [s]")
        ax.set_title(title)
        ax.legend()

    ax_before.set_ylabel("angle [deg]")
    fig.suptitle("Day7 attitude stabilization — PID gain tuning comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    plot_comparison()
