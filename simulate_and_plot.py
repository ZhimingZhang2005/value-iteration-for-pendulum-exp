"""
simulate_and_plot.py
====================
仿真与可视化模块。

功能
----
1. plot_value_function : 绘制价值函数 3D 曲面图
2. plot_policy         : 绘制最优策略 2D 热力图
3. plot_trajectory     : 绘制倒立摆仿真轨迹（相图 + 时间序列）
4. compare_experiments : 并排对比两种代价函数下的实验结果

所有绘图函数均接受可选的 ax/fig 参数，方便在 main.py 中组合使用。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
from matplotlib import cm

from pendulum import Pendulum
from grid import StateInputGrid


# ===========================================================================
# 1. 价值函数曲面图
# ===========================================================================

def plot_value_function(J: np.ndarray, grid: StateInputGrid,
                        title: str = "Value Function",
                        ax=None) -> plt.Figure:
    """
    绘制价值函数 J(θ, θ̇) 的 3D 曲面图。

    参数
    ----
    J     : 价值函数数组，形状 (n_theta, n_theta_dot)
    grid  : StateInputGrid 实例（提供网格坐标）
    title : 图标题
    ax    : 可选的 Axes3D 对象；为 None 时自动创建

    返回
    ----
    fig : matplotlib Figure 对象
    """
    TH, THD = grid.meshgrid()

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # 截断极大值避免曲面畸形（最短时间代价时边缘可能极大）
    J_plot = np.clip(J, 0, np.percentile(J, 99))

    surf = ax.plot_surface(TH, THD, J_plot, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='J*(θ, θ̇)')
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_zlabel('J*')
    ax.set_title(title)
    return fig


# ===========================================================================
# 2. 策略热力图
# ===========================================================================

def plot_policy(policy: np.ndarray, grid: StateInputGrid,
                title: str = "Optimal Policy",
                ax=None) -> plt.Figure:
    """
    绘制最优策略 π*(θ, θ̇) 的 2D 热力图。

    参数
    ----
    policy : 策略数组，形状 (n_theta, n_theta_dot)，值为控制力矩
    grid   : StateInputGrid 实例
    title  : 图标题
    ax     : 可选 Axes 对象

    返回
    ----
    fig : matplotlib Figure 对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    # imshow 期望 (行=y轴, 列=x轴)，需转置策略矩阵
    u_max = max(abs(grid.u_lim[0]), abs(grid.u_lim[1]))
    im = ax.imshow(policy.T,
                   origin='lower',
                   extent=[grid.theta_lim[0], grid.theta_lim[1],
                           grid.theta_dot_lim[0], grid.theta_dot_lim[1]],
                   aspect='auto',
                   cmap='RdBu_r',
                   vmin=-u_max, vmax=u_max)
    fig.colorbar(im, ax=ax, label='u* (N·m)')
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title(title)

    # 标出目标点（θ=π, θ̇=0）
    ax.plot(np.pi, 0, 'k*', markersize=12, label='Goal (θ=π)')
    ax.plot(-np.pi, 0, 'k*', markersize=12)
    ax.legend(loc='upper right', fontsize=8)
    return fig


# ===========================================================================
# 3. 仿真轨迹图
# ===========================================================================

def plot_trajectory(thetas: np.ndarray, theta_dots: np.ndarray,
                    us: np.ndarray, dt: float,
                    title: str = "Simulation Trajectory",
                    fig=None) -> plt.Figure:
    """
    绘制仿真轨迹的三子图：
      (a) 相图 (θ vs θ̇)
      (b) 摆角随时间变化
      (c) 控制量随时间变化

    参数
    ----
    thetas     : 摆角序列，形状 (N+1,)
    theta_dots : 角速度序列，形状 (N+1,)
    us         : 控制序列，形状 (N,)
    dt         : 时间步长（s）
    title      : 图总标题
    fig        : 可选 Figure 对象

    返回
    ----
    fig : matplotlib Figure 对象
    """
    t = np.arange(len(thetas)) * dt

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = fig.subplots(1, 3)

    # (a) 相图
    ax = axes[0]
    ax.plot(thetas, theta_dots, 'b-', linewidth=1.2, alpha=0.8)
    ax.plot(thetas[0], theta_dots[0], 'go', markersize=10, label='Start')
    ax.plot(thetas[-1], theta_dots[-1], 'r^', markersize=10, label='End')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(np.pi, color='r', linewidth=0.8, linestyle='--', label='θ=π')
    ax.axvline(-np.pi, color='r', linewidth=0.8, linestyle='--')
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('Phase Portrait')
    ax.legend(fontsize=8)

    # (b) 摆角时间序列
    ax = axes[1]
    ax.plot(t, thetas, 'b-', linewidth=1.2)
    ax.axhline(np.pi, color='r', linewidth=0.8, linestyle='--', label='θ=π')
    ax.axhline(-np.pi, color='r', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('θ vs Time')
    ax.legend(fontsize=8)

    # (c) 控制量时间序列
    ax = axes[2]
    t_u = np.arange(len(us)) * dt
    ax.plot(t_u, us, 'g-', linewidth=1.2)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('u (N·m)')
    ax.set_title('Control Input vs Time')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ===========================================================================
# 4. 对比两种代价函数的实验结果（综合图）
# ===========================================================================

def compare_experiments(results: dict,
                        pendulum: Pendulum,
                        grid: StateInputGrid,
                        theta0: float = 0.0,
                        theta_dot0: float = 0.0,
                        n_steps: int = 300,
                        save_prefix: str = "result"):
    """
    对两种代价函数（二次型 vs 最短时间）的实验结果进行仿真并生成对比图。

    results 字典格式：
        {
          "quadratic": {"J": J_array, "policy": policy_array, "label": "..."},
          "min_time":  {"J": J_array, "policy": policy_array, "label": "..."},
        }

    每种代价函数生成三张图：
      1. 价值函数 3D 曲面图
      2. 策略热力图
      3. 仿真轨迹图

    参数
    ----
    results    : 包含两种实验结果的字典
    pendulum   : Pendulum 实例（用于仿真）
    grid       : StateInputGrid 实例
    theta0     : 初始摆角（rad），默认 0.0（最低点）
    theta_dot0 : 初始角速度（rad/s），默认 0.0
    n_steps    : 仿真步数，默认 300
    save_prefix: 图片保存路径前缀
    """
    from value_iteration import make_policy_fn

    for key, res in results.items():
        J      = res["J"]
        policy = res["policy"]
        label  = res["label"]

        # ---- 价值函数曲面图 ----
        fig_v = plot_value_function(J, grid,
                                    title=f"Value Function — {label}")
        fig_v.savefig(f"{save_prefix}_{key}_value_function.png",
                      dpi=120, bbox_inches='tight')
        plt.close(fig_v)
        print(f"  已保存: {save_prefix}_{key}_value_function.png")

        # ---- 策略热力图 ----
        fig_p = plot_policy(policy, grid,
                            title=f"Optimal Policy — {label}")
        fig_p.savefig(f"{save_prefix}_{key}_policy.png",
                      dpi=120, bbox_inches='tight')
        plt.close(fig_p)
        print(f"  已保存: {save_prefix}_{key}_policy.png")

        # ---- 仿真轨迹图 ----
        policy_fn = make_policy_fn(policy, grid)
        thetas, theta_dots, us = pendulum.simulate(
            theta0, theta_dot0, policy_fn, n_steps)

        fig_t = plot_trajectory(thetas, theta_dots, us,
                                dt=pendulum.dt,
                                title=f"Trajectory — {label}")
        fig_t.savefig(f"{save_prefix}_{key}_trajectory.png",
                      dpi=120, bbox_inches='tight')
        plt.close(fig_t)
        print(f"  已保存: {save_prefix}_{key}_trajectory.png")

    # ---- 综合对比图：策略热力图并排 ----
    _plot_policy_comparison(results, grid, save_prefix)


def _plot_policy_comparison(results: dict, grid: StateInputGrid,
                             save_prefix: str):
    """绘制两种代价函数策略的并排对比热力图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    keys = list(results.keys())
    u_max = max(abs(grid.u_lim[0]), abs(grid.u_lim[1]))

    for ax, key in zip(axes, keys):
        policy = results[key]["policy"]
        label  = results[key]["label"]
        im = ax.imshow(policy.T,
                       origin='lower',
                       extent=[grid.theta_lim[0], grid.theta_lim[1],
                               grid.theta_dot_lim[0], grid.theta_dot_lim[1]],
                       aspect='auto',
                       cmap='RdBu_r',
                       vmin=-u_max, vmax=u_max)
        fig.colorbar(im, ax=ax, label='u* (N·m)')
        ax.set_xlabel('θ (rad)')
        ax.set_ylabel('θ̇ (rad/s)')
        ax.set_title(f"Policy — {label}")
        ax.plot(np.pi, 0, 'k*', markersize=12)
        ax.plot(-np.pi, 0, 'k*', markersize=12)

    fig.suptitle("Policy Comparison: Quadratic vs Minimum Time",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_policy_comparison.png",
                dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_prefix}_policy_comparison.png")
