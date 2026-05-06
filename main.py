"""
main.py
=======
主入口文件 —— 一键复现"倒立摆价值迭代"实验。

运行方式
--------
    python main.py

输出
----
在当前目录下生成以下 PNG 图片：
  result_quadratic_value_function.png   — 二次型代价：价值函数曲面
  result_quadratic_policy.png           — 二次型代价：最优策略热力图
  result_quadratic_trajectory.png       — 二次型代价：仿真轨迹（平滑控制）
  result_min_time_value_function.png    — 最短时间代价：价值函数曲面
  result_min_time_policy.png            — 最短时间代价：最优策略热力图
  result_min_time_trajectory.png        — 最短时间代价：仿真轨迹（砰-砰控制）
  result_policy_comparison.png          — 两种策略并排对比图

实验说明
--------
倒立摆参数：
  - 质量 m=1 kg，摆长 l=1 m，重力 g=9.81 m/s²，阻尼 b=0.1 N·m·s
  - 时间步长 dt=0.05 s
  - 最大控制力矩 u_max=3 N·m（即控制量 u ∈ [-3, 3]）

状态空间网格：
  - θ  ：[-π, π] 分为 51 个点
  - θ̇ ：[-8, 8] rad/s 分为 51 个点
  - u  ：[-3, 3] N·m  分为 9 个离散点

价值迭代参数：
  - 折扣因子 γ = 0.999
  - 收敛阈值 ε = 1e-3
  - 最大迭代次数 2000

初始仿真状态：θ₀ = 0（最低点），θ̇₀ = 0（静止）
"""

import time
import numpy as np

from pendulum import Pendulum
from grid import StateInputGrid
from cost_functions import QuadraticCost, MinimumTimeCost
from value_iteration import value_iteration
from simulate_and_plot import compare_experiments


def main():
    print("=" * 60)
    print("  基于价值迭代的倒立摆全局最优控制实验")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 初始化倒立摆模型
    # ------------------------------------------------------------------
    pendulum = Pendulum(m=1.0, l=1.0, g=9.81, b=0.1, dt=0.05)
    print(f"\n[系统] 倒立摆参数：m={pendulum.m} kg, l={pendulum.l} m, "
          f"g={pendulum.g} m/s², b={pendulum.b}, dt={pendulum.dt} s")

    # ------------------------------------------------------------------
    # 2. 定义状态-输入网格
    # ------------------------------------------------------------------
    grid = StateInputGrid(
        n_theta=51,          # θ 方向网格点数
        n_theta_dot=51,      # θ̇ 方向网格点数
        n_u=9,               # 输入离散点数（奇数方便包含 u=0）
        theta_lim=(-np.pi, np.pi),
        theta_dot_lim=(-8.0, 8.0),
        u_lim=(-3.0, 3.0)
    )
    print(f"[网格] θ: {grid.n_theta} 点, θ̇: {grid.n_theta_dot} 点, "
          f"u: {grid.n_u} 点（{grid.u_lim[0]} ~ {grid.u_lim[1]} N·m）")
    print(f"       总状态数: {grid.n_theta * grid.n_theta_dot}, "
          f"每状态输入数: {grid.n_u}")

    # ------------------------------------------------------------------
    # 3. 定义代价函数
    # ------------------------------------------------------------------
    # 3a. 二次型代价：平滑控制
    quadratic_cost = QuadraticCost(
        q_theta=1.0,        # 角度误差权重
        q_theta_dot=0.1,    # 角速度惩罚权重
        r_u=0.05,           # 控制量惩罚权重（较大 → 更平滑）
        theta_goal=np.pi    # 目标：直立位置
    )

    # 3b. 最短时间代价：砰-砰控制
    min_time_cost = MinimumTimeCost(
        theta_tol=0.3,      # 角度容差（rad）
        theta_dot_tol=1.5,  # 角速度容差（rad/s）
        theta_goal=np.pi    # 目标：直立位置
    )

    print("\n[代价函数]")
    print(f"  ① 二次型代价 (QuadraticCost)："
          f"q_θ={quadratic_cost.q_theta}, "
          f"q_θ̇={quadratic_cost.q_theta_dot}, "
          f"r_u={quadratic_cost.r_u}")
    print(f"  ② 最短时间代价 (MinimumTimeCost)："
          f"θ 容差={min_time_cost.theta_tol} rad, "
          f"θ̇ 容差={min_time_cost.theta_dot_tol} rad/s")

    # ------------------------------------------------------------------
    # 4. 执行价值迭代（两种代价函数）
    # ------------------------------------------------------------------
    results = {}

    for name, cost_fn, label in [
        ("quadratic", quadratic_cost, "Quadratic Cost (Smooth)"),
        ("min_time",  min_time_cost,  "Minimum Time (Bang-Bang)"),
    ]:
        print(f"\n{'─' * 50}")
        print(f"[实验] {label}")
        print(f"{'─' * 50}")
        t0 = time.time()

        J, policy, iters = value_iteration(
            pendulum=pendulum,
            grid=grid,
            cost_fn=cost_fn,
            gamma=0.999,
            eps=1e-3,
            max_iter=2000,
            verbose=True
        )

        elapsed = time.time() - t0
        print(f"[完成] 耗时 {elapsed:.1f} s，迭代 {iters} 次")
        print(f"       J* 范围: [{J.min():.4f}, {J.max():.4f}]")
        print(f"       π* 力矩范围: [{policy.min():.2f}, {policy.max():.2f}] N·m")

        results[name] = {"J": J, "policy": policy, "label": label}

    # ------------------------------------------------------------------
    # 5. 仿真并生成可视化图表
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("[可视化] 生成价值函数曲面图、策略热力图和仿真轨迹图 ...")
    print(f"{'─' * 50}")

    compare_experiments(
        results=results,
        pendulum=pendulum,
        grid=grid,
        theta0=0.0,        # 初始：最低点
        theta_dot0=0.0,    # 初始：静止
        n_steps=300,       # 仿真 300 步 = 15 s
        save_prefix="result"
    )

    # ------------------------------------------------------------------
    # 6. 打印实验总结
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  实验完成！生成图片清单：")
    print("  ├── result_quadratic_value_function.png   (二次型代价 - 价值函数)")
    print("  ├── result_quadratic_policy.png           (二次型代价 - 最优策略)")
    print("  ├── result_quadratic_trajectory.png       (二次型代价 - 仿真轨迹)")
    print("  ├── result_min_time_value_function.png    (最短时间代价 - 价值函数)")
    print("  ├── result_min_time_policy.png            (最短时间代价 - 最优策略)")
    print("  ├── result_min_time_trajectory.png        (最短时间代价 - 仿真轨迹)")
    print("  └── result_policy_comparison.png          (两种策略并排对比)")
    print("=" * 60)
    print()
    print("观察要点：")
    print("  • 二次型代价策略热力图：颜色平滑渐变 → 平滑控制（Smooth Control）")
    print("  • 最短时间代价策略热力图：颜色呈现明显阶跃断层 → 砰-砰控制（Bang-Bang）")
    print("  • 两种控制策略在倒立摆摆起轨迹上的明显差异")


if __name__ == "__main__":
    main()
