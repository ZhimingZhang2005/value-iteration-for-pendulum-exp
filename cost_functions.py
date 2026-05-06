"""
cost_functions.py
=================
代价函数（Running Cost）的定义。

提供两种典型代价函数：

1. **二次型代价（Quadratic Cost）**
   g(x, u) = (θ - θ_goal)² · q_theta
            + θ̇² · q_theta_dot
            + u² · r_u
   适合"平滑控制"实验：对控制量大小加以惩罚，生成连续渐变的策略。

2. **最短时间代价（Minimum Time Cost）**
   g(x, u) = 0  若状态落在目标邻域内
           = 1  否则
   适合"砰-砰控制"实验：系统会以最快速度冲向目标点，
   控制量总取边界值（±u_max）。

目标点
------
默认目标点为倒立摆直立位置：θ = π（或 -π），θ̇ = 0。

用法示例
--------
    from cost_functions import QuadraticCost, MinimumTimeCost
    qc = QuadraticCost(q_theta=1.0, q_theta_dot=0.1, r_u=0.01)
    mt = MinimumTimeCost(theta_tol=0.2, theta_dot_tol=1.0)
    cost = qc(theta, theta_dot, u)
"""

import numpy as np
from pendulum import wrap_angle


class QuadraticCost:
    """
    二次型代价函数（Quadratic / LQR-like Running Cost）。

    代价公式：
        g(θ, θ̇, u) = q_theta  · (θ - θ_goal)²
                    + q_theta_dot · θ̇²
                    + r_u         · u²

    角度差值使用 wrap_angle 折叠到 (-π, π] 以避免 2π 跳变问题。

    参数
    ----
    q_theta     : θ 误差权重，默认 1.0
    q_theta_dot : θ̇ 惩罚权重，默认 0.1
    r_u         : 控制量惩罚权重，默认 0.01
    theta_goal  : 目标摆角（rad），默认 np.pi（倒立点）
    """

    def __init__(self, q_theta: float = 1.0,
                 q_theta_dot: float = 0.1,
                 r_u: float = 0.01,
                 theta_goal: float = np.pi):
        self.q_theta = q_theta
        self.q_theta_dot = q_theta_dot
        self.r_u = r_u
        self.theta_goal = theta_goal

    def __call__(self, theta: float, theta_dot: float, u: float) -> float:
        """
        计算给定状态和控制量下的即时代价。

        参数
        ----
        theta     : 当前摆角（rad）
        theta_dot : 当前角速度（rad/s）
        u         : 控制力矩（N·m）

        返回
        ----
        cost : 即时代价（标量，≥ 0）
        """
        # 角度误差折叠到 (-π, π]，避免 θ 跨越 ±π 时的不连续性
        angle_err = wrap_angle(theta - self.theta_goal)
        cost = (self.q_theta     * angle_err ** 2 +
                self.q_theta_dot * theta_dot ** 2 +
                self.r_u         * u ** 2)
        return float(cost)


class MinimumTimeCost:
    """
    最短时间代价函数（Minimum Time / Bang-Bang Cost）。

    代价公式：
        g(θ, θ̇, u) = 0  若 |θ - θ_goal| ≤ theta_tol 且 |θ̇| ≤ theta_dot_tol
                   = 1  否则

    此代价函数鼓励系统以最快速度到达目标邻域，通常产生"砰-砰"控制策略
    （控制量总取最大或最小边界值）。

    参数
    ----
    theta_tol     : 角度容差（rad），默认 0.2
    theta_dot_tol : 角速度容差（rad/s），默认 1.0
    theta_goal    : 目标摆角（rad），默认 np.pi（倒立点）
    """

    def __init__(self, theta_tol: float = 0.2,
                 theta_dot_tol: float = 1.0,
                 theta_goal: float = np.pi):
        self.theta_tol = theta_tol
        self.theta_dot_tol = theta_dot_tol
        self.theta_goal = theta_goal

    def at_goal(self, theta: float, theta_dot: float) -> bool:
        """
        判断当前状态是否在目标邻域内。

        参数
        ----
        theta     : 当前摆角（rad）
        theta_dot : 当前角速度（rad/s）

        返回
        ----
        bool : True 表示已到达目标邻域
        """
        angle_err = abs(wrap_angle(theta - self.theta_goal))
        return (angle_err <= self.theta_tol and
                abs(theta_dot) <= self.theta_dot_tol)

    def __call__(self, theta: float, theta_dot: float, u: float) -> float:
        """
        计算给定状态和控制量下的即时代价。

        参数
        ----
        theta     : 当前摆角（rad）
        theta_dot : 当前角速度（rad/s）
        u         : 控制力矩（N·m）（此代价函数不对 u 惩罚）

        返回
        ----
        cost : 0.0（目标邻域内）或 1.0（目标邻域外）
        """
        return 0.0 if self.at_goal(theta, theta_dot) else 1.0
