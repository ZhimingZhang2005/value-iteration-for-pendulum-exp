"""
pendulum.py
===========
倒立摆（Simple Pendulum）的动力学建模与仿真步进。

系统动力学方程（连续时间）：
    ml²·θ̈ = u - mgl·sin(θ) - b·θ̇

其中：
    θ   : 摆角（弧度），θ=0 对应最低点，θ=±π 对应最高点（倒立点）
    θ̇  : 角速度（rad/s）
    u   : 施加的力矩（N·m）
    m   : 摆球质量（kg）
    l   : 摆长（m）
    g   : 重力加速度（m/s²）
    b   : 阻尼系数（N·m·s/rad）

离散化方式：欧拉前向积分（Forward Euler）
    θ_{k+1}  = θ_k  + dt·θ̇_k
    θ̇_{k+1} = θ̇_k + dt·θ̈_k
"""

import numpy as np


class Pendulum:
    """
    简单倒立摆动力学模型。

    参数
    ----
    m : float
        摆球质量（kg），默认 1.0
    l : float
        摆长（m），默认 1.0
    g : float
        重力加速度（m/s²），默认 9.81
    b : float
        阻尼系数（N·m·s/rad），默认 0.1
    dt : float
        仿真时间步长（s），默认 0.05
    """

    def __init__(self, m: float = 1.0, l: float = 1.0,
                 g: float = 9.81, b: float = 0.1, dt: float = 0.05):
        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.dt = dt

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def dynamics(self, theta: float, theta_dot: float, u: float):
        """
        计算连续时间动力学的加速度 θ̈。

        参数
        ----
        theta     : 当前摆角（rad）
        theta_dot : 当前角速度（rad/s）
        u         : 施加的控制力矩（N·m）

        返回
        ----
        theta_ddot : 角加速度（rad/s²）
        """
        m, l, g, b = self.m, self.l, self.g, self.b
        # ml²·θ̈ = u - mgl·sin(θ) - b·θ̇
        theta_ddot = (u - m * g * l * np.sin(theta) - b * theta_dot) / (m * l ** 2)
        return theta_ddot

    def step(self, theta: float, theta_dot: float, u: float):
        """
        向前积分一个时间步（Forward Euler），返回下一时刻状态。

        参数
        ----
        theta     : 当前摆角（rad）
        theta_dot : 当前角速度（rad/s）
        u         : 控制力矩（N·m）

        返回
        ----
        theta_next     : 下一时刻摆角（rad），已折叠到 (-π, π]
        theta_dot_next : 下一时刻角速度（rad/s）
        """
        theta_ddot = self.dynamics(theta, theta_dot, u)

        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_ddot

        # 将角度折叠到 (-π, π]，方便后续与目标点（θ=π）比较
        theta_next = wrap_angle(theta_next)

        return theta_next, theta_dot_next

    def simulate(self, theta0: float, theta_dot0: float,
                 policy, n_steps: int):
        """
        给定初始状态和策略，仿真 n_steps 步，记录完整轨迹。

        参数
        ----
        theta0     : 初始摆角（rad）
        theta_dot0 : 初始角速度（rad/s）
        policy     : 可调用对象，接受 (theta, theta_dot) 返回控制量 u
        n_steps    : 仿真步数

        返回
        ----
        thetas     : 摆角序列，形状 (n_steps+1,)
        theta_dots : 角速度序列，形状 (n_steps+1,)
        us         : 控制序列，形状 (n_steps,)
        """
        thetas = np.zeros(n_steps + 1)
        theta_dots = np.zeros(n_steps + 1)
        us = np.zeros(n_steps)

        thetas[0] = theta0
        theta_dots[0] = theta_dot0

        for k in range(n_steps):
            u = policy(thetas[k], theta_dots[k])
            us[k] = u
            thetas[k + 1], theta_dots[k + 1] = self.step(thetas[k], theta_dots[k], u)

        return thetas, theta_dots, us


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def wrap_angle(theta: float) -> float:
    """
    将角度折叠到 (-π, π]。

    参数
    ----
    theta : 任意角度（rad）

    返回
    ----
    折叠后的角度（rad）
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi
