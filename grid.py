"""
grid.py
=======
状态空间和输入空间的网格划分工具。

功能
----
1. 定义 θ（摆角）、θ̇（角速度）和 u（控制力矩）的离散网格。
2. 提供网格索引与物理值之间的双向转换。
3. 支持 θ 的周期性边界处理（摆角在 (-π, π] 范围内循环）。
4. 提供邻近网格点插值，用于价值函数查询。

典型用法
--------
    grid = StateInputGrid(n_theta=51, n_theta_dot=51, n_u=9,
                          theta_lim=(-np.pi, np.pi),
                          theta_dot_lim=(-8.0, 8.0),
                          u_lim=(-3.0, 3.0))
    # 索引 → 物理值
    theta, theta_dot = grid.index_to_state(i, j)
    # 物理值 → 最近索引
    i, j = grid.state_to_index(theta, theta_dot)
    # 双线性插值查询价值
    v = grid.interpolate_value(J, theta, theta_dot)
"""

import numpy as np


class StateInputGrid:
    """
    二维状态网格 (θ, θ̇) 与一维输入网格 (u) 的统一管理。

    参数
    ----
    n_theta     : θ 方向网格点数（含两端端点）
    n_theta_dot : θ̇ 方向网格点数（含两端端点）
    n_u         : 输入 u 的离散点数（含两端端点）
    theta_lim   : θ 取值范围，默认 (-π, π)
    theta_dot_lim : θ̇ 取值范围，默认 (-8.0, 8.0) rad/s
    u_lim       : u 取值范围，默认 (-3.0, 3.0) N·m
    """

    def __init__(self,
                 n_theta: int = 51,
                 n_theta_dot: int = 51,
                 n_u: int = 9,
                 theta_lim: tuple = (-np.pi, np.pi),
                 theta_dot_lim: tuple = (-8.0, 8.0),
                 u_lim: tuple = (-3.0, 3.0)):

        self.n_theta = n_theta
        self.n_theta_dot = n_theta_dot
        self.n_u = n_u
        self.theta_lim = theta_lim
        self.theta_dot_lim = theta_dot_lim
        self.u_lim = u_lim

        # 生成各轴的物理值数组
        self.thetas = np.linspace(theta_lim[0], theta_lim[1], n_theta)
        self.theta_dots = np.linspace(theta_dot_lim[0], theta_dot_lim[1], n_theta_dot)
        self.us = np.linspace(u_lim[0], u_lim[1], n_u)

        # 步长（用于插值）
        self.dtheta = self.thetas[1] - self.thetas[0]
        self.dtheta_dot = self.theta_dots[1] - self.theta_dots[0]

    # ------------------------------------------------------------------
    # 索引 ↔ 物理值 转换
    # ------------------------------------------------------------------

    def index_to_state(self, i: int, j: int):
        """
        将网格索引 (i, j) 转换为对应的物理状态 (θ, θ̇)。

        参数
        ----
        i : θ 方向索引，范围 [0, n_theta-1]
        j : θ̇ 方向索引，范围 [0, n_theta_dot-1]

        返回
        ----
        theta     : 摆角（rad）
        theta_dot : 角速度（rad/s）
        """
        return self.thetas[i], self.theta_dots[j]

    def state_to_index(self, theta: float, theta_dot: float):
        """
        将物理状态 (θ, θ̇) 映射到最近的网格索引 (i, j)。
        超出范围的状态会被 clip 到边界索引。

        参数
        ----
        theta     : 摆角（rad）
        theta_dot : 角速度（rad/s）

        返回
        ----
        i : θ 方向索引
        j : θ̇ 方向索引
        """
        i = int(np.round((theta - self.theta_lim[0]) / self.dtheta))
        j = int(np.round((theta_dot - self.theta_dot_lim[0]) / self.dtheta_dot))
        i = np.clip(i, 0, self.n_theta - 1)
        j = np.clip(j, 0, self.n_theta_dot - 1)
        return i, j

    def u_index_to_value(self, k: int) -> float:
        """将输入索引 k 转换为对应的物理力矩值。"""
        return self.us[k]

    # ------------------------------------------------------------------
    # 双线性插值：在任意状态处查询价值函数
    # ------------------------------------------------------------------

    def interpolate_value(self, J: np.ndarray,
                          theta: float, theta_dot: float) -> float:
        """
        对价值函数数组 J（形状 n_theta × n_theta_dot）做双线性插值，
        返回物理状态 (theta, theta_dot) 处的估计价值。

        超出网格边界的状态被裁剪到最边缘格点。

        参数
        ----
        J         : 价值函数数组，形状 (n_theta, n_theta_dot)
        theta     : 摆角（rad）
        theta_dot : 角速度（rad/s）

        返回
        ----
        value : 插值得到的价值估计
        """
        # θ 方向连续坐标（格点单位）
        fi = (theta - self.theta_lim[0]) / self.dtheta
        fj = (theta_dot - self.theta_dot_lim[0]) / self.dtheta_dot

        # 边界裁剪
        fi = np.clip(fi, 0, self.n_theta - 1)
        fj = np.clip(fj, 0, self.n_theta_dot - 1)

        # 取整数部分和小数部分
        i0 = int(fi)
        j0 = int(fj)
        i1 = min(i0 + 1, self.n_theta - 1)
        j1 = min(j0 + 1, self.n_theta_dot - 1)

        alpha = fi - i0   # θ 方向权重
        beta = fj - j0    # θ̇ 方向权重

        # 双线性插值
        value = ((1 - alpha) * (1 - beta) * J[i0, j0] +
                 alpha       * (1 - beta) * J[i1, j0] +
                 (1 - alpha) * beta       * J[i0, j1] +
                 alpha       * beta       * J[i1, j1])
        return float(value)

    # ------------------------------------------------------------------
    # 辅助：生成网格 meshgrid（用于绘图）
    # ------------------------------------------------------------------

    def meshgrid(self):
        """
        返回适合 matplotlib 3D 绘图的 meshgrid。

        返回
        ----
        TH  : θ 的 meshgrid，形状 (n_theta, n_theta_dot)
        THD : θ̇ 的 meshgrid，形状 (n_theta, n_theta_dot)
        """
        TH, THD = np.meshgrid(self.thetas, self.theta_dots, indexing='ij')
        return TH, THD
