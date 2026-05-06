"""
value_iteration.py
==================
离散状态/输入空间上的价值迭代（Value Iteration）核心算法。

算法描述
--------
在离散化的状态网格和输入网格上，反复应用贝尔曼最优算子：

    J_{k+1}(x_i) = min_u [ g(x_i, u) + γ · J_k(f(x_i, u)) ]

直到价值函数在所有网格点上的最大变化量 ‖J_{k+1} - J_k‖_∞ < ε 为止。

收敛后通过贪心策略提取最优控制策略表（lookup table）：

    π*(x_i) = argmin_u [ g(x_i, u) + γ · J*(f(x_i, u)) ]

输出
----
- J*  : 最优价值函数，形状 (n_theta, n_theta_dot)
- π*  : 最优策略表，形状 (n_theta, n_theta_dot)，值为控制力矩（N·m）

关键超参数
----------
gamma  : 折扣因子，越接近 1 越重视远期代价（默认 0.999）
eps    : 收敛判断阈值（默认 1e-3）
max_iter : 最大迭代次数（默认 2000），防止不收敛时死循环
"""

import numpy as np
from pendulum import Pendulum
from grid import StateInputGrid


def value_iteration(pendulum: Pendulum,
                    grid: StateInputGrid,
                    cost_fn,
                    gamma: float = 0.999,
                    eps: float = 1e-3,
                    max_iter: int = 2000,
                    verbose: bool = True):
    """
    在离散状态-输入网格上执行价值迭代，返回最优价值函数和最优策略。

    参数
    ----
    pendulum  : Pendulum 实例，提供 step 方法
    grid      : StateInputGrid 实例，定义网格结构
    cost_fn   : 可调用对象，签名 cost_fn(theta, theta_dot, u) -> float
    gamma     : 折扣因子（0 < γ ≤ 1），默认 0.999
    eps       : 收敛阈值，‖ΔJ‖_∞ < eps 时停止，默认 1e-3
    max_iter  : 最大迭代轮次，默认 2000
    verbose   : 是否打印迭代进度，默认 True

    返回
    ----
    J      : 最优价值函数数组，形状 (n_theta, n_theta_dot)
    policy : 最优策略数组（控制力矩），形状 (n_theta, n_theta_dot)
    iters  : 实际迭代次数
    """
    n_th = grid.n_theta
    n_thd = grid.n_theta_dot

    # ----------------------------------------------------------------
    # 步骤 1：预计算状态转移表（transition table）
    # ----------------------------------------------------------------
    # next_state[i, j, k] = (theta_next, theta_dot_next)
    # 在全部 (状态, 输入) 组合下只计算一次，大幅加速贝尔曼更新。
    if verbose:
        print("预计算状态转移表 ...", flush=True)

    # 存储下一状态的连续物理值（用于插值）
    next_theta    = np.zeros((n_th, n_thd, grid.n_u))
    next_theta_dot = np.zeros((n_th, n_thd, grid.n_u))
    running_cost  = np.zeros((n_th, n_thd, grid.n_u))

    for i in range(n_th):
        for j in range(n_thd):
            theta, theta_dot = grid.index_to_state(i, j)
            for k, u in enumerate(grid.us):
                th_next, thd_next = pendulum.step(theta, theta_dot, u)
                next_theta[i, j, k]     = th_next
                next_theta_dot[i, j, k] = thd_next
                running_cost[i, j, k]   = cost_fn(theta, theta_dot, u)

    if verbose:
        print("状态转移表计算完毕。开始价值迭代 ...", flush=True)

    # ----------------------------------------------------------------
    # 步骤 2：初始化价值函数
    # ----------------------------------------------------------------
    J = np.zeros((n_th, n_thd))

    # ----------------------------------------------------------------
    # 步骤 3：迭代贝尔曼更新
    # ----------------------------------------------------------------
    for iteration in range(max_iter):
        J_new = np.full((n_th, n_thd), np.inf)

        for i in range(n_th):
            for j in range(n_thd):
                best_val = np.inf
                for k in range(grid.n_u):
                    # 用双线性插值查询下一状态的价值
                    v_next = grid.interpolate_value(
                        J,
                        next_theta[i, j, k],
                        next_theta_dot[i, j, k]
                    )
                    val = running_cost[i, j, k] + gamma * v_next
                    if val < best_val:
                        best_val = val
                J_new[i, j] = best_val

        # 收敛判断：最大绝对差值
        delta = np.max(np.abs(J_new - J))
        J = J_new

        if verbose and (iteration % 100 == 0 or delta < eps):
            print(f"  迭代 {iteration:4d}，最大变化量 δ = {delta:.6f}")

        if delta < eps:
            if verbose:
                print(f"价值迭代收敛！共迭代 {iteration + 1} 次。")
            break
    else:
        if verbose:
            print(f"已达最大迭代次数 {max_iter}，当前 δ = {delta:.6f}（未完全收敛）。")

    # ----------------------------------------------------------------
    # 步骤 4：提取最优策略（贪心策略）
    # ----------------------------------------------------------------
    policy = _extract_policy(J, grid, next_theta, next_theta_dot,
                              running_cost, gamma)

    return J, policy, iteration + 1


def _extract_policy(J: np.ndarray,
                    grid: StateInputGrid,
                    next_theta: np.ndarray,
                    next_theta_dot: np.ndarray,
                    running_cost: np.ndarray,
                    gamma: float) -> np.ndarray:
    """
    根据收敛的价值函数 J，通过一步贪心搜索提取最优策略表。

    返回
    ----
    policy : 最优控制力矩数组，形状 (n_theta, n_theta_dot)
    """
    n_th = grid.n_theta
    n_thd = grid.n_theta_dot
    policy = np.zeros((n_th, n_thd))

    for i in range(n_th):
        for j in range(n_thd):
            best_val = np.inf
            best_u = grid.us[0]
            for k, u in enumerate(grid.us):
                v_next = grid.interpolate_value(
                    J,
                    next_theta[i, j, k],
                    next_theta_dot[i, j, k]
                )
                val = running_cost[i, j, k] + gamma * v_next
                if val < best_val:
                    best_val = val
                    best_u = u
            policy[i, j] = best_u

    return policy


def make_policy_fn(policy: np.ndarray, grid: StateInputGrid):
    """
    将策略数组（lookup table）封装为可调用函数，
    接受连续状态 (theta, theta_dot) 并返回最近网格点的控制量。

    参数
    ----
    policy : 最优策略数组，形状 (n_theta, n_theta_dot)
    grid   : StateInputGrid 实例

    返回
    ----
    policy_fn : 函数 (theta, theta_dot) -> u
    """
    def policy_fn(theta: float, theta_dot: float) -> float:
        i, j = grid.state_to_index(theta, theta_dot)
        return float(policy[i, j])

    return policy_fn
