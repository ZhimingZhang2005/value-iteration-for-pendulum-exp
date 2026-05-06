# 基于价值迭代的倒立摆全局最优控制

**Value Iteration for Inverted Pendulum — Global Optimal Control**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 本项目针对经典欠驱动非线性系统——**简单倒立摆（Simple Pendulum）**，基于**价值迭代（Value Iteration）**动态规划算法，在离散状态-输入网格上求解全局最优反馈控制策略，并系统对比**砰-砰控制（Bang-Bang Control）**与**平滑控制（Smooth Control）**两种典型控制行为。代码结构清晰，一键运行 `main.py` 即可复现所有实验结果。

---

## 目录

- [项目简介](#项目简介)
- [主要功能与亮点](#主要功能与亮点)
- [运行环境与依赖](#运行环境与依赖)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [核心算法说明](#核心算法说明)
- [实验结果示例](#实验结果示例)
- [致谢与引用](#致谢与引用)

---

## 项目简介

倒立摆（Inverted Pendulum）是非线性控制与动态规划领域的经典基准问题。传统线性控制器（如 LQR）只能在平衡点附近实现局部稳定，无法处理大范围的 swing-up 控制任务。

本项目通过**动态规划 / 价值迭代**方法：

1. 将连续的二维状态空间 $(\theta,\, \dot{\theta})$ 和一维输入空间 $u$ 离散为均匀网格；
2. 在网格上反复应用 **Bellman 最优性方程**，迭代计算全局最优代价函数 $J^*(x)$；
3. 依据 $J^*$ 提取每个网格点上的最优反馈策略 $\pi^*(x)$；
4. 对比两种代价函数下的控制行为差异：
   - **最短时间代价（Minimum Time）** → 产生典型的 **Bang-Bang 控制**（控制量在 $\pm u_{\max}$ 之间急剧切换）；
   - **二次型代价（Quadratic Cost）** → 产生**平滑控制**（控制量随状态连续渐变）。

该实验参考了 MIT《欠驱动机器人学（Underactuated Robotics）》课程内容，完全基于 **NumPy + Matplotlib** 实现，无需安装 Drake 等重型依赖。

---

## 主要功能与亮点

| 功能 | 说明 |
|------|------|
| **状态空间网格化** | $\theta \in [-\pi, \pi]$ 和 $\dot\theta \in [-8, 8]\ \text{rad/s}$ 均匀离散，支持周期边界条件 |
| **输入空间网格化** | 控制力矩 $u \in [-u_{\max}, u_{\max}]$ 离散为 $N_u$ 个点（含 $u=0$） |
| **预计算状态转移表** | 提前计算全部 $(i,j,k)$ 对应的下一状态，避免内层循环重复积分，显著加速迭代 |
| **双线性插值** | 下一状态不在网格点时，通过双线性插值获取 $J$ 值，减少离散化误差 |
| **两种代价函数** | `QuadraticCost`（平滑）与 `MinimumTimeCost`（Bang-Bang），可一行切换 |
| **一键复现** | `main.py` 串联所有模块，依次执行两套实验并保存全部图表至当前目录 |
| **丰富可视化** | 价值函数 3-D 曲面图、策略热力图、仿真轨迹曲线、两种策略并排对比图，共 7 张 PNG |

---

## 运行环境与依赖

- **Python** ≥ 3.8
- **NumPy** ≥ 1.21
- **Matplotlib** ≥ 3.4

标准库（`time`、`math`）无需额外安装。

完整依赖见 [`requirements.txt`](requirements.txt)：

```
numpy>=1.21
matplotlib>=3.4
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/ZhimingZhang2005/value-iteration-for-pendulum-exp.git
cd value-iteration-for-pendulum-exp
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行主程序

```bash
python main.py
```

运行后控制台将打印每次迭代的收敛信息，完成后在当前目录生成以下 7 张图片：

```
result_quadratic_value_function.png   — 二次型代价：价值函数 3-D 曲面
result_quadratic_policy.png           — 二次型代价：最优策略热力图（平滑渐变）
result_quadratic_trajectory.png       — 二次型代价：仿真轨迹（平滑控制）
result_min_time_value_function.png    — 最短时间代价：价值函数 3-D 曲面
result_min_time_policy.png            — 最短时间代价：最优策略热力图（Bang-Bang）
result_min_time_trajectory.png        — 最短时间代价：仿真轨迹（砰-砰控制）
result_policy_comparison.png          — 两种策略并排对比
```

### 4. 切换或自定义代价函数

打开 `main.py`，在"3. 定义代价函数"一节修改参数即可：

```python
# 二次型代价 —— 调整权重影响平滑程度
quadratic_cost = QuadraticCost(
    q_theta=1.0,       # 角度误差权重（越大越快收敛到目标角度）
    q_theta_dot=0.1,   # 角速度惩罚权重
    r_u=0.05,          # 控制量惩罚权重（越大 → 控制越平滑，力矩越小）
    theta_goal=np.pi
)

# 最短时间代价 —— 调整容差影响目标邻域大小
min_time_cost = MinimumTimeCost(
    theta_tol=0.3,     # 角度容差（rad），越大越容易触发"到达"
    theta_dot_tol=1.5, # 角速度容差（rad/s）
    theta_goal=np.pi
)
```

若只想运行单一代价函数实验，在 `main.py` 中注释掉对应的循环项即可。

### 5. 调整网格分辨率

在 `main.py` 的"2. 定义状态-输入网格"中修改：

```python
grid = StateInputGrid(
    n_theta=51,       # θ 方向网格点数（越大精度越高，但速度越慢）
    n_theta_dot=51,   # θ̇ 方向网格点数
    n_u=9,            # 控制输入离散点数
    theta_lim=(-np.pi, np.pi),
    theta_dot_lim=(-8.0, 8.0),
    u_lim=(-3.0, 3.0)
)
```

---

## 目录结构

```
value-iteration-for-pendulum-exp/
├── pendulum.py           # 倒立摆动力学模型（Forward Euler 离散化）
├── grid.py               # 状态-输入空间网格化工具（坐标↔索引互转、双线性插值）
├── cost_functions.py     # 代价函数：QuadraticCost / MinimumTimeCost
├── value_iteration.py    # 价值迭代主循环与最优策略提取
├── simulate_and_plot.py  # 策略仿真与可视化（价值函数曲面、热力图、轨迹图）
├── main.py               # 主入口：串联所有模块，一键复现实验
├── requirements.txt      # Python 依赖列表
└── README.md             # 本文件
```

各模块职责说明：

| 文件 | 核心职责 |
|------|----------|
| `pendulum.py` | `Pendulum` 类：`dynamics()`、`step()`、`simulate()`；`wrap_angle()` 辅助函数 |
| `grid.py` | `StateInputGrid` 类：网格坐标生成、`index_to_state()`、`state_to_index()`、`interpolate_value()` |
| `cost_functions.py` | `QuadraticCost`、`MinimumTimeCost`，均实现 `__call__(theta, theta_dot, u) -> float` 接口 |
| `value_iteration.py` | `value_iteration()`：预计算转移表 → Bellman 迭代 → 返回 $(J^*, \pi^*, \text{iters})$ |
| `simulate_and_plot.py` | `compare_experiments()`：对两套结果调用仿真并输出所有图表 |
| `main.py` | 参数配置入口，组装上述模块并运行完整实验流程 |

---

## 核心算法说明

### 系统动力学

倒立摆非线性动力学方程（连续时间）：

$$ml^2\ddot{\theta} + b\dot{\theta} + mgl\sin\theta = u$$

采用 **Forward Euler** 方法离散化（步长 $dt = 0.05\ \text{s}$）：

$$\theta_{k+1} = \theta_k + dt \cdot \dot{\theta}_k, \qquad \dot{\theta}_{k+1} = \dot{\theta}_k + dt \cdot \ddot{\theta}_k$$

### Bellman 最优性方程（离散）

$$J_{k+1}(x) = \min_{u \in \mathcal{U}} \bigl[g(x, u) + \gamma\, J_k\bigl(f(x, u)\bigr)\bigr]$$

- $g(x, u)$：即时代价（见下方代价函数说明）
- $\gamma = 0.999$：折扣因子
- $f(x, u)$：下一状态（由 `pendulum.step()` 给出）
- 收敛条件：$\max_{x} |J_{k+1}(x) - J_k(x)| < \varepsilon$（默认 $\varepsilon = 10^{-3}$）

### 两种代价函数对比

| 代价函数 | 公式 | 控制行为 |
|----------|------|----------|
| **二次型代价** | $g = q_\theta(\theta-\pi)^2 + q_{\dot\theta}\dot{\theta}^2 + r_u u^2$ | **平滑控制**：控制量随状态连续渐变 |
| **最短时间代价** | $g = 0$（在目标邻域内），$g = 1$（否则） | **Bang-Bang 控制**：控制量总取 $\pm u_{\max}$ |

### 策略提取

价值函数收敛后，对每个网格点提取贪心策略：

$$\pi^*(x) = \arg\min_{u \in \mathcal{U}} \bigl[g(x, u) + \gamma\, J^*\bigl(f(x, u)\bigr)\bigr]$$

---

## 实验结果示例

运行 `python main.py` 后，在当前目录可得到以下图表（示例描述）：

### 最优策略热力图（`result_*_policy.png`）

- **二次型代价**：热力图颜色从深蓝（$-u_{\max}$）到深红（$+u_{\max}$）**平滑渐变**，控制策略连续。
- **最短时间代价**：热力图仅出现深蓝与深红两种颜色，边界处颜色**骤然跳变**——典型的 Bang-Bang 特征。

### 价值函数曲面（`result_*_value_function.png`）

- 以 $(\theta, \dot\theta)$ 为横纵轴绘制 $J^*$ 的三维曲面，目标点 $(\pi, 0)$ 处代价最低（谷底）。

### 仿真轨迹（`result_*_trajectory.png`）

从初始静止最低点 $(\theta_0, \dot\theta_0) = (0, 0)$ 出发，仿真 300 步（15 s）：

| 对比项 | 二次型代价（平滑） | 最短时间代价（Bang-Bang） |
|--------|-------------------|--------------------------|
| 控制输入 $u(t)$ | 随状态连续变化 | 在 $\pm 3\ \text{N·m}$ 之间频繁切换 |
| 摆起时间 | 较长，但能量消耗更少 | 更短，但力矩反复跳变 |
| 稳定性 | 靠近目标点后平稳维持 | 到达目标邻域后保持 |

### 并排对比图（`result_policy_comparison.png`）

将两套最优策略热力图并排显示，直观呈现"渐变 vs. 跳变"的本质差异。

---

## 致谢与引用

本项目基于以下资料和课程开展：

1. **MIT Underactuated Robotics**  
   Russ Tedrake. *Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation.*  
   Course Notes, MIT CSAIL. [https://underactuated.mit.edu](https://underactuated.mit.edu)  
   > 本项目的算法框架（网格化动态规划、价值迭代、倒立摆 swing-up）直接参考了该课程第 Chapter 8（Dynamic Programming）及配套 Notebook `dp/on_a_mesh.ipynb`。

2. **强化学习经典教材**  
   Richard S. Sutton & Andrew G. Barto. *Reinforcement Learning: An Introduction (2nd ed.).* MIT Press, 2018.  
   [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

3. **最优控制经典教材**  
   Dimitri P. Bertsekas. *Dynamic Programming and Optimal Control (Vol. I & II, 4th ed.).* Athena Scientific, 2017.

4. **Bang-Bang 控制理论**  
   L. S. Pontryagin et al. *The Mathematical Theory of Optimal Processes.* Wiley-Interscience, 1962.

---

## License

本项目采用 [MIT License](LICENSE) 开源协议。