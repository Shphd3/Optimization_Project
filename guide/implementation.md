# Forward 与 Backward 实现指南

本指南详细介绍了项目中 `forward` 和 `backward` 过程的逻辑实现，这是传统优化算法（如 Adam）和深度学习模型的基础。

## 1. Forward 过程 (前向传播)

前向传播的主要任务是计算当前决策变量 $X$ 下的系统总成本，并缓存中间结果以供梯度计算使用。

### 算法步骤：
1. **计算偏置后的 RSRP**: $Rsrp\_cio_{mn} = Rsrp_{mn} + X_m$。
2. **计算分配概率 (Softmax)**:
   - 为防止溢出，进行归一化：$Rsrp\_norm_{mn} = Rsrp\_cio_{mn} - \max_k(Rsrp\_cio_{kn})$。
   - 计算 $Prob_{mn} = \frac{\exp(Rsrp\_norm_{mn} \cdot \beta) \cdot Conn_{mn}}{\sum_k \exp(Rsrp\_norm_{kn} \cdot \beta) \cdot Conn_{kn}}$。
3. **计算负载**:
   - 每个小区的带宽需求 $L_m = \sum_n \frac{A_n \cdot Prob_{mn}}{Capa_{mn}}$。
   - 带宽利用率 $l_m = L_m / B_m$。
4. **计算成本**:
   - 计算各小区的 QoS 成本和能源成本。
   - 汇总得到总目标函数值 $obj$。
5. **缓存结果**: 存储 `Prob`, `Load_m`, `L_m` 等变量。

---

## 2. Backward 过程 (反向传播)

反向传播利用链式法则计算目标函数对 $X$ 的梯度 $\nabla_X J$。

### 核心梯度公式推导：

定义 **边际成本 (Marginal Cost)** $U_{mn}$ 为将用户 $n$ 的单位概率分配给小区 $m$ 所带来的成本变化：
$$U_{mn} = \frac{\partial J}{\partial L_m} \cdot \frac{\partial L_m}{\partial Prob_{mn}} = \frac{\partial J}{\partial L_m} \cdot \frac{A_n}{Capa_{mn}}$$

其中，系统总成本对小区 $m$ 消耗带宽 $L_m$ 的导数为：
$$\frac{\partial J}{\partial L_m} = W_q \cdot \left( Q_0 e^{\alpha(l_m - L_0)} (1 + \alpha l_m) \right) + W_e \cdot \left( 2 \cdot C_{0,m} \cdot L_m \right)$$

最终，目标函数对小区 $i$ 的偏置因子 $X_i$ 的梯度为：
$$\frac{\partial J}{\partial X_i} = \beta \cdot \sum_{n \in \mathcal{N}} Prob_{in} \left( U_{in} - \sum_{m \in \mathcal{M}} Prob_{mn} U_{mn} \right)$$

### 实现技巧：
- **期望边际成本**: 先计算每个用户 $n$ 的期望边际成本 $\bar{U}_n = \sum_m Prob_{mn} U_{mn}$。
- **矢量化计算**: 利用 NumPy 的矩阵运算，避免显式的 Python 循环，显著提升性能。
- **邻区加速**: 在计算 $\sum_m$ 时，仅考虑 `NeighCell` 中标记的邻近小区，因为非邻区的 $Prob_{mn}$ 几乎为 0。

---

## 3. 优化器集成

计算出梯度 `grad` 后，可以使用以下逻辑更新 $X$：

- **SGD**: $X_{new} = X_{old} - \eta \cdot grad$
- **Adam**:
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
  $$X_{new} = X_{old} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
