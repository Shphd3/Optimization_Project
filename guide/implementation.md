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

### Python 代码参考

```python
def forward(x_t, beta, Q0, Alpha, L0, C0_all, Wq, We, Rsrp, Conn, Capa, ALL_Users_Traffic, ALL_Cells_Bw):
    # 1. RSRP + CIO
    rsrp_cio = Rsrp + x_t[:, np.newaxis]
    
    # 2. Softmax Probability (with stability fix)
    rsrp_norm = rsrp_cio - rsrp_cio.max(axis=0)
    exp_rsrp = np.exp(rsrp_norm * beta) * Conn
    sum_exp = exp_rsrp.sum(axis=0, keepdims=True)
    sum_exp[sum_exp == 0] = 1e-9
    Prob = exp_rsrp / sum_exp

    # 3. Load
    # L_m_abs: Absolute bandwidth usage
    L_m_abs = np.sum((ALL_Users_Traffic * Prob) / Capa, axis=1)
    # l_m_ratio: Bandwidth utilization ratio
    l_m_ratio = L_m_abs / ALL_Cells_Bw

    # 4. Costs
    qos_cost_m = func_Q(l_m_ratio, ALL_Cells_Bw, Q0, L0, Alpha)
    energy_cost_m = func_E(l_m_ratio, ALL_Cells_Bw, C0_all)

    # Total Objective
    obj = np.sum(Wq * qos_cost_m + We * energy_cost_m)

    cache = {
        'Prob': Prob,
        'Load_m': l_m_ratio,
        'Unit_Cost': Wq * qos_cost_m + We * energy_cost_m
    }
    return obj, cache
```

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

### Python 代码参考

```python
def backward(x_t, cache, M, N, beta, Q0, Alpha, L0, C0_all, Wq, We, Conn, Capa, ALL_Users_Traffic, ALL_Cells_Bw):
    Prob = cache['Prob']
    l_m_ratio = cache['Load_m']

    # 1. Marginal Cost w.r.t Load Ratio (l_m)
    # Note: The code calculates derivative w.r.t l_m, then converts to L_m implicitly during U_mn calculation
    grad_Q = grad_Q_to_l(l_m_ratio, ALL_Cells_Bw, Q0, L0, Alpha)
    grad_E = grad_E_to_l(l_m_ratio, ALL_Cells_Bw, C0_all)
    dJ_dl = Wq * grad_Q + We * grad_E  # Shape (M,)

    # 2. Marginal Cost U_mn = dJ/dl_m * dl_m/dProb_mn
    # dl_m/dProb_mn = A_n / (Capa_mn * B_m)
    U_mn = dJ_dl[:, np.newaxis] * ALL_Users_Traffic / (Capa * ALL_Cells_Bw[:, np.newaxis])

    # 3. Gradient w.r.t X_i
    avg_U_n = np.sum(Prob * U_mn, axis=0) # Shape (N,)
    grad_obj_to_x = beta * np.sum(Prob * (U_mn - avg_U_n), axis=1) # Shape (M,)

    return grad_obj_to_x
```

---

## 3. 优化器集成

计算出梯度 `grad` 后，可以使用以下逻辑更新 $X$：

- **SGD**: $X_{new} = X_{old} - \eta \cdot grad$
- **Adam**:
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
  $$X_{new} = X_{old} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
