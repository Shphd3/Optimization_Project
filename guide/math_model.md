# 项目数学模型详细描述

本项目旨在通过优化小区偏置因子（CIO）来解决无线网络中的负载不均衡问题。以下是该项目的数学建模详细描述。

## 1. 基础参数定义

- **小区集合**: $\mathcal{M} = \{1, 2, \dots, M\}$，数量 $M=100$。
- **用户集合**: $\mathcal{N} = \{1, 2, \dots, N\}$，数量 $N=600$。
- **带宽资源**: 每个小区 $m$ 的总带宽为 $B_m$。
- **用户流量需求**: 每个用户 $n$ 的平均到达数据量为 $A_n$。
- **距离矩阵**: $Dist_{mn}$ 表示用户 $n$ 与小区 $m$ 之间的欧几里得距离。

## 2. 物理层模型

### 2.1 信号强度 (RSRP)
采用典型城区路径衰减模型：
$$Rsrp_{mn} = -65 - 37.6 \cdot \log_{10}\left(\frac{Dist_{mn}}{50}\right) \text{ (dBm)}$$

### 2.2 信道容量 (Capacity)
基于香农公式计算单位带宽信道容量：
$$SINR_{mn} = Rsrp_{mn} - Noise\_Int$$
$$Capa_{mn} = \log_2(1 + 10^{SINR_{mn}/10}) \text{ (bit/s/Hz)}$$
其中 $Noise\_Int = -100 \text{ dBm}$。

### 2.3 关联矩阵 (Connectivity)
只有当距离小于阈值且接入概率不低于阈值时，用户才能接入小区：
1. **初始关联**: 
   $$Conn_{mn} = \begin{cases} 1, & \text{if } Dist_{mn} \le 500m \\ 0, & \text{otherwise} \end{cases}$$
2. **基于概率的修剪**: 
   在计算初始分配概率 $Prob_{mn}$ 后，移除小概率链路以降低复杂度：
   $$Conn_{mn} = 0, \text{ if } Prob_{mn} < 0.05$$
   然后重新归一化计算最终的 $Prob_{mn}$。

## 3. 流量分配模型 (Softmax 引导)

引入决策变量 **CIO** ($X_m$) 来调整接入优先级。

### 3.1 流量分配概率
用户 $n$ 接入小区 $m$ 的概率为：
$$Prob_{mn} = \frac{\exp((Rsrp_{mn} + X_m) \cdot \beta) \cdot Conn_{mn}}{\sum_{k \in \mathcal{M}} \exp((Rsrp_{kn} + X_k) \cdot \beta) \cdot Conn_{kn}}$$
其中 $\beta = 0.2$ 是概率分散度调控因子。

### 3.2 小区负载与利用率
- **流入流量**: $\Lambda_{mn} = A_n \cdot Prob_{mn}$，其中 $A_n \in [3, 6]$。
- **消耗带宽资源**: $L_m = \sum_{n \in \mathcal{N}} \frac{\Lambda_{mn}}{Capa_{mn}}$
- **带宽利用率 (Load)**: $l_m = \frac{L_m}{B_m}$，其中 $B_m = 10$。

## 4. 目标函数 (Cost Function)

系统总成本 $J(X)$ 由两部分组成：

$$J(X) = \sum_{m \in \mathcal{M}} \left( W_q \cdot QoS\_Degradation_m + W_e \cdot Energy\_Cost_m \right)$$

### 4.1 QoS 下降成本
当负载超过阈值 $L_0$ 时，服务质量下降。公式如下：
$$QoS\_Degradation_m = Q_0 \cdot \exp(\alpha \cdot (l_m - L_0)) \cdot L_m$$
参数设置：$Q_0 = 0.5, \alpha = 20, L_0 = 0.8$。

### 4.2 能源成本
采用准二次函数（基于消耗的总带宽资源）：
$$Energy\_Cost_m = C_{0,m} \cdot (L_m)^2$$
其中 $C_{0,m}$ 根据小区编号分布在 $[0.5, 1.5]$ 之间。

## 5. 优化问题定义

$$\min_{X \in \mathbb{R}^M} J(X)$$
目标是通过调整 $X = [X_1, X_2, \dots, X_M]$，在满足用户需求的同时，使系统总成本最小。
