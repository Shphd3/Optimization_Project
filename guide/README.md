# AI for Network Optimization 项目指南

本项目专注于通过优化小区独立偏置因子（CIO）来实现无线通信网络中的负载均衡。

## 目录

1. [数学模型详细描述](./math_model.md)
   - 涵盖信号强度、信道容量、流量分配及成本函数的完整数学定义。

2. [Forward 与 Backward 实现指南](./implementation.md)
   - 详细解释了前向计算逻辑与基于边际成本的梯度推导过程。

3. [LLM 优化策略指南](./llm_strategy.md)
   - 探讨如何将 LLM 作为智能控制器、规则生成器或通过微调成为神经优化器。

## 快速开始

- 核心代码与仿真环境请参考项目根目录下的 `Project_AI_for_Network_Optimization.ipynb`。
- 本项目包含完整的场景生成、传统优化算法框架以及 AI 优化接口。

## 项目背景

在 LTE/NR 网络中，单纯依靠信号强度（RSRP）选择小区会导致负载不均衡。本项目通过引入可优化的 CIO 参数，利用梯度下降或 AI 方法（如 GNN, DRL）动态调整用户接入策略，从而在降低能耗的同时保证服务质量（QoS）。
