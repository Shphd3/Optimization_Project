# AI for Network Optimization 项目指南

本项目专注于通过优化小区独立偏置因子（CIO）来实现无线通信网络中的负载均衡。

在环境中除了基本的依赖之外还需要安装：

`pip install ffmpeg`

## 目录

1. [数学模型详细描述](./math_model.md)
   - 涵盖信号强度、信道容量、流量分配及成本函数的完整数学定义。

2. [Forward 与 Backward 实现指南](./implementation.md)
   - 详细解释了前向计算逻辑与基于边际成本的梯度推导过程。
   - **新增**: 包含完整的 Python 参考代码实现。

3. [LLM 优化策略指南](./llm_strategy.md)
   - 探讨如何将 LLM 作为智能控制器、规则生成器或通过微调成为神经优化器。

## 快速开始

- **核心代码**: 项目根目录下的 `Project_AI_for_Network_Optimization.ipynb` 已经包含完整的场景生成和**已实现的传统优化算法**（Forward, Backward, Adam）。
- **运行方式**: 直接打开 Jupyter Notebook 运行所有单元格即可观察优化效果。
- **扩展开发**: 可参考 `implementation.md` 中的代码逻辑，尝试替换为其他优化器或修改成本函数。

## 项目背景

在 LTE/NR 网络中，单纯依靠信号强度（RSRP）选择小区会导致负载不均衡。本项目通过引入可优化的 CIO 参数，利用梯度下降或 AI 方法（如 GNN, DRL）动态调整用户接入策略，从而在降低能耗的同时保证服务质量（QoS）。


写在后面：
## **注意**

直接运行Project_AI_for_Network_Optimization.ipynb就完事儿了！