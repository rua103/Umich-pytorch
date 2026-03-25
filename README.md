# EECS 498-007 / 598-005: Deep Learning for Computer Vision

> University of Michigan — 作业代码仓库

本仓库包含课程全部作业（A1–A6）及 Mini Project 的实现代码，使用 **PyTorch** 完成。

---

## 目录结构

```
UMICH/
├── A1/           K-Nearest Neighbor & PyTorch 入门
├── A2/           线性分类器（SVM / Softmax）& 两层网络
├── A3/           全连接网络 & 卷积神经网络
├── A4/           目标检测（单阶段 & 双阶段）
├── A5/           RNN / LSTM / Transformer 图像描述
├── A6/           生成模型（GAN / VAE / 风格迁移 / 网络可视化）
└── Mini_Project/ 课程 Mini Project
```

---

## 各作业简介

### A1 — K-Nearest Neighbor & PyTorch 101

| 文件 | 内容 |
|------|------|
| `knn.py` | KNN 分类器：计算 L2 距离（两层循环 / 单层循环 / 向量化），交叉验证选 K |
| `pytorch101.py` | PyTorch 基础操作练习 |

**核心知识点：**
- L2 距离的向量化计算（广播 + 矩阵运算）
- KNN 的时间复杂度与向量化加速
- PyTorch Tensor 基本操作

---

### A2 — 线性分类器 & 两层网络

| 文件 | 内容 |
|------|------|
| `linear_classifier.py` | SVM Hinge Loss（naive + vectorized）、Softmax Loss（naive + vectorized）、SGD 训练循环 |
| `two_layer_net.py` | 两层全连接网络前向传播、反向传播（手动推导梯度） |

**核心知识点：**
- Structured SVM：Hinge Loss + Mask 梯度
- Softmax：数值稳定、交叉熵、`prob - onehot` 梯度
- 手动反向传播（链式法则）
- SGD + minibatch 训练

> 详细推导见 [`A2/README.md`](A2/README.md)

---

### A3 — 全连接网络 & 卷积神经网络

| 文件 | 内容 |
|------|------|
| `fully_connected_networks.py` | 任意深度全连接网络、Dropout、BatchNorm、各种优化器（SGD/Adam/RMSProp） |
| `convolutional_networks.py` | 卷积层、池化层、空间BatchNorm、CNN 分类器 |

**核心知识点：**
- 模块化网络设计（Layer 抽象）
- Dropout / BatchNorm 的前向与反向
- Adam、RMSProp 优化器实现
- 卷积的 naive 与 fast 实现

---

### A4 — 目标检测

| 文件 | 内容 |
|------|------|
| `one_stage_detector.py` | FCOS 单阶段检测器：Anchor-free，回归中心点偏移 + 类别 |
| `two_stage_detector.py` | Faster R-CNN 双阶段检测器：RPN + RoI Pooling + 分类头 |

**核心知识点：**
- Anchor-free 检测（FCOS）
- Region Proposal Network (RPN)
- IoU 计算、NMS 后处理
- 多任务损失（分类 + 回归）

---

### A5 — 序列模型 & 图像描述

| 文件 | 内容 |
|------|------|
| `rnn_lstm_captioning.py` | RNN / LSTM 手动实现，图像描述（Image Captioning） |
| `transformers.py` | Self-Attention、Multi-Head Attention、Transformer Decoder |

**核心知识点：**
- RNN / LSTM 前向与反向（BPTT）
- Attention 机制与 Transformer
- 序列生成（Greedy / Beam Search）
- 视觉特征与语言模型结合

---

### A6 — 生成模型

| 文件 | 内容 |
|------|------|
| `gan.py` | GAN 训练：判别器 + 生成器损失，DCGAN |
| `vae.py` | 变分自编码器：ELBO、重参数化技巧 |
| `style_transfer.py` | 神经风格迁移：Gram Matrix、内容损失 + 风格损失 |
| `network_visualization.py` | Saliency Map、Fooling Image、Class Visualization |

**核心知识点：**
- GAN 的 minimax 训练稳定性
- VAE 的 KL 散度 + 重参数化
- Gram Matrix 与风格表示
- 梯度上升可视化

---

### Mini Project

课程自选项目，详见 [`Mini_Project/`](Mini_Project/) 目录。

---

## 环境配置

```bash
# 推荐使用 conda 或 venv
pip install torch torchvision
pip install jupyter matplotlib numpy
```

> 所有实现块均禁止使用 `.to()` / `.cuda()`，设备管理由框架统一处理。

---

## 学习路线建议

```
A1（基础）→ A2（线性模型 + 手动梯度）→ A3（深层网络 + 优化器）
         → A4（检测）→ A5（序列模型）→ A6（生成模型）
```

| 作业 | 难度 | 重点能力 |
|------|------|----------|
| A1 | ★☆☆☆☆ | 向量化思维、PyTorch 基础 |
| A2 | ★★☆☆☆ | 手动梯度推导、Loss 设计 |
| A3 | ★★★☆☆ | 模块化、优化器、BatchNorm |
| A4 | ★★★★☆ | 检测框架、多任务损失 |
| A5 | ★★★★☆ | 序列建模、Attention |
| A6 | ★★★★★ | 生成模型、高级可视化 |

