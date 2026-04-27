# EECS 498-007 / 598-005: Deep Learning for Computer Vision

> University of Michigan — 作业代码仓库总览

本仓库包含课程全部作业（A1–A6）及 Mini Project 的实现代码，主目录 `README.md` 用于记录**整体大纲 / 学习路线 / 每个子作业笔记入口**；更细的推导、难点、易错点放在各子目录自己的 README 里。

---

## 目录结构

```text
UMICH/
├── A1/           K-Nearest Neighbor & PyTorch 入门
├── A2/           线性分类器（SVM / Softmax）& 两层网络
├── A3/           全连接网络 & 卷积神经网络
├── 2020FA_A4/    Autograd、CNN、ResNet、可视化、风格迁移、Captioning
├── A5/           RNN / LSTM / Transformer 图像描述
├── A6/           生成模型（VAE / CVAE / GAN / DCGAN）
└── Mini_Project/ 课程 Mini Project
```

---

## 仓库使用说明

本文件只做三件事：

1. 说明每个作业在学什么
2. 给出复习主线和学习路线
3. 链接到各子作业自己的笔记 README

如果要看：

- 数学推导
- 代码实现细节
- 难点总结
- 易错点清单

请优先进入对应作业文件夹查看各自 README。

---

## 各作业概览与笔记入口

### A1 — K-Nearest Neighbor & PyTorch 101

**主题：**
- KNN 分类器
- PyTorch Tensor 基础操作
- 向量化思维入门

**核心能力：**
- L2 距离的向量化计算
- 广播与矩阵运算
- 张量索引、切片与 reshape
- 从循环实现过渡到批量张量实现

**主要文件：**
- `A1/knn.py`
- `A1/pytorch101.py`

**子作业笔记：**
- `A1/README.md`

**这份笔记重点覆盖：**
- KNN 从双循环到零循环的实现主线
- `topk` 投票与交叉验证选 `k`
- PyTorch 基础索引、布尔掩码与 reshape
- batched matrix multiply 与向量化思维

---

### A2 — 线性分类器 & 两层网络

**主题：**
- Structured SVM
- Softmax
- Two-Layer Network
- SGD 训练

**核心能力：**
- Loss 函数设计
- 手动梯度推导
- 向量化实现
- 基础神经网络训练流程

**主要文件：**
- `A2/linear_classifier.py`
- `A2/two_layer_net.py`

**子作业笔记：**
- `A2/README.md`

**这份笔记重点覆盖：**
- SVM 向量化实现
- Softmax 数值稳定
- mask 梯度写法
- 两层网络训练主线

---

### A3 — 全连接网络 & 卷积神经网络

**主题：**
- 多层全连接网络
- Dropout / BatchNorm
- 优化器
- CNN 前向与反向

**核心能力：**
- 模块化搭建深层网络
- forward / backward 串联
- 卷积与池化的 shape 推导
- 优化器实现与训练稳定性

**主要文件：**
- `A3/fully_connected_networks.py`
- `A3/convolutional_networks.py`

**子作业笔记：**
- `A3/README.md`

**这份笔记重点覆盖：**
- A3 总体主线
- CNN 原理
- CNN 实现难点
- backward 和 shape 易错点

---

### 2020FA_A4 — Autograd、CNN、ResNet、可视化、风格迁移与 Captioning

**主题：**
- PyTorch Autograd
- CNN / `nn.Module` / `nn.Sequential`
- ResNet
- Network Visualization
- Style Transfer
- RNN / LSTM / Attention Captioning

**核心能力：**
- 理解动态计算图与自动求导
- 从裸张量实现过渡到模块化搭建 CNN
- 掌握残差网络的结构设计
- 理解 saliency / adversarial / class visualization 的输入梯度用法
- 在特征空间中定义内容损失与风格损失
- 理解图像描述中的序列建模与注意力机制

**主要文件：**
- `2020FA_A4/pytorch_autograd_and_nn.py`
- `2020FA_A4/network_visualization.py`
- `2020FA_A4/style_transfer.py`
- `2020FA_A4/rnn_lstm_attention_captioning.py`

**子作业笔记：**
- `2020FA_A4/README.md`

**这份笔记重点覆盖：**
- Autograd 与 CNN 实现主线
- ResNet 结构理解
- 网络可视化与对输入求梯度
- Style Transfer 的损失设计
- 从视觉模型过渡到 Captioning

---

### A5 — 序列模型 & 图像描述 / Transformer

**主题：**
- RNN / LSTM
- Attention
- Transformer
- 图像描述（Image Captioning）

**核心能力：**
- 序列建模
- Teacher Forcing / 推理
- Self-Attention
- Encoder-Decoder Transformer

**主要文件：**
- `A5/rnn_lstm_captioning.py`
- `A5/transformers.py`

**子作业笔记：**
- `A5/README_Transformer_A5_笔记.md`

**这份笔记重点覆盖：**
- Transformer 原理
- 训练 / 推理主线
- Multi-Head Attention
- Positional Encoding
- 个人难点整理

---

### A6 — 生成模型

**主题：**
- VAE / CVAE
- GAN / LSGAN
- DCGAN

**核心能力：**
- 概率生成模型理解
- 重参数化技巧
- ELBO / KL 散度
- 对抗训练与稳定性
- 卷积式生成模型

**主要文件：**
- `A6/vae.py`
- `A6/gan.py`
- `A6/variational_autoencoders.ipynb`
- `A6/generative_adversarial_networks.ipynb`

**子作业笔记：**
- `A6/README.md`

**这份笔记重点覆盖：**
- VAE / CVAE 实现
- GAN / LSGAN / DCGAN 实现
- 数学原理总结
- 难点、易错点与复习清单

---

### Mini Project

课程自选项目。

**目录：**
- `Mini_Project/`

---

## 学习路线建议

推荐按下面的顺序复习：

```text
A1 → A2 → A3 → A4 → A5 → A6
```

对应能力递进：

1. **A1**：先建立向量化思维和 PyTorch 基础
2. **A2**：学会 loss、梯度、最基本训练流程
3. **A3**：学会深层网络模块化，以及 CNN 的核心机制
4. **2020FA_A4**：理解 autograd、CNN/ResNet、可视化与 style transfer，并接触 captioning
5. **A5**：系统学习序列建模与 Transformer
6. **A6**：进入生成模型，理解更高级的建模思想

---

## 各作业复习定位

| 作业 | 主题 | 难度 | 复习关键词 |
|------|------|------|------------|
| A1 | KNN / PyTorch 入门 | ★☆☆☆☆ | 向量化、广播、L2 距离 |
| A2 | 线性分类器 / 两层网络 | ★★☆☆☆ | SVM、Softmax、手动梯度、SGD |
| A3 | MLP / CNN | ★★★☆☆ | 模块化、Dropout、BatchNorm、卷积 backward |
| 2020FA_A4 | Autograd / CNN / ResNet / Visualization / Style Transfer / Captioning | ★★★★☆ | 动态计算图、残差连接、输入梯度、风格损失、Attention |
| A5 | RNN / LSTM / Transformer | ★★★★☆ | Attention、Mask、Positional Encoding、解码 |
| A6 | Generative Models | ★★★★★ | ELBO、KL、Reparameterization、GAN 稳定性 |

---

## README 导航表

为了以后查笔记更快，这里做一个统一入口：

| 目录 | 笔记文件 | 状态 | 主要内容 |
|------|-----------|------|----------|
| `A1` | `A1/README.md` | 已整理 | KNN、PyTorch 基础、向量化 |
| `A2` | `A2/README.md` | 已整理 | SVM、Softmax、Two-Layer Net |
| `A3` | `A3/README.md` | 已整理 | FC Net、CNN、实现难点 |
| `2020FA_A4` | `2020FA_A4/README.md` | 已整理 | Autograd、ResNet、可视化、风格迁移 |
| `A5` | `A5/README_Transformer_A5_笔记.md` | 已整理 | Transformer 原理与难点 |
| `A6` | `A6/README.md` | 已整理 | VAE / GAN / DCGAN 笔记 |
