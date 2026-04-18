# A3 README：全连接网络与 CNN 笔记（重点突出 CNN 难点）

这份笔记是给 `D:\CS\UMICH\A3` 整理的复习版 README。
重点不是把所有内容平均展开，而是抓住：

1. **A3 在学什么**
2. **全连接网络和 CNN 的主线**
3. **CNN 实现里我最容易卡住的难点**
4. **写代码时最容易错的 shape / backward / 参数初始化问题**

如果时间不多，优先看：

- 第 2 部分：A3 总体主线
- 第 4 部分：CNN 原理
- 第 5 部分：CNN 实现难点
- 第 8 部分：一页速记版

---

# 1. A3 在做什么

A3 的核心其实是两条线：

## 第一条线：Fully Connected Networks

也就是普通多层感知机（MLP）。

你会实现：

- `Linear`
- `ReLU`
- `Linear_ReLU`
- `TwoLayerNet`
- `FullyConnectedNet`
- `Dropout`
- 优化器如 `sgd`、`sgd_momentum`、`adam`

这一部分重点是：

> **把前向传播、反向传播、参数更新整条链打通。**

## 第二条线：Convolutional Networks

也就是 CNN。

你会实现：

- `Conv.forward`
- `Conv.backward`
- `MaxPool.forward`
- `MaxPool.backward`
- `ThreeLayerConvNet`
- `DeepConvNet`

这一部分重点是：

> **理解卷积层到底怎么滑动、怎么算输出、梯度怎么回传。**

而你已经说了：

> **CNN 的实现是我的难点**

所以这份笔记会重点讲 CNN。

---

# 2. A3 的主线

A3 真正的主线不是某个函数，而是：

```text
输入 X
-> 前向传播算 scores
-> softmax loss
-> 反向传播算 grads
-> optimizer 更新参数
-> 重复训练
```

所以不管是全连接网络还是 CNN，本质都在做同一件事：

1. 定义网络结构
2. 做 forward
3. 算 loss
4. 做 backward
5. 更新参数

区别只在于：

- 全连接层：每个神经元看整个输入
- 卷积层：每个卷积核只看局部区域，并且参数共享

---

# 3. 全连接网络部分要抓住什么

这一部分相对比 CNN 更直一些，但它是 CNN 的基础。

---

## 3.1 `Linear.forward` 的本质

输入可能是：

```python
(N, d1, d2, ..., dk)
```

但线性层想要的是二维：

```python
(N, D)
```

所以要先 reshape：

```python
x_row = x.reshape(N, -1)
out = x_row.mm(w) + b
```

这里最重要的是：

> **线性层前向 = flatten + 矩阵乘法 + bias**

---

## 3.2 `Linear.backward` 的本质

如果前向是：

```python
out = x_row @ w + b
```

那么反向就是：

- `dx = dout @ w^T`
- `dw = x_row^T @ dout`
- `db = sum(dout, dim=0)`

再把 `dx` reshape 回原输入形状。

这里是整个反向传播最基础的模板，后面 CNN 的线性层部分完全一样。

---

## 3.3 `ReLU` 的本质

前向：

```python
out = max(0, x)
```

反向：

- `x > 0` 的地方梯度保留
- `x <= 0` 的地方梯度变 0

所以：

```python
dx = dout.clone()
dx[x <= 0] = 0
```

一句话：

> **ReLU backward 就是在用输入的正负做 mask。**

---

## 3.4 `TwoLayerNet` 和 `FullyConnectedNet`

### `TwoLayerNet`

结构：

```text
linear - relu - linear - softmax
```

这是最简单的分类网络。

### `FullyConnectedNet`

结构：

```text
{linear - relu - [dropout]} x (L - 1) - linear - softmax
```

这里真正重要的是：

- 学会把多层 forward 串起来
- 学会把多层 backward 倒着拆回来
- 学会统一管理 `self.params`

---

# 4. CNN 的原理

这一部分是你最需要真正吃透的。

---

## 4.1 CNN 为什么比全连接更适合图像

图像有两个特点：

1. **局部相关性强**
   - 一个像素最相关的是附近像素
2. **同一种模式会在不同位置出现**
   - 比如边缘、角点、纹理，不会只出现在固定坐标

如果用全连接层：

- 参数很多
- 完全忽略空间结构

CNN 的核心想法是：

> **用一个小卷积核在图像上滑动，检测局部模式，并且在所有位置共享同一组参数。**

这就是：

- 局部连接
- 参数共享

---

## 4.2 卷积层到底在做什么

假设输入是：

```python
x.shape = (N, C, H, W)
```

卷积核是：

```python
w.shape = (F, C, HH, WW)
```

其中：

- `N`：样本数
- `C`：通道数
- `H, W`：输入高宽
- `F`：滤波器个数
- `HH, WW`：卷积核高宽

每个卷积核会：

1. 覆盖输入中的一个局部窗口
2. 和该窗口逐元素相乘再求和
3. 加上偏置
4. 得到输出特征图中的一个数

一句话：

> **卷积就是“滑动窗口内做点积”。**

---

## 4.3 卷积输出尺寸怎么计算

这是 CNN 实现最核心的公式之一。

如果：

- 输入高宽是 `H, W`
- padding 是 `pad`
- stride 是 `stride`
- filter 大小是 `HH, WW`

那么输出高宽：

\[
H' = 1 + \frac{H + 2 \cdot pad - HH}{stride}
\]

\[
W' = 1 + \frac{W + 2 \cdot pad - WW}{stride}
\]

写代码时就是：

```python
H_out = 1 + (H + 2 * pad - HH) // stride
W_out = 1 + (W + 2 * pad - WW) // stride
```

这个公式一定要非常熟，因为：

- forward 要靠它开输出张量
- backward 也要靠它遍历每个输出位置
- 初始化全连接层输入维度也要靠它

---

## 4.4 Max Pooling 在干什么

max pooling 不是提取新特征，而是在做：

- 下采样
- 保留局部最强响应
- 减少空间尺寸

比如 `2x2` pooling, stride `2`：

- 每个 `2x2` 窗口只保留最大值
- 高宽大约减半

所以 pooling 的本质是：

> **压缩空间分辨率，保留最强激活。**

---

# 5. CNN 实现里我最容易卡住的难点

这一部分是重点中的重点。

---

## 难点 1：卷积层 shape 总是容易乱

卷积层输入是：

```python
(N, C, H, W)
```

卷积核是：

```python
(F, C, HH, WW)
```

输出是：

```python
(N, F, H_out, W_out)
```

最容易错的地方：

1. 把 `C` 和 `F` 搞混
2. 忘了每个 filter 会跨越所有输入通道 `C`
3. 忘了输出通道数是 `F`，不是 `C`

### 强记一句话

> **输入通道数决定 filter 厚度，filter 个数决定输出通道数。**

---

## 难点 2：卷积 forward 到底在滑什么

在你的代码里，卷积 forward 本质是：

```python
for n in range(N):
    for f in range(F):
        for i in range(H_out):
            for j in range(W_out):
                window = x_pad[n, :, hs:he, ws:we]
                out[n, f, i, j] = torch.sum(window * w[f]) + b[f]
```

### 这里真正发生了什么

对于固定的：

- 第 `n` 个样本
- 第 `f` 个卷积核
- 输出位置 `(i, j)`

会从输入中切出一个局部窗口：

```python
window.shape = (C, HH, WW)
```

然后和：

```python
w[f].shape = (C, HH, WW)
```

逐元素相乘求和。

所以：

> **一个输出位置 = 一个输入局部块 和 一个 filter 的点积。**

---

## 难点 3：为什么要 padding

padding 的作用主要有两个：

1. 不让输出空间尺寸缩得太快
2. 让边缘像素也能被充分卷积到

比如在 `ThreeLayerConvNet` 里：

```python
conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
```

这样当 filter 大小是奇数时，卷积后高宽不变。

例如 `7x7` filter：

```python
pad = (7 - 1) // 2 = 3
```

这就能保持输入输出空间尺寸相同。

### 一句话记忆

> **stride=1 且 pad=(K-1)/2 时，卷积通常保持高宽不变。**

---

## 难点 4：卷积 backward 最难理解

这是 CNN 最大难点之一。

卷积 backward 里要算：

- `dx`
- `dw`
- `db`

### 1）`db` 最简单

因为每个输出位置都会加同一个 bias，所以：

```python
db = torch.sum(dout, dim=(0, 2, 3))
```

即：

- 对 batch 维求和
- 对空间维求和
- 保留 filter 维

---

### 2）`dw` 怎么理解

每个卷积核参数 `w[f]` 会被很多输出位置反复使用。

所以它的梯度要把所有相关位置的贡献都加起来：

```python
dw[f] += x_pad[n, :, hs:he, ws:we] * dout[n, f, i, j]
```

直觉上：

> **某个 filter 参数对 loss 的影响，要把它在所有位置、所有样本上的使用贡献全部累加。**

---

### 3）`dx` 怎么理解

输入的某个像素，可能会被多个卷积窗口覆盖。

所以它的梯度也要把所有覆盖它的输出梯度都加回来：

```python
dx_pad[n, :, hs:he, ws:we] += w[f] * dout[n, f, i, j]
```

直觉上：

> **输入像素的梯度 = 所有“用过它”的卷积输出位置，把梯度反向分摊回来。**

这就是为什么卷积 backward 看起来像“把窗口里的梯度往回撒”。

---

## 难点 5：为什么 backward 里要对 padded input 求梯度

forward 用的是 `x_pad`，因为边界补了零。

所以 backward 时也要先对 `x_pad` 求梯度：

```python
dx_pad = torch.zeros_like(x_pad)
```

全部梯度算完后，再把中间对应原图的部分裁出来：

```python
dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
```

### 一句话记忆

> **forward 是在 padded 图上卷，backward 也要先回传到 padded 图，再裁回原图。**

---

## 难点 6：MaxPool backward 为什么只传给最大值位置

max pooling forward 是：

- 取窗口里的最大值

所以 backward 时，只有那个最大值真正影响了输出。

因此梯度只给它：

```python
mask = (window == m)
dx[n, c, hs:he, ws:we] += mask * dout[n, c, i, j]
```

这就是 max pool backward 的本质。

### 一个需要注意的小点

如果窗口里有多个相同最大值，这种实现会把梯度分给所有最大值位置。

在这份作业里通常是可以接受的。

---

## 难点 7：ThreeLayerConvNet 为什么 `W2` 维度这样算

`ThreeLayerConvNet` 结构是：

```text
conv - relu - 2x2 max pool - linear - relu - linear - softmax
```

在你的代码里初始化：

- 卷积后高宽不变
- pooling 后高宽减半

所以如果输入是 `(C, H, W)`，卷积后是：

```python
(num_filters, H, W)
```

pooling 后是：

```python
(num_filters, H/2, W/2)
```

展平后长度就是：

```python
num_filters * (H // 2) * (W // 2)
```

所以：

```python
self.params['W2']
```

的第一维必须是这个值。

这一步如果算错，后面 Linear 层就会 shape mismatch。

---

## 难点 8：为什么 ConvNet 的 forward 也能接 `Linear_ReLU`

因为经过 conv + pool 之后，输出还是一个 tensor，形状大概是：

```python
(N, F, H', W')
```

而你前面实现过的 `Linear.forward` 会自动：

```python
x.reshape(N, -1)
```

所以它能把卷积输出直接 flatten 后接到全连接层。

也就是说：

> **CNN 接全连接层，本质就是先 flatten。**

---

## 难点 9：DeepConvNet 到底比 ThreeLayerConvNet 多了什么

`ThreeLayerConvNet` 是固定结构：

```text
conv - relu - pool - linear - relu - linear
```

而 `DeepConvNet` 是可扩展版：

```text
{conv - [batchnorm] - relu - [pool]} x (L - 1) - linear
```

关键变化：

1. 可以有很多个卷积层
2. 某些层才做 pooling
3. 可以选择是否加 batchnorm
4. 参数初始化要能自动按层数处理

所以 `DeepConvNet` 的难点不在卷积公式，而在：

> **如何系统化管理多层参数和 shape 变化。**

---

# 6. A3 里最容易错的代码点

---

## 6.1 L2 regularization 的系数别写错

在不同类里，正则项有时写法不同，作业说明会强调：

- 有的地方不要乘 `0.5`
- 有的地方要乘 `0.5`

一定要跟注释要求保持一致，否则梯度检查可能不过。

---

## 6.2 bias 的梯度经常忘了 sum

无论在线性层还是卷积层，bias 梯度本质都是把对应维度上的 `dout` 累加。

- Linear：

```python
db = torch.sum(dout, dim=0)
```

- Conv：

```python
db = torch.sum(dout, dim=(0, 2, 3))
```

---

## 6.3 `dx` 最后要 reshape / crop 回去

- Linear backward：要 reshape 回原输入形状
- Conv backward：要从 `dx_pad` 裁回 `dx`

这两种“回原形”都很容易忘。

---

## 6.4 Pooling 不要和 convolution 混

卷积层：

- 有参数 `w, b`
- 会学习

Pooling 层：

- 没有参数
- 只是做固定的空间下采样

很多时候脑子里会把二者混在一起，但实际上它们作用不同。

---

# 7. 学 A3 的正确心智模型

如果你学到一半会乱，建议用下面这个方式理解。

---

## 7.1 对全连接网络

你只需要牢牢记住：

```text
Linear.forward
Linear.backward
ReLU.forward
ReLU.backward
softmax_loss
```

多层网络只是这些模块的反复堆叠。

---

## 7.2 对 CNN

你只需要牢牢记住：

### 卷积 forward

> 切窗口 -> 和 filter 点积 -> 加 bias

### 卷积 backward

> 把每个输出梯度分别贡献回输入窗口和 filter 参数

### pooling forward

> 每个窗口取最大值

### pooling backward

> 梯度只回给最大值位置

如果这四句话能自己讲清楚，CNN 的核心就已经通了。

---

# 8. 一页速记版

## 8.1 A3 主线

```text
X -> forward -> scores -> softmax loss -> backward -> grads -> update
```

---

## 8.2 Fully Connected 核心

### Linear

```text
forward: flatten + xW + b
backward: dx = dout W^T, dw = x^T dout, db = sum(dout)
```

### ReLU

```text
forward: max(0, x)
backward: x<=0 的地方梯度为 0
```

### FullyConnectedNet

```text
{linear - relu - [dropout]} x (L-1) - linear - softmax
```

---

## 8.3 CNN 核心

### Conv 输入输出

```text
x: (N, C, H, W)
w: (F, C, HH, WW)
out: (N, F, H_out, W_out)
```

### Conv 输出尺寸

\[
H' = 1 + \frac{H + 2pad - HH}{stride}
\]

\[
W' = 1 + \frac{W + 2pad - WW}{stride}
\]

### Conv forward

```text
滑动窗口和 filter 做点积
```

### Conv backward

```text
db: 对 dout 求和
dw: 把每个窗口 * 对应 dout 累加
dx: 把 dout 通过 filter 分摊回输入窗口
```

### MaxPool forward

```text
每个窗口取最大值
```

### MaxPool backward

```text
梯度只传给最大值位置
```

---

## 8.4 你最该重点盯住的 5 个点

1. 卷积层 shape：`(N, C, H, W)`、`(F, C, HH, WW)`、`(N, F, H_out, W_out)`
2. 输出尺寸公式
3. 卷积 backward 里 `dx / dw / db` 的来源
4. pooling backward 为什么只给最大值
5. `ThreeLayerConvNet` 中卷积后接全连接时的 flatten 维度

---

# 9. 最后一句总结

A3 里真正最重要的不是把每段代码背下来，而是把下面这条线真正想通：

> **全连接网络是在学模块化 forward / backward，而 CNN 是把这种 forward / backward 思想扩展到“局部窗口 + 参数共享 + 空间结构”上。**

对你来说，CNN 的难点本质上不是公式多，而是：

> **你要同时盯住窗口、filter、输出位置、shape 变化和梯度回传路径。**

一旦这几件事能在脑子里对应起来，A3 就顺了。

