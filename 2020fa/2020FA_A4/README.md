# A4 课程作业总结：PyTorch Autograd、CNN、ResNet、网络可视化与风格迁移

## 概述

本次 Assignment 4 不是几个彼此独立的小题，而是一条非常清晰的技术主线。整个作业从最底层的张量运算与自动求导出发，逐步过渡到卷积神经网络的模块化实现，再扩展到残差网络、输入梯度可视化，以及基于特征空间优化的风格迁移。

如果从整体上理解，A4 主要完成了以下五个层次的学习：

1. 理解 PyTorch 的动态计算图和自动求导机制。
2. 从裸张量运算到 `nn.Module`、`nn.Sequential`，掌握 CNN 的不同实现层级。
3. 通过手写 ResNet 理解深层网络的结构设计与残差连接原理。
4. 通过 saliency map、adversarial attack、class visualization 理解“对输入求梯度”的意义。
5. 通过 style transfer 理解如何在预训练网络的特征空间中定义损失并直接优化图像。

整份作业的核心思想可以概括为：

> 神经网络不仅可以作为分类器使用，也可以被看作一个可微的函数和一个强大的特征提取器。一旦掌握了自动求导，就可以优化参数、分析中间表示，甚至直接优化输入本身。

---

## 1. PyTorch 自动求导的底层原理

### 1.1 动态计算图

A4 中很多任务虽然表面形式不同，但底层都依赖同一个机制：自动求导。

在 PyTorch 中，只要一个张量设置了 `requires_grad=True`，并且它参与了后续的可微运算，那么这些运算就会被记录在一张动态计算图中。前向传播不仅是在计算结果，同时也在记录“这个结果是如何一步步得到的”。这样当我们最终对一个标量调用 `backward()` 时，PyTorch 就能自动沿着计算图反向传播梯度。

例如：

```python
scores = model(X)
loss = scores.gather(1, y.view(-1, 1)).sum()
loss.backward()
```

这里实际发生的过程是：

1. `model(X)` 建立从输入 `X` 到输出 `scores` 的计算图。
2. `gather` 从所有类别分数中抽取目标类别对应的分数。
3. `sum()` 把多个样本的目标分数合并成一个标量。
4. `backward()` 从这个标量开始，将梯度自动传播到图中的各个叶子节点。

### 1.2 梯度保存在哪里

对于叶子张量，梯度会存放在 `.grad` 属性中。

例如：

- 参数的梯度保存在 `weight.grad`、`bias.grad`
- 输入图像的梯度保存在 `X.grad`

因此，在本次作业后半部分中，很多代码结构都遵循如下模式：

```python
X.requires_grad_()
loss.backward()
print(X.grad)
```

这说明梯度不一定只能用来更新模型参数，也可以直接用来分析和更新输入。

### 1.3 为什么更新时使用 `with torch.no_grad()`

在对输入图像进行优化时，比如生成对抗样本或做类可视化，我们往往会在得到梯度后直接更新输入值。此时推荐的写法是：

```python
loss.backward()
with torch.no_grad():
    x += step
```

这是因为：

- 前向与反向传播阶段需要 autograd 记录计算图。
- 更新步骤本身不应该被继续加入图中。
- `with torch.no_grad()` 比直接使用 `.data` 更安全，也更符合当前 PyTorch 的推荐风格。

因此，本次作业也帮助建立了一个重要习惯：

> 正常构图并计算梯度，更新时关闭梯度跟踪。

---

## 2. 从裸张量实现三层卷积网络

`pytorch_autograd_and_nn.py` 的前半部分要求在较低抽象层面实现一个三层卷积网络，其目的不是“写出最简洁的代码”，而是让我们真正理解卷积网络在底层是如何由张量运算组成的。

### 2.1 网络结构

该三层卷积网络结构如下：

1. `Conv2d(3 -> 32, 5x5, padding=2)`
2. `ReLU`
3. `Conv2d(32 -> 16, 3x3, padding=1)`
4. `ReLU`
5. `Linear(16 * 32 * 32 -> 10)`

其函数式实现形式为：

```python
x = F.relu(F.conv2d(x, conv_w1, conv_b1, padding=2))
x = F.relu(F.conv2d(x, conv_w2, conv_b2, padding=1))
scores = F.linear(flatten(x), fc_w, fc_b)
```

这个实现方式让我们非常明确地看到：

- 卷积本质上是一个带权重和偏置的局部线性运算；
- ReLU 只是一个逐元素非线性函数；
- 全连接层本质上是矩阵乘法加偏置；
- 整个网络是多个可微算子的组合。

### 2.2 张量形状的理解

这一部分最核心的能力之一，就是准确理解每个张量的形状。

#### 卷积层权重形状

PyTorch 中卷积层的权重张量形状为：

```python
(out_channels, in_channels, KH, KW)
```

因此第一层卷积的权重形状是：

```python
(32, 3, 5, 5)
```

第二层卷积的权重形状是：

```python
(16, 32, 3, 3)
```

#### 全连接层输入输出形状

对于 `F.linear(input, weight, bias)`：

- `input` 的形状为 `(N, Din)`
- `weight` 的形状为 `(Dout, Din)`
- `bias` 的形状为 `(Dout,)`

在本题中，由于卷积后特征图大小仍然是 `32 x 32`，并且输出通道数为 16，所以全连接层输入维度应为：

```python
16 * 32 * 32
```

### 2.3 `flatten` 的作用

卷积层输出通常为四维张量 `(N, C, H, W)`，而线性层要求输入是二维 `(N, D)`。因此必须在卷积部分结束后将空间维和通道维展开：

```python
flatten(x)
```

这个操作并不改变数据内容，只是改变了张量视角，使其能够接入全连接层。

### 2.4 Kaiming 初始化的意义

卷积网络使用 ReLU 激活时，如果初始化不合理，随着网络加深，激活值和梯度很容易逐层衰减或爆炸。Kaiming 初始化是专门针对 ReLU 设计的初始化方式，其目标是让每层输出的方差保持在合理范围内。

实现上通常写作：

```python
nn.init.kaiming_normal_(weight, nonlinearity='relu')
```

偏置则通常初始化为 0。

这一点说明：

> 初始化并不是实现细节，而是影响训练稳定性的重要组成部分。

---

## 3. 使用 `nn.Module` 组织网络

在用函数式 API 明确了网络的本质后，下一步就是理解为什么 PyTorch 推荐使用 `nn.Module`。

### 3.1 `nn.Module` 的作用

`nn.Module` 的核心价值在于：

1. 自动注册参数。
2. 自动管理子模块。
3. 统一使用 `forward()` 描述前向传播。

例如：

```python
class ThreeLayerConvNet(nn.Module):
    def __init__(self, ...):
        self.conv1 = nn.Conv2d(...)
        self.conv2 = nn.Conv2d(...)
        self.fc = nn.Linear(...)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(flatten(x))
```

这种写法的本质是把“前向传播逻辑”和“参数管理逻辑”封装到一个对象中。

### 3.2 为什么这比裸张量实现更好

因为模型参数已经被自动注册，所以可以直接：

```python
optimizer = optim.SGD(model.parameters(), lr=..., weight_decay=...)
```

而不需要手动把所有参数收集到列表里。这样代码更加模块化，可复用性也更强。

---

## 4. 使用 `nn.Sequential` 表达串行结构

对于纯串行的前馈网络，还可以继续提高抽象层次，使用 `nn.Sequential`。

### 4.1 设计思想

`nn.Sequential` 的本质是：将多个模块按顺序串联起来，前一个模块的输出自动作为下一个模块的输入。

例如：

```python
model = nn.Sequential(
    nn.Conv2d(...),
    nn.ReLU(),
    nn.Conv2d(...),
    nn.ReLU(),
    Flatten(),
    nn.Linear(...),
)
```

### 4.2 适用边界

这种写法适用于：

- 单输入、单输出
- 无分支结构
- 无跳连结构
- 前向传播是简单顺序堆叠

这也是为什么三层 ConvNet 可以方便地写成 `Sequential`，但 ResNet 则需要重新定义 block 和 shortcut 模块。

---

## 5. ResNet：残差学习的原理与实现

A4 中最能体现现代卷积网络结构思想的部分就是 ResNet。

### 5.1 为什么深层网络难训练

理论上更深的网络表达能力更强，但实际训练中，网络加深后往往出现：

- 梯度传播变差
- 优化难度加大
- 更深网络训练误差反而更高

ResNet 的核心思想是引入残差连接，使每个 block 不必直接学习完整映射 \(H(x)\)，而只需学习残差：

\[
F(x) = H(x) - x
\]

最终输出为：

\[
H(x) = F(x) + x
\]

在代码中，这个思想体现为：

```python
out = self.block(x) + self.shortcut(x)
```

shortcut 为梯度提供了一条更直接的传播路径，也降低了深层网络的优化难度。

### 5.2 PlainBlock 与 ResidualBlock

#### PlainBlock

`PlainBlock` 是不带残差连接的卷积块，其结构为：

1. BatchNorm
2. ReLU
3. Conv
4. BatchNorm
5. ReLU
6. Conv

注意这里采用的是 pre-activation 风格，也就是说归一化和激活发生在卷积之前。

#### ResidualBlock

`ResidualBlock` 在 `PlainBlock` 的基础上加入 shortcut 分支：

```python
return self.block(x) + self.shortcut(x)
```

### 5.3 为什么 shortcut 有时需要 `1x1` 卷积

只有当主分支输出与 shortcut 输出形状完全一致时，才能做逐元素相加。如果出现以下情况，就需要对 shortcut 做额外变换：

- 输入通道数与输出通道数不同
- 主分支做了下采样，空间尺寸减半

这时 shortcut 会使用：

- `1x1` 卷积调整通道数
- `stride=2` 实现同步下采样

如果输入输出形状一致，则可以直接使用：

```python
nn.Identity()
```

### 5.4 Stage 的概念

一个 `ResNetStage` 通常由多个 block 组成：

- 第一个 block 负责必要的通道变化和下采样
- 后续 block 保持输出通道和空间尺寸不变

其典型逻辑为：

```python
blocks = [block(Cin, Cout, downsample)]
for _ in range(num_blocks - 1):
    blocks.append(block(Cout, Cout))
```

这体现了 ResNet 结构的层次化设计：stem -> stages -> classifier。

### 5.5 `stage_args` 的理解

在作业中，stage 的配置通常写为：

```python
(8, 8, 5, False)
(8, 16, 5, True)
(16, 32, 5, True)
```

其含义是：

```python
(Cin, Cout, num_blocks, downsample)
```

因此，网络最开始的 stem 需要负责把输入图片从原始通道数映射到第一个 stage 所需通道数。例如对 CIFAR-10 图像来说：

```python
ResNetStem(3, 8)
```

这里的 `3` 是原图 RGB 通道数，而 `8` 是第一个 stage 的输入通道数。两者不是同一个概念。

### 5.6 Global Average Pooling

在卷积部分结束后，特征张量仍然具有形状 `(N, C, H, W)`。如果直接展平接全连接层，会引入大量参数，也会强依赖输入空间大小。

作业中的实现方式是：

```python
x = x.mean(dim=(2, 3))
```

这相当于 global average pooling。其优点包括：

- 大幅减少参数量
- 减少对固定输入尺寸的依赖
- 让每个通道更像一种全局语义响应

---

## 6. BatchNorm、ReLU 与 pre-activation 结构

在 A4 的 ResNet 实现中，block 的顺序是：

```python
BN -> ReLU -> Conv
```

而不是初学者更熟悉的：

```python
Conv -> BN -> ReLU
```

这对应的是 pre-activation ResNet 的设计思想。

### 6.1 BatchNorm 的作用

BatchNorm 对每个通道使用小批量统计量进行归一化，再加上可学习的缩放和平移参数。它的主要作用包括：

- 稳定各层输入分布
- 加快训练速度
- 改善深层网络中的梯度传播

### 6.2 为什么卷积常设 `bias=False`

如果卷积层后面马上接 BatchNorm，那么卷积的 bias 往往会被 BatchNorm 的平移参数抵消，因此会显得冗余。于是很多 ResNet 实现会写：

```python
nn.Conv2d(..., bias=False)
```

这体现了结构设计中的一个重要原则：如果某个参数不会真正增加模型表达能力，就可以省略。

---

## 7. 输入梯度：从训练参数到分析输入

`network_visualization.py` 这一部分的重要意义在于，它让我们从“参数优化”的思维转换到“输入优化”的思维。

也就是说，梯度不仅能回答：

- 参数该怎么改，损失会下降？

也能回答：

- 输入图像该怎么改，某个类别分数会升高？
- 哪些像素最影响模型的判断？

这正是 saliency map、adversarial attack 和 class visualization 的共同基础。

---

## 8. Saliency Map：类别分数对输入的敏感性

### 8.1 数学含义

对输入图像 \(X\) 和某个类别分数 \(s_y(X)\)，saliency map 的基础量是：

\[
\frac{\partial s_y}{\partial X}
\]

这个梯度描述的是：如果对输入像素做一个很小的扰动，目标类别分数会变化多少。

绝对值越大，说明该像素对模型当前判断越重要。

### 8.2 为什么对通道取绝对值最大值

输入图像是 RGB 三通道，因此梯度的形状为 `(N, 3, H, W)`。为了得到单通道热力图，作业采用：

```python
saliency = X.grad.abs().max(dim=1)[0]
```

这样做有两个目的：

1. 先取绝对值，强调“影响大小”而不区分正负方向。
2. 再对通道维取最大值，保留每个像素位置上最强的颜色通道响应。

最终得到 `(N, H, W)` 形状的显著图。

### 8.3 代码实现结构

```python
X.requires_grad_()
scores = model(X)
correct_scores = scores.gather(1, y.view(-1, 1)).sum()
correct_scores.backward()
saliency = X.grad.abs().max(dim=1)[0]
```

这一段代码很好地体现了“数学定义 -> 张量实现”的转换过程。

---

## 9. Adversarial Attack：利用梯度操控模型输出

### 9.1 基本目标

对抗攻击的目标不是更新模型参数，而是修改输入图像，使模型把它误判为指定类别。形式上可以写成：

\[
\max_X s_{target}(X)
\]

也就是说，我们希望目标类别分数不断上升，直到模型预测结果发生改变。

### 9.2 为什么这里是梯度上升

训练模型时常用梯度下降最小化损失，而对抗攻击的目标是增大目标类别得分，因此使用的是梯度上升：

```python
X_adv += step
```

而不是传统训练中的：

```python
param -= step
```

### 9.3 为什么要归一化梯度

在作业中，更新写作：

```python
grad = X_adv.grad
X_adv += learning_rate * grad / grad.norm()
```

归一化的作用在于：

- 控制每一步移动距离
- 防止梯度过大导致更新不稳定
- 强调更新方向，而不是绝对数值大小

### 9.4 停止条件

每轮优化后检查：

```python
pred = scores.argmax(dim=1).item()
if pred == target_y:
    break
```

一旦模型已经把图像判成目标类，就说明攻击成功，可以停止迭代。

从本质上看，这说明对抗样本的关键不是“让目标类得分无限大”，而是“让输入跨过决策边界”。

---

## 10. Class Visualization：直接生成让模型高置信识别的图像

### 10.1 优化目标

类可视化试图直接构造一张图像，使模型对某个类别有较高分数。常见目标函数为：

\[
\max_I \left(s_y(I) - \lambda \|I\|_2^2\right)
\]

其中：

- 第一项鼓励图像激活目标类别
- 第二项是 L2 正则项，用于抑制像素值无限增大

### 10.2 为什么需要正则化

如果不加约束，优化过程很容易得到极端噪声图案，这些图案可能能强烈激活网络，但不具备可解释性。L2 正则可以一定程度上约束图像幅值，使生成过程更稳定。

### 10.3 更新步骤

典型实现为：

```python
score = model(img)[0, target_y] - l2_reg * torch.sum(img * img)
score.backward()
with torch.no_grad():
    img += learning_rate * img.grad
img.grad.zero_()
```

这里和对抗攻击非常相似，区别在于：

- 对抗攻击从真实图像出发，目标是误导分类器
- 类可视化通常从随机图像出发，目标是生成能代表某一类别特征的图案

### 10.4 Jitter、Blur 与 Clamp 的意义

notebook 在外层优化过程中还加入了：

- 随机抖动（jitter）
- 周期性模糊（blur）
- 像素值截断（clamp）

这些操作本质上都是隐式正则化，用来：

- 降低对固定像素排列的过拟合
- 减少高频噪声
- 提高结果的平滑性与可解释性

---

## 11. Style Transfer：在特征空间中优化图像

A4 的最后一部分将前面所有思想整合到一起：

- 使用预训练 CNN 作为固定特征提取器
- 定义内容损失、风格损失和总变分损失
- 将生成图像本身作为优化变量
- 用梯度下降/Adam 不断更新图像

总损失形式为：

\[
L = L_{content} + L_{style} + L_{tv}
\]

这部分的关键在于：

> 图像不是在像素空间里直接和风格图“长得一样”，而是在卷积特征空间中同时满足“内容接近内容图、统计风格接近风格图”。

---

## 12. Content Loss：匹配高层语义表示

### 12.1 原理

设某一层 \(\ell\) 上：

- 当前生成图像特征为 \(F^\ell\)
- 内容图像特征为 \(P^\ell\)

则内容损失定义为：

\[
L_c = w_c \sum (F^\ell - P^\ell)^2
\]

这意味着生成图像需要在该层特征空间中接近内容图像，从而保留其整体语义结构，而不是严格匹配每个像素。

### 12.2 代码实现

```python
return content_weight * torch.sum((content_current - content_original) ** 2)
```

这是一个标准的加权平方差损失。

---

## 13. Gram Matrix：用通道相关性表示风格

### 13.1 为什么风格适合用统计量表示

内容信息更依赖空间位置，例如“建筑物在中间”“边界在哪里”；而风格信息更像是：

- 纹理分布
- 颜色关系
- 局部图案重复
- 画笔笔触模式

这些信息往往不需要精确的空间对齐，因此可以通过统计特征通道之间的相关性来表示。

### 13.2 Gram Matrix 的定义

给定形状为 `(N, C, H, W)` 的特征图，将其 reshape 为 `(N, C, H*W)` 后，Gram 矩阵定义为：

\[
G = F F^T
\]

输出形状为 `(N, C, C)`。

其中 \(G_{ij}\) 表示第 `i` 个通道和第 `j` 个通道在所有空间位置上的共现程度，也就是通道间的相关性。

### 13.3 代码实现

```python
N, C, H, W = features.shape
features_flat = features.reshape(N, C, H * W)
gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
if normalize:
    gram /= (C * H * W)
```

这里使用 `torch.bmm`，是因为对 batch 中每张图都要单独进行矩阵乘法。

### 13.4 为什么要归一化

如果不归一化，不同层的 Gram 矩阵数值会受到以下因素显著影响：

- 通道数 `C`
- 空间大小 `H * W`
- 特征值整体幅度

归一化后，不同层之间的损失尺度更容易协调。

---

## 14. Style Loss：多层风格统计匹配

### 14.1 原理

风格损失并不是只在一层上定义，而是在多个层上共同约束：

\[
L_s = \sum_{\ell \in \mathcal{L}} w_\ell \sum (G^\ell - A^\ell)^2
\]

其中：

- \(G^\ell\) 是当前图像第 \(\ell\) 层的 Gram 矩阵
- \(A^\ell\) 是风格图像第 \(\ell\) 层的 Gram 矩阵

### 14.2 为什么要多层

不同深度的卷积层编码不同尺度的信息：

- 浅层更关注边缘、颜色块、细纹理
- 深层更关注较大尺度的纹理结构与模式组合

因此，多层风格损失可以同时约束局部纹理与整体风格模式。

### 14.3 代码实现

```python
loss = 0
for i, layer in enumerate(style_layers):
    loss += style_weights[i] * torch.sum(
        (gram_matrix(feats[layer]) - style_targets[i]) ** 2
    )
return loss
```

这一部分展示了 style transfer 的核心实现逻辑：

- 提取当前图像的特征
- 计算其风格统计量
- 与目标风格统计量进行距离比较

---

## 15. Total Variation Loss：鼓励空间平滑性

### 15.1 原理

总变分损失用于惩罚相邻像素之间过大的变化，从而降低噪声并提升图像平滑性。它同时考虑：

- 垂直方向相邻像素差异
- 水平方向相邻像素差异

### 15.2 向量化实现

```python
vertical = img[:, :, 1:, :] - img[:, :, :-1, :]
horizontal = img[:, :, :, 1:] - img[:, :, :, :-1]
return tv_weight * (torch.sum(vertical ** 2) + torch.sum(horizontal ** 2))
```

这段代码没有使用任何显式 Python 循环，而是通过张量切片直接构造所有相邻像素差。这是一种非常典型、也非常重要的深度学习实现方式：

> 用张量操作替代 Python 循环，提高效率并保持可微分性。

---

## 16. 从优化角度理解风格迁移

风格迁移表面上像图像编辑任务，但从优化角度看，它与训练模型有统一的形式。

区别只在于：

- 训练模型时，优化对象是网络参数；
- 风格迁移时，优化对象是图像本身。

整体过程可以概括为：

```python
img.requires_grad_()
optimizer = torch.optim.Adam([img], lr=...)

for t in range(T):
    optimizer.zero_grad()
    feats = extract_features(img, cnn)
    loss = content_loss(...) + style_loss(...) + tv_loss(...)
    loss.backward()
    optimizer.step()
```

这说明一个非常重要的观念：

> 深度学习中的“可学习变量”不一定是模型参数，也可以是输入图像本身。

这也是为什么本次作业中，网络可视化、对抗攻击、类可视化和风格迁移在底层逻辑上其实是一致的。

---

## 17. A4 训练出的核心代码能力

通过完成本次作业，可以系统训练以下几类能力。

### 17.1 将数学公式翻译为张量代码

这是 A4 最核心的训练目标之一。几乎每一部分都要求将数学定义直接实现成向量化张量运算，例如：

- content loss：特征差平方和
- Gram matrix：批量矩阵乘法
- style loss：多层损失加权求和
- TV loss：相邻像素差的平方和
- saliency map：输入梯度的绝对值与通道最大化

### 17.2 对张量形状保持高度敏感

很多错误并不是来自公式理解错误，而是来自维度理解错误。因此本作业尤其强调：

- 卷积前后通道数如何变化
- `stride=2` 后空间尺寸如何变化
- 什么时候需要 `flatten`
- Gram matrix 为什么是 `(N, C, C)`
- global average pooling 为什么输出 `(N, C)`

### 17.3 理解“到底在优化谁”

这是从初学者迈向真正理解深度学习的重要一步。

A4 中不同任务优化的对象不同：

- 标准训练：优化模型参数
- saliency map：不优化，只分析输入梯度
- adversarial attack：优化输入图像
- class visualization：优化输入图像
- style transfer：优化生成图像

只有明确这一点，才能真正理解代码的意义。

### 17.4 掌握安全的 autograd 使用习惯

本作业也反复强化了若干重要实现习惯：

- 只给真正需要梯度的张量打开 `requires_grad`
- 每次复用梯度前先清零
- 避免使用 `.data` 进行更新
- 在更新步骤中优先使用 `with torch.no_grad()`

这些看似细节，实际上对调试和写出稳定代码非常重要。

---

## 18. A4 的整体知识主线

如果用一句话总结 A4，可以写成：

> 一旦把神经网络视为一个可微函数和一个特征提取器，就可以不仅训练模型本身，还可以分析模型、操控输入、重建图像，并在特征空间中定义复杂的视觉目标。

这也是本次作业最重要的知识串联：

- 自动求导解释了为什么这些任务都能用梯度完成；
- 卷积网络结构解释了特征是如何逐层提取的；
- ResNet 解释了现代深层 CNN 如何通过残差连接变得可训练；
- 输入梯度解释了模型为何可以被可视化和攻击；
- 风格迁移则展示了如何利用预训练网络的表示能力完成图像生成任务。

因此，A4 并不是简单地“学了几个技巧”，而是在训练一种统一的深度学习思维方式：

1. 把问题写成一个可微目标；
2. 把目标写成张量运算；
3. 借助自动求导得到梯度；
4. 用优化方法更新参数或输入；
5. 从结果中理解模型的表示和行为。

---

## 19. 总结

Assignment 4 的价值，不仅在于完成了若干具体实现，更在于它把深度学习中几个非常重要但常被割裂理解的主题连接到了一起：

- 计算图与自动求导
- 卷积神经网络实现
- 网络模块化抽象
- 残差学习
- 输入梯度分析
- 特征空间优化
- 图像生成与风格迁移

从代码层面看，这份作业训练了把数学公式转化为 PyTorch 张量程序的能力；从原理层面看，这份作业帮助建立了“梯度不仅能训练模型，也能解释模型和操控输入”的统一认识。

因此，A4 可以被视为一次从“会调用深度学习框架”走向“真正理解深度学习计算本质”的关键训练。对于后续学习更复杂的生成模型、注意力机制、对比学习以及视觉大模型而言，这份作业打下了非常坚实的基础。
