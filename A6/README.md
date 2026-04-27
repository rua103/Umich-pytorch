# A6 README / 实现笔记

## 1. 作业内容概览

本次 A6 主要实现两大类生成模型：

1. `VAE / CVAE`：
   - `VAE`：全连接变分自编码器
   - `CVAE`：带类别条件的全连接变分自编码器
2. `GAN`：
   - 基础全连接 GAN
   - Least Squares GAN (LSGAN)
   - DCGAN 风格卷积生成器 / 判别器

对应代码文件：

- `A6/vae.py`
- `A6/gan.py`
- `A6/variational_autoencoders.ipynb`
- `A6/generative_adversarial_networks.ipynb`
- `A6/log`

---

## 2. 我在 A6 中完成的实现

### 2.1 `vae.py`

实现了以下内容：

#### `VAE`
- Encoder:
  - `Flatten`
  - `Linear(input_size -> hidden_dim)`
  - `ReLU`
- 两个后验参数层：
  - `mu_layer`
  - `logvar_layer`
- Decoder:
  - `Linear(latent_size -> hidden_dim)`
  - 多层 `ReLU`
  - `Linear(hidden_dim -> input_size)`
  - `Sigmoid`
  - `Unflatten`
- `forward`:
  - 编码得到 `mu, logvar`
  - 用 `reparametrize` 采样 `z`
  - 用 decoder 重建 `x_hat`

#### `CVAE`
- 在 encoder 输入端拼接 `x_flat` 和 one-hot 类别向量 `c`
- 在 decoder 输入端拼接 `z` 和类别向量 `c`
- 实现条件生成

#### `reparametrize(mu, logvar)`
- 用公式：`z = mu + eps * std`
- 其中：`std = exp(0.5 * logvar)`

#### `loss_function(x_hat, x, mu, logvar)`
- 重建损失：`binary_cross_entropy(..., reduction='sum')`
- KL 散度：
  \[
  -\frac{1}{2} \sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)
  \]
- 最后按 batch 平均

### 2.2 `gan.py`

实现了以下内容：

#### 基础 GAN 部分
- `sample_noise`
  - 生成 `[-1, 1]` 上的均匀噪声
- `discriminator`
  - 三层全连接 + `LeakyReLU(0.01)`
- `generator`
  - 两层隐藏层 + 输出到 `784` + `Tanh`
- `discriminator_loss`
  - `binary_cross_entropy_with_logits`
- `generator_loss`
  - 目标是假样本被判成真
- `get_optimizer`
  - `Adam(lr=1e-3, betas=(0.5, 0.999))`

#### LSGAN
- `ls_discriminator_loss`
- `ls_generator_loss`

#### DCGAN
- `build_dc_classifier`
  - `Unflatten`
  - 两层卷积 + `MaxPool`
  - 两层全连接
- `build_dc_generator`
  - 全连接 + `BatchNorm1d`
  - `Unflatten`
  - 两层 `ConvTranspose2d`
  - `BatchNorm2d`
  - `Tanh`
  - `Flatten`

---

## 3. 关键数学原理总结

---

### 3.1 VAE 的本质

普通 AutoEncoder 只是“压缩再还原”，而 VAE 不是学一个确定的隐变量，而是学一个分布：

\[
q_\phi(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))
\]

也就是说，对于每张输入图像，encoder 输出的不是单个点，而是一个高斯分布的参数：

- 均值 `mu`
- 方差 `sigma^2`（代码里用 `logvar = log(sigma^2)`）

这样在推理阶段，不仅能重建，还能从隐空间采样并生成新图像。

---

### 3.2 为什么预测 `logvar` 而不是 `var`

这是 VAE 的一个经典技巧。

原因有两个：

#### 1）值域更自然
方差必须大于 0，但线性层输出范围是整个实数轴。

如果直接预测 `var`，需要额外想办法保证它恒正。

而如果预测：

\[
\log \sigma^2 \in (-\infty, +\infty)
\]

那线性层可以直接输出任意实数，再通过：

\[
\sigma^2 = \exp(\text{logvar})
\]

恢复成正的方差。

#### 2）数值更稳定
KL 散度公式本来就带有 `log(sigma^2)`，直接输出 `logvar` 可以避免 `log(0)` 一类问题。

---

### 3.3 重参数化技巧（Reparameterization Trick）

VAE 最大的难点之一就是：采样这个操作不可直接反向传播。

如果直接写：

\[
z \sim \mathcal{N}(\mu, \sigma^2)
\]

那么梯度过不去。

所以改写为：

\[
\epsilon \sim \mathcal{N}(0, I)
\]

\[
z = \mu + \sigma \odot \epsilon
\]

其中：

\[
\sigma = \exp(0.5 \cdot \text{logvar})
\]

这样随机性全部放在 `epsilon` 上，而 `z` 关于 `mu` 和 `logvar` 仍然是可导的。

这就是 VAE 能训练的核心。

---

### 3.4 VAE 损失函数

VAE 的损失由两部分组成：

#### 1）重建损失 Reconstruction Loss
衡量重建图 `x_hat` 和原图 `x` 是否相似。

这里使用 BCE：

\[
-\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]
\]

在实现上就是：

- `F.binary_cross_entropy(x_hat, x, reduction='sum')`

#### 2）KL 散度 KL Divergence
约束后验分布接近标准正态分布：

\[
p(z)=\mathcal{N}(0, I)
\]

单样本公式：

\[
D_{KL}(q_\phi(z|x) || p(z))
=
-\frac{1}{2}\sum_j \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
\]

向量化后直接在 batch 上求和再平均。

#### 总体损失

\[
\mathcal{L} = \text{Reconstruction Loss} + \text{KL Loss}
\]

注意这里是在最小化负 ELBO。

---

### 3.5 CVAE 的数学意义

CVAE 相比 VAE 的区别在于：加入条件变量 `c`。

普通 VAE：

\[
q_\phi(z|x), \quad p_\theta(x|z)
\]

CVAE：

\[
q_\phi(z|x,c), \quad p_\theta(x|z,c)
\]

这意味着：

- encoder 不只是看图像 `x`，还看标签 `c`
- decoder 不只是看 latent `z`，还看标签 `c`

这样生成时可以指定类别，比如“生成数字 7”，而不是随机吐出任意数字。

本质上是把“类别信息”和“风格信息”做一定程度的解耦：

- `c` 负责内容类别
- `z` 更倾向负责形态风格

---

### 3.6 GAN 的本质

GAN 由两个网络组成：

- `Generator G`：从噪声生成图像
- `Discriminator D`：判断图像是真是假

可以理解成一个对抗博弈：

- `D` 想正确区分真图和假图
- `G` 想骗过 `D`

原始 minimax 形式：

\[
\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1-D(G(z)))]
\]

实际训练时常用 non-saturating generator loss：

\[
\ell_G = -\mathbb{E}_{z \sim p(z)}[\log D(G(z))]
\]

因为这比原始的 `log(1-D(G(z)))` 梯度更稳定。

---

### 3.7 为什么判别器最后不加 `Sigmoid`

因为实现中直接使用：

- `binary_cross_entropy_with_logits`

它会把：

1. `sigmoid`
2. `BCE`

合并起来做，数值更稳定。

如果自己先手动 `sigmoid`，再算 `log`，在极端值时容易溢出或出现 `NaN`。

---

### 3.8 LSGAN 为什么更稳定

LSGAN 用最小二乘损失替代 BCE：

生成器损失：

\[
\ell_G = \frac{1}{2} \mathbb{E}[(D(G(z)) - 1)^2]
\]

判别器损失：

\[
\ell_D = \frac{1}{2}\mathbb{E}[(D(x)-1)^2] + \frac{1}{2}\mathbb{E}[D(G(z))^2]
\]

好处：

1. 梯度更平滑
2. 不容易像原始 GAN 一样在判别器过强时梯度消失
3. 对离目标很远的假样本惩罚更强

本质上把“真假二分类”部分转换成了“回归到目标值”的问题。

---

### 3.9 DCGAN 的空间归纳偏置

基础全连接 GAN 最大的问题是：

- 把 28×28 图像直接看成 784 维向量
- 缺少局部空间结构建模能力

DCGAN 使用卷积和转置卷积，优势是：

- 卷积判别器更擅长识别局部纹理、边缘、笔画
- 转置卷积生成器更适合逐步上采样出二维结构图像

所以生成效果通常明显好于 FC-GAN。

---

## 4. 结合本次实现的难点记录

### 难点 1：VAE encoder / decoder 结构容易少层

Notebook 对 FC-VAE encoder 的说明是“三层 Linear+ReLU”。

我在写代码时很容易先只写成：

- `Flatten`
- `Linear`
- `ReLU`

这样虽然能跑，但不完全符合题目描述。

经验：以后看 network architecture 时，必须把每一层逐条核对，不要只看第一层和最后一层。

---

### 难点 2：`hidden_dim` 如果没赋值会直接炸

在早期调试里，`self.hidden_dim = None` 时就拿去建 `nn.Linear`，会报类型错误。

本质原因：

- `nn.Linear(in_features, out_features)` 的两个参数都必须是整数
- `None` 不能拿来创建权重矩阵

经验：网络超参数必须先定死，再接模块。

---

### 难点 3：`forward` 里很容易忘传参数

例如：

- 错写成 `self.mu_layer()`
- 正确应该是 `self.mu_layer(x_hidden)`

这种错误在写 layer-by-layer 前向传播时非常常见，因为模块本身是可调用对象，少写一个参数不会在编辑时立刻提示。

经验：每写完 `forward`，都手工检查数据流形状：

- `x -> hidden`
- `hidden -> mu, logvar`
- `mu, logvar -> z`
- `z -> x_hat`

---

### 难点 4：`reparametrize` 容易忘记最终赋给 `z`

模板里先写了：

- `z = None`

如果只写中间过程而忘记 `z = mu + eps * std`，最后测试时会出现 `NoneType` 错误。

经验：凡是模板函数先初始化为 `None`，最后一定确认返回值真的被覆盖。

---

### 难点 5：VAE loss 里符号最容易写错

尤其是 KL 项：

\[
-\frac{1}{2} \sum (1 + logvar - mu^2 - exp(logvar))
\]

容易错成：

- 漏掉最前面的负号
- `mu^2` 写成正号
- `exp(logvar)` 写成正号
- 用了 `mean` 而不是题目要求的先 `sum` 再除 batch

经验：KL 公式不要靠记忆模糊写，必须按 notebook 原式逐符号搬运。

---

### 难点 6：CVAE 的拼接维度很容易搞错

正确做法：

- encoder 输入：`torch.cat([x_flat, c], dim=1)`
- decoder 输入：`torch.cat([z, c], dim=1)`

如果写成 `dim=0`，就会把 batch 维拼掉，形状直接错。

经验：

- `dim=0` 是 batch 维
- `dim=1` 才是 feature 维

只要是“给每个样本附加额外特征”，通常都拼在 `dim=1`。

---

### 难点 7：GAN 的标签方向容易反

在实现 `generator_loss` 时，虽然输入是假图 `logits_fake`，但目标标签应该是 `1`，不是 `0`。

原因：

- 生成器想让判别器把假图看成真图

所以：

- `discriminator_loss`: real -> 1, fake -> 0
- `generator_loss`: fake -> 1

这是 GAN 新手非常容易反的点。

---

### 难点 8：`device` 拼写错误会报很奇怪的错

在 `sample_noise` 中，曾出现过：

- `devcie=device`

而不是：

- `device=device`

这类错误的特点是：

- 表面上报的是 `torch.rand()` 参数组合错误
- 实际上只是关键字参数拼错了

经验：看到 “invalid combination of arguments” 时，不只检查类型，也要检查参数名拼写。

---

### 难点 9：`nn.Linear` 和 `nn.Unflatten` 的职责不能混

`nn.Linear` 输出必须是整数维度，比如 `784`。

`nn.Unflatten` 才负责把 `784` 还原成：

- `(1, 28, 28)`

如果把 tuple 误传给 `nn.Linear`，就会报非常绕的类型错误。

经验：

- 线性层处理“长度”
- `Unflatten` 处理“形状”

---

### 难点 10：GAN / DCGAN 中 `BatchNorm1d` 和 `BatchNorm2d` 容易混

规律：

- 全连接输出 `(N, C)`：用 `BatchNorm1d`
- 卷积特征图 `(N, C, H, W)`：用 `BatchNorm2d`

本次 DCGAN 生成器里：

- FC 后用 `BatchNorm1d`
- `ConvTranspose2d` 后用 `BatchNorm2d`

经验：看输入张量维度决定 BatchNorm 类型，不要靠感觉记。

---

## 5. 易错点清单

下面列一个更适合考前/提交前快速检查的版本。

### VAE / CVAE

- [ ] `hidden_dim` 是否已赋整数值
- [ ] encoder 是否按题目要求写够层数
- [ ] decoder 最后一层是否是 `Sigmoid`
- [ ] decoder 最后是否 `Unflatten`
- [ ] `reparametrize` 是否用 `exp(0.5 * logvar)`
- [ ] `eps` 是否用标准正态 `randn_like`
- [ ] `forward` 是否正确返回 `x_hat, mu, logvar`
- [ ] `loss_function` 是否是 `BCE + KL`
- [ ] BCE 是否 `reduction='sum'`
- [ ] 最后是否除以 `batch_size`
- [ ] CVAE 是否在 encoder 和 decoder 两端都拼接条件向量

### GAN

- [ ] `sample_noise` 是否生成 `[-1, 1]`
- [ ] 判别器最后一层是否不加 `Sigmoid`
- [ ] generator 最后一层是否 `Tanh`
- [ ] `generator_loss` 是否把 fake 的 label 设成 1
- [ ] `discriminator_loss` 是否分别处理 real=1, fake=0
- [ ] 是否使用 `binary_cross_entropy_with_logits`
- [ ] `Adam` 参数是否是 `lr=1e-3, betas=(0.5, 0.999)`

### DCGAN

- [ ] 输入全连接后是否正确 `Unflatten`
- [ ] `ConvTranspose2d` 的 `padding=1` 是否正确
- [ ] FC 后是否用 `BatchNorm1d`
- [ ] Conv 后是否用 `BatchNorm2d`
- [ ] 最终是否回到 `784` 向量

---

## 6. 对 VAE、CVAE、GAN 的理解总结

### VAE
优点：
- 训练稳定
- 隐空间连续且可插值
- 数学解释清楚

缺点：
- 样本通常偏模糊

### CVAE
优点：
- 可控生成
- 可以指定类别
- 相比普通 VAE 更有目标性

缺点：
- 本质上仍继承 VAE 偏模糊的问题

### GAN
优点：
- 生成图像更锐利、更逼真

缺点：
- 训练不稳定
- 容易模式崩溃
- 对超参数敏感

### LSGAN
相比原始 GAN：
- 梯度更平滑
- 训练更稳

### DCGAN
相比 FC-GAN：
- 更利用图像空间结构
- 生成质量更高

---

## 7. 我这次作业里最值得记住的结论

### 1）VAE 的核心不是“压缩”，而是“学习分布”
不是输出一个 latent point，而是输出 `mu` 和 `logvar`，再通过重参数化采样。

### 2）重参数化技巧是 VAE 可训练的关键
如果没有：

\[
z = \mu + \sigma \epsilon
\]

梯度无法穿过采样操作。

### 3）KL 项的作用是把 latent space 整理成标准正态附近
这样 latent space 才能连续采样、平滑插值，而不是碎成孤岛。

### 4）CVAE 的核心是条件拼接
把类别信息显式送给 encoder 和 decoder，生成结果才可控。

### 5）GAN 训练本质是二人博弈
- D 负责鉴假
- G 负责造假

### 6）GAN 实现里最重要的工程细节之一是 `with_logits`
数值稳定远比手写 `sigmoid + log` 安全。

### 7）卷积式生成模型比全连接式更适合图像
因为图像不是普通向量，而是带空间结构的数据。

---

## 8. 如果以后重做这份作业，我会优先检查什么

1. 先核对每个网络的 architecture 是否逐层一致
2. 再核对 tensor shape 是否一致
3. 再核对损失函数符号
4. 再核对 `mean/sum` 是否与题目一致
5. 最后才查训练效果

因为这类作业大部分 bug 都不是“理论不会”，而是：

- 维度错
- 层数少
- 激活函数放错
- 标签方向反
- reduction 用错

---

## 9. 简短结语

A6 是我第一次比较系统地把两类经典生成模型放在一起实现和比较：

- `VAE/CVAE` 代表“概率建模 + 变分推断”路线
- `GAN/LSGAN/DCGAN` 代表“对抗训练 + 高质量生成”路线

这次最大的收获不是单纯把代码跑通，而是理解了：

- 为什么 VAE 需要 `logvar`
- 为什么必须重参数化
- 为什么 KL 项能整理隐空间
- 为什么 GAN 要用 `with_logits`
- 为什么卷积结构对图像生成更重要

如果以后复习生成模型，这份 README 可以直接作为速查笔记使用。

