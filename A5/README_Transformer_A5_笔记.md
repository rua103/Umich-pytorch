# Transformer A5 README：原理、重点与我的难点整理

这份 README 是为 **A5 Transformer** 专门整理的复习版笔记。
目标不是面面俱到，而是抓住三件事：

1. **Transformer 的原理到底是什么**
2. **这份作业里最重要的主线和重点是什么**
3. **我自己最容易卡住的难点是什么**

如果只想快速复习，优先看：

- 第 2 部分：整体原理
- 第 3 部分：训练 / 推理主线
- 第 4 部分：重点与难点

---

# 1. 这份作业在做什么

这份作业实现的是一个 **Encoder-Decoder Transformer**，用于处理一个简化版的 sequence-to-sequence 问题。

和标准 NLP 翻译任务相比，这里更简单，因为：

- 输入序列长度固定
- 输出序列长度固定
- 数据是 toy dataset（算术表达式）

所以这份作业的重点不是复杂数据处理，而是：

> **从零理解并实现 Transformer 的核心结构。**

---

# 2. Transformer 的原理

## 2.1 为什么 Transformer 会出现

在 RNN / LSTM 中，序列是按时间一步一步处理的：

- 第 1 个 token 先算
- 再算第 2 个 token
- 再算第 3 个 token

这样的问题是：

1. **难并行**：因为后一个时刻依赖前一个时刻
2. **长距离依赖困难**：虽然 LSTM 比普通 RNN 好，但长序列仍然不容易

Transformer 的核心想法是：

> **不要再靠“时间递推”传播信息，而是让每个位置直接和所有位置建立联系。**

这就是 Attention。

---

## 2.2 Self-Attention 是什么

Transformer 最核心的公式是：

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

它的直觉可以理解成：

- `Q`：我现在想找什么信息
- `K`：每个位置各自提供什么信息索引
- `V`：每个位置真正携带的内容

流程是：

1. `QK^T` 计算相似度
2. 除以 `sqrt(d_k)` 防止数值过大
3. `softmax` 变成权重
4. 用这些权重对 `V` 加权平均

一句话理解：

> **每个 token 都去看全序列里哪些 token 对自己最重要。**

---

## 2.3 为什么它比 RNN 更并行

因为 Self-Attention 不需要按时间一步一步传隐藏状态。

对于长度为 `K` 的序列：

- RNN：第 `t` 步必须等第 `t-1` 步先算完
- Transformer：所有位置的表示可以一起算

这就是 Transformer 训练快的重要原因。

但是它也有代价：

- 注意力矩阵大小是 `K x K`
- 所以复杂度是二次的

\[
O(K^2)
\]

所以：

> **Transformer 并行性强，但对长序列开销大。**

---

## 2.4 为什么需要 Positional Encoding

Self-Attention 本身只关心“谁和谁相关”，并不知道顺序。

比如：

- `I love you`
- `you love I`

如果没有位置信息，模型很难区分它们。

所以需要 **Positional Encoding**，把“位置”信息加到 embedding 上。

你这份作业里有两种位置编码：

### 1）简单版

第 `n` 个位置编码成 `n / K`。

优点：

- 简单
- 好理解

缺点：

- 序列一长，相邻位置差别越来越小
- embedding 各维只是重复同一个值，信息太单一

### 2）Sinusoid 版

原论文使用的是正余弦位置编码：

\[
PE(p, 2i) = \sin\left(\frac{p}{10000^{2i/M}}\right)
\]

\[
PE(p, 2i+1) = \cos\left(\frac{p}{10000^{2i/M}}\right)
\]

它的作用是：

- 不同维度用不同频率编码位置
- 让模型既能区分绝对位置，也更容易感知相对位置

一句话理解：

> **Transformer 自己不懂顺序，所以要手动把位置喂给它。**

---

## 2.5 Multi-Head Attention 是什么

单头注意力只是在一个表示空间里做一次匹配。

多头注意力的想法是：

> **让模型从多个“角度”同时看序列。**

比如某个 head 更关注：

- 局部关系
- 长距离关系
- 语法模式
- 位置对齐

最后把多个 head 的结果拼起来。

工业实现通常是：

- 先用一个大线性层得到所有 head 的 Q/K/V
- 再 reshape 拆成多个 head

而你这份作业里为了简单，是用多个 `SingleHeadAttention` 拼起来实现的。

这不是最快的实现，但更好理解。

---

## 2.6 Encoder 和 Decoder 分别干什么

### Encoder

Encoder 的任务是：

> **把输入序列编码成一组上下文表示。**

它的每一层通常包含：

1. Multi-Head Self-Attention
2. Add & Norm
3. FeedForward
4. Add & Norm

Encoder 输出的结果通常记作：

```python
enc_out
```

它表示“输入序列已经被理解后的特征”。

### Decoder

Decoder 的任务是：

> **在参考 encoder 输出的同时，逐步生成目标序列。**

Decoder 每层通常包含：

1. Masked Self-Attention
2. Add & Norm
3. Cross-Attention
4. Add & Norm
5. FeedForward
6. Add & Norm

Decoder 比 Encoder 多出来的核心就是：

- **masked self-attention**：防止看未来
- **cross-attention**：去读取 encoder 的输出

---

# 3. 训练和推理主线

这一部分是整份作业最重要的部分。

---

## 3.1 训练主线

训练时大致流程如下：

### 第一步：输入过 Embedding 和位置编码

```python
q_emb = self.emb_layer(ques_b)
a_emb = self.emb_layer(ans_b)
q_emb_inp = q_emb + ques_pos
a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]
```

含义：

- `ques_b`：输入序列 token id
- `ans_b`：目标序列 token id
- `ques_pos / ans_pos`：位置编码

注意这里最关键的一句：

```python
a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]
```

这是因为 decoder 输入的是 **目标序列去掉最后一个 token 的前缀**。

---

### 第二步：输入进 Encoder

在你的代码里：

```1158:1161:D:/CS/UMICH/A5/transformers.py
enc_out = self.encoder(q_emb_inp)
mask = get_subsequent_mask(ans_b[:, :-1])
dec_out = self.decoder(a_emb_inp, enc_out, mask)
```

这里：

- `enc_out = self.encoder(q_emb_inp)`
- 表示 encoder 先把输入序列编码成上下文表示

---

### 第三步：给 Decoder 加 mask

```python
mask = get_subsequent_mask(ans_b[:, :-1])
```

这个 mask 的作用是：

> decoder 当前位置只能看自己和前面，不能看未来。

这一步非常关键，因为训练时如果不 mask，模型会直接看到未来 token，相当于作弊。

---

### 第四步：Decoder 生成输出

```python
dec_out = self.decoder(a_emb_inp, enc_out, mask)
```

这里 decoder 同时依赖两部分信息：

1. `a_emb_inp`：目标前缀
2. `enc_out`：输入序列编码结果

也就是说：

- decoder 先看自己目前已经生成到哪里
- 再去 encoder 输出里查找相关输入信息

---

### 第五步：计算 loss

模型输出后，loss 在训练代码里计算。

你现在的演示文件里是这样的：

```11:23:D:/CS/UMICH/A5/transformer_train_infer_demo.py
def train_one_step(model, optimizer, ques_b, ques_pos, ans_b, ans_pos, use_label_smoothing=False):
    model.train()
    optimizer.zero_grad()

    dec_out = model(ques_b, ques_pos, ans_b, ans_pos)
    pred = dec_out.reshape(-1, dec_out.shape[-1])

    if use_label_smoothing:
        loss = LabelSmoothingLoss(pred, ans_b[:, 1:])
    else:
        target = ans_b[:, 1:].reshape(-1)
        loss = CrossEntropyLoss(pred, target)
```

重点是：

- decoder 输入：`ans_b[:, :-1]`
- 监督标签：`ans_b[:, 1:]`

这就是标准的“错位一位预测”。

---

## 3.2 推理主线

推理和训练最大的区别是：

- 训练时有完整 target，可以 teacher forcing
- 推理时没有完整 target，只能自己一步一步生成

你的演示代码里是这样做的：

```27:47:D:/CS/UMICH/A5/transformer_train_infer_demo.py
@torch.no_grad()
def greedy_decode(model, ques_b, ques_pos, bos_idx, eos_idx=None, max_len=10):
    model.eval()

    q_emb = model.emb_layer(ques_b)
    q_emb_inp = q_emb + ques_pos
    enc_out = model.encoder(q_emb_inp)

    generated = torch.full(
        (ques_b.shape[0], 1), bos_idx, dtype=torch.long, device=ques_b.device
    )

    for _ in range(max_len):
        pos = position_encoding_sinusoid(generated.shape[1], q_emb.shape[-1]).to(
            generated.device
        )
        pos = pos.expand(generated.shape[0], -1, -1)

        dec_inp = model.emb_layer(generated) + pos
        mask = get_subsequent_mask(generated)
        dec_out = model.decoder(dec_inp, enc_out, mask)
```

推理过程可以概括成：

1. 输入先过 encoder 得到 `enc_out`
2. 初始化 decoder 输入为 `[BOS]`
3. 用当前生成序列喂给 decoder
4. 取最后一个位置的预测作为下一个 token
5. 拼接回输入继续生成
6. 直到输出 `EOS` 或达到长度上限

一句话：

> **训练时并行学整段，推理时自回归一步步生成。**

---

# 4. 这份作业最重要的重点和我的难点

这一部分是最值得反复看的。

---

## 难点 1：为什么 `ans_b[:, :-1]` 和 `ans_b[:, 1:]`

这是整个 decoder 训练里最重要、最容易错的点。

假设目标序列是：

```python
[BOS, A, B, C, EOS]
```

那么训练时：

- decoder 输入：`[BOS, A, B, C]`
- 正确标签：`[A, B, C, EOS]`

也就是：

- 看 `BOS`，预测 `A`
- 看 `BOS A`，预测 `B`
- 看 `BOS A B`，预测 `C`
- 看 `BOS A B C`，预测 `EOS`

一句话记忆：

> **输入去尾，标签去头。**

这不是技巧，而是 autoregressive 训练的本质。

---

## 难点 2：为什么 Decoder 一定要 mask

因为训练时你把 target 前缀送进去了。

如果不加 mask，那么当前位置在算 attention 时，会直接看到未来 token。

这样模型学到的就不是“根据过去预测未来”，而是“直接偷看答案”。

所以 decoder 的 self-attention 必须是：

> **masked self-attention**

mask 的作用是：

- 允许看当前位置和过去
- 禁止看未来

---

## 难点 3：为什么 target 还要进入 decoder

这点很容易误解成“是不是作弊”。

其实不是。

训练时把 target 前缀送进 decoder，叫做 **teacher forcing**。

作用是：

- 让模型稳定学习“下一个 token 预测”
- 避免因为前一步预测错了，后面全部崩掉

训练时：

- 用正确前缀教模型

推理时：

- 不再有正确 target
- 只能用自己已经生成出来的 token 继续往后生成

所以：

> **训练时 teacher forcing，推理时 autoregressive。**

---

## 难点 4：Self-Attention 里 `query/key/value` 到底是不是同一个 `x`

对 **self-attention** 来说，通常是的。

也就是：

```python
query = key = value = x
```

然后分别经过不同线性层：

```python
q = self.q(query)
k = self.k(key)
v = self.v(value)
```

虽然输入都是 `x`，但：

- `self.q`
- `self.k`
- `self.v`

是不同线性层，所以最终得到的 `q/k/v` 不一样。

### 那为什么函数接口写三个参数？

因为这样可以兼容 **cross-attention**。

在 cross-attention 里：

```python
Q = decoder_hidden
K = encoder_out
V = encoder_out
```

所以接口写成三个参数，是为了泛化，不是逻辑有问题。

---

## 难点 5：Cross-Attention 到底怎么理解

Cross-Attention 的关键不是公式变了，而是：

> **Q 和 K/V 的来源不同。**

### Self-Attention

```python
Q = x
K = x
V = x
```

### Cross-Attention

```python
Q = decoder_hidden
K = encoder_output
V = encoder_output
```

直觉上可以理解成：

- decoder 现在正在生成输出
- 它需要问：“输入里哪些信息和我现在最相关？”
- 然后去 encoder 输出里查答案

所以：

> **decoder 是带着问题去读 encoder memory。**

---

## 难点 6：LayerNorm 到底是在哪个维度上做

假设输入形状是：

```python
(N, K, M)
```

其中：

- `N`：batch size
- `K`：sequence length
- `M`：embedding dim

LayerNorm 是在最后一维 `M` 上做归一化。

也就是：

- 每个 token 单独处理
- 对它自己的 feature 向量求均值 / 方差

所以统计量形状通常是：

```python
(N, K, 1)
```

不是：

- 不在 batch 上求平均
- 不在 sequence 上求平均

一句话：

> **LayerNorm 是 token 内部归一化，不是跨 token 归一化。**

---

## 难点 7：为什么 LayerNorm 后还要 `gamma` 和 `beta`

标准 LayerNorm 不是只做归一化，而是：

\[
y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

归一化只是把数据拉到均值 0、方差 1。

但模型还需要保留表达能力，所以再加上可学习的：

- `gamma`：缩放
- `beta`：偏移

这样模型可以自己决定最终最适合的分布。

所以：

> **`gamma` 和 `beta` 不是多余的，而是 LayerNorm 的一部分。**

---

## 难点 8：为什么 `gamma` 维度 `(M,)` 能和 `(N, K, M)` 相乘

因为 PyTorch 广播机制是 **右对齐**。

所以：

```python
(M,)
```

会被看成：

```python
(1, 1, M)
```

然后和 `(N, K, M)` 逐元素运算。

这个点虽然小，但在自己写 LayerNorm 时很容易不放心。

---

## 难点 9：为什么 FeedForward 输出维度又变回输入维度

标准 FFN 是：

```python
Linear(inp_dim -> hidden_dim)
ReLU
Linear(hidden_dim -> inp_dim)
```

之所以最后要回到 `inp_dim`，是因为后面还有残差连接：

```python
x + FFN(x)
```

残差相加要求两边形状一致。

所以这是结构上的必需，不是随便设计的。

---

## 难点 10：Multi-Head 为什么这里不是“拆分输入”实现

你在理解时已经发现这一点了：

标准工业实现一般是：

- 一次性得到所有 head 的 Q/K/V
- 再 reshape / transpose 拆成多个 head

但作业里为了教学清晰，采用的是：

- 建多个 `SingleHeadAttention`
- 各自独立计算
- 最后 `torch.cat`

所以：

> **这份作业更偏“概念清晰”，不是“工程最优”。**

---

# 5. 其他容易混的零碎点

## 5.1 `@` 是什么

在 PyTorch 里：

- `*`：逐元素乘法
- `@`：矩阵乘法

在 attention 里，`Q @ K^T` 就是矩阵乘法。

---

## 5.2 `masked_fill_` 是什么

常见写法：

```python
scores.masked_fill_(mask, float('-inf'))
```

意思是：

- `mask == True` 的地方
- 直接填成 `-inf`

这样经过 softmax 后，对应位置权重就是 0。

这是实现 decoder mask 的关键。

---

## 5.3 `QK^T` 像不像协方差矩阵

有点像，因为都是在衡量某种相关性。

但不完全一样，因为：

- 协方差更偏静态统计量
- `QK^T` 是动态、可学习、与当前输入相关的相似度矩阵

可以把它类比成：

> 一个动态相关性矩阵

这样理解就够用了。

---

## 5.4 为什么 Transformer 时间复杂度是二次的

因为 self-attention 要构造：

```python
(K, K)
```

大小的注意力矩阵。

所以复杂度大致是：

\[
O(K^2)
\]

序列长度翻倍，注意力计算量大约变 4 倍。

---

# 6. 一页速记版

如果考试前只看一页，就看这一段。

## 6.1 Transformer 原理一句话

> **每个 token 直接和全序列交互，而不是像 RNN 那样一步步传递隐藏状态。**

## 6.2 Encoder 做什么

> **把输入序列编码成上下文表示 `enc_out`。**

## 6.3 Decoder 做什么

> **根据目标前缀和 `enc_out`，预测下一个 token。**

## 6.4 训练主线

```text
ques_b -> Embedding + ques_pos -> Encoder -> enc_out
ans_b[:, :-1] -> Embedding + ans_pos[:, :-1] -> Decoder(masked)
Decoder 再 cross-attention 到 enc_out
输出 logits
和 ans_b[:, 1:] 算 loss
```

## 6.5 推理主线

```text
输入先过 Encoder 得到 enc_out
Decoder 从 BOS 开始
每次生成一个 token
拼回输入继续生成
直到 EOS
```

## 6.6 最关键的 4 个考点

1. `ans_b[:, :-1]` 和 `ans_b[:, 1:]`
2. decoder 必须加 subsequent mask
3. self-attention 和 cross-attention 区别
4. LayerNorm 是在最后一维做

---

# 7. 复习建议

如果你接下来继续看 `A5/transformers.py`，建议按这个顺序复习：

1. `scaled_dot_product_no_loop_batch`
2. `SelfAttention`
3. `MultiHeadAttention`
4. `LayerNormalization`
5. `FeedForwardBlock`
6. `EncoderBlock`
7. `DecoderBlock`
8. `Transformer.forward`
9. `CrossEntropyLoss / LabelSmoothingLoss`
10. `greedy_decode`

因为真正难的不是每个函数单独看懂，而是：

> **把这些模块串成一条“训练 / 推理”的完整链。**

---

# 8. 最后一句总结

这份 A5 真正最重要的，不是背公式，而是把这条主线彻底打通：

> **输入 embedding + 位置编码 -> encoder 编码 -> target 前缀进 decoder -> mask 防偷看 -> cross-attention 查 encoder 输出 -> 预测下一个 token -> 训练时和右移标签算 loss，推理时一步一步自回归生成。**

只要这条主线你能自己讲出来，Transformer 这一部分就真的理解了。
