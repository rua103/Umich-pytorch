# A2 线性分类器 / 两层网络 知识点整理

此文档是你 `d:\CS\UMICH\A2` 作业的结构化总结，按章节划分便于复盘。

- **章节 1**：Structured SVM 向量化实现（详细推导 + 代码 + 常见错误）
- **章节 2**：两层网络实现与训练
- **章节 3**：工具函数与实验
- **章节 4**：作业复盘建议

---

## 1. 线性分类器（Structured SVM - 向量化实现）

### 1.0 问题定义

给定：
- 数据：$X \in \mathbb{R}^{N \times D}$（N 个样本，每个 D 维）
- 权重：$W \in \mathbb{R}^{D \times C}$（D 维到 C 个类别）
- 标签：$y = [0, 1, ..., N-1]$（每个样本的正确类别索引）

目标：计算 **loss** 和 **梯度 dW**

---

### 1.1 数学公式

**Score（得分）：**
$$s = XW \quad \text{形状: } [N, C]$$

**Hinge Loss（SVM 损失）：**
$$L = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \ne y_i} \max(0, s_{ij} - s_{iy_i} + 1)$$

**梯度结论**：对于每个样本 $i$，若 margin > 0，则：
- 错误类别 $j$：贡献 +1 到 $\frac{\partial L}{\partial s}$
- 正确类别 $y_i$：贡献 $-k$（$k =$ 被违反类别数）

---

### 1.2 变量形状表（牢记）

| 变量 | 形状 | 含义 |
|------|------|------|
| X | (N, D) | N 个 D 维特征 |
| W | (D, C) | 权重矩阵 |
| scores | (N, C) | 预测得分 |
| correct_scores | (N, 1) | 真实类的得分（已 reshape） |
| margins | (N, C) | hinge margin |
| mask | (N, C) | 梯度权重指示 |
| dW | (D, C) | 权重梯度 |

---

### 1.3 向量化实现步骤

**Step 1：计算 scores**
```python
scores = X.mm(W)  # [N,D] @ [D,C] -> [N,C]
```

**Step 2：提取正确类别分数（⚠️ 必须 reshape）**
```python
correct_scores = scores[torch.arange(N), y].view(-1, 1)  # [N,1]
```

**Step 3：计算 margins**
```python
margins = scores - correct_scores + 1  # [N,C]
```

**Step 4：清除正确类别（hinge loss 定义）**
```python
margins[torch.arange(N), y] = 0
```

**Step 5：应用 max(0, ·)（核心）**
```python
margins = torch.clamp(margins, min=0)
```

**Step 6：计算损失**
```python
loss = margins.sum() / N
loss += reg * torch.sum(W * W)  # 加入 L2 正则
```

---

### 1.4 梯度计算（最关键）

**Step 7：构造 mask（梯度的权重矩阵）**
```python
mask = (margins > 0).float()  # [N,C]
```

**Step 8：统计每个样本的违规数**
```python
row_sum = mask.sum(dim=1)  # [N]
```

**Step 9：修正正确类别（⚠️ 最常出错的地方）**
```python
mask[torch.arange(N), y] = -row_sum  # 正确类设为负累计
```

**直觉**：对每个样本 $i$，所有被违反的类贡献 $+X[i]$，正确类贡献 $-k \times X[i]$

**Step 10：矩阵聚合（链式法则）**
$$\frac{\partial L}{\partial W} = X^T \cdot \text{mask}$$

```python
dW = X.t().mm(mask) / N  # [D,C]
dW += 2 * reg * W        # 加入正则梯度
```

---

### 1.5 梯度直觉理解

由链式法则：$\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial s}$

- **mask** 就是 $\frac{\partial L}{\partial s}$（scores 对 loss 的梯度）
- **X.t().mm(mask)** 把所有样本对权重的贡献汇总

---

### 1.6 常见错误（高频）

| 错误 | 后果 | 正确做法 |
|------|------|---------|
| `scores[range(N), y]` 不 reshape | shape [N] 广播错误 | `.view(-1, 1)` |
| 忘记清 `margins[y] = 0` | 梯度计算错误 | 必须清除正确类 |
| 没有 clamp `max(0, ·)` | loss 为负 | `torch.clamp(margins, min=0)` |
| mask 没修正正确类 | dW 完全错误 | `mask[y] = -row_sum` |
| 正则项系数错 | 正则化作用不当 | loss `+ reg*sum(W²)`，dW `+ 2*reg*W` |
| 忘记除以 N | 数值尺度错误 | `loss/N`, `dW/N` |

---

### 1.7 完整 PyTorch 代码

```python
def svm_loss_vectorized(W, X, y, reg):
    """
    Vectorized SVM loss and gradient
    
    Args:
        W: [D, C] weight matrix
        X: [N, D] data
        y: [N] labels
        reg: regularization strength
    
    Returns:
        loss: scalar
        dW: [D, C] gradient
    """
    N = X.shape[0]
    
    # Forward pass
    scores = X.mm(W)                              # [N, C]
    correct_scores = scores[torch.arange(N), y].view(-1, 1)  # [N, 1]
    
    margins = scores - correct_scores + 1        # [N, C]
    margins[torch.arange(N), y] = 0              # Clear correct class
    margins = torch.clamp(margins, min=0)        # Apply max(0, ·)
    
    loss = margins.sum() / N
    loss += reg * torch.sum(W * W)
    
    # Backward pass
    mask = (margins > 0).float()                 # [N, C]
    row_sum = mask.sum(dim=1)                    # [N]
    mask[torch.arange(N), y] = -row_sum          # Correct class = -k
    
    dW = X.t().mm(mask) / N
    dW += 2 * reg * W
    
    return loss, dW
```

---

### 1.8 一句话总结

✅ **SVM = max(0, ·) + mask 控制梯度 + $X^T$ 汇总贡献**

---

### 1.9 核心记忆点（考前必看）

1. **margins = max(0, score差 + 1)**
2. **正确类不算（清除）**
3. **mask：错误类=+1，正确类=−k**
4. **dW = $X^T$ @ mask**
5. **除以 N，加正则**

---

### 1.10 验证方法：naive vs vectorized

```python
# 验证两个实现结果一致
loss_naive, dW_naive = svm_loss_naive(W, X_small, y_small, reg)
loss_vec, dW_vec = svm_loss_vectorized(W, X_small, y_small, reg)

assert torch.allclose(loss_naive, loss_vec, atol=1e-5)
assert torch.allclose(dW_naive, dW_vec, atol=1e-5)
```

---

### 1.11 性能对比

- **naive**: $O(N \cdot C)$ 循环，Python 层开销大
- **vectorized**: 同样复杂度，利用 PyTorch 底层优化，通常 **10-100x 更快**

---

## 2. Softmax 分类器（`linear_classifier.py`）

### 2.0 问题设定

| 符号 | 形状 | 含义 |
|------|------|------|
| X | (N, D) | N 个样本，每个 D 维 |
| W | (D, C) | 权重矩阵 |
| y | (N,) | 标签（0 到 C-1） |
| scores | (N, C) | 原始得分 `X @ W` |
| prob | (N, C) | Softmax 概率 |

---

### 2.1 Softmax 定义

对第 $i$ 个样本，第 $j$ 个类别的得分 $s_j = X_i^T W_j$，Softmax 概率：

$$p_j = \frac{e^{s_j}}{\sum_k e^{s_k}}$$

**数值稳定处理**（减去每行最大值，不影响结果）：

```python
scores = X.mm(W)                                          # (N, C)
scores -= scores.max(dim=1, keepdim=True).values          # 数值稳定
exp_scores = torch.exp(scores)
prob = exp_scores / exp_scores.sum(dim=1, keepdim=True)   # (N, C)
```

---

### 2.2 Cross-Entropy Loss

单个样本：
$$L_i = -\log(p_{i, y_i})$$

向量化（对所有样本取均值）：

```python
loss = -torch.log(prob[torch.arange(N), y]).mean()
loss += reg * torch.sum(W * W)   # L2 正则（无 1/2 系数）
```

---

### 2.3 梯度推导（完整链式法则）

**① Loss 对 scores 的梯度（关键结论）：**

$$\frac{\partial L_i}{\partial s_j} = \begin{cases} p_j - 1 & j = y_i \\ p_j & j \ne y_i \end{cases}$$

统一写法：$\dfrac{\partial L}{\partial s} = P - Y_{\text{onehot}}$

**② scores 对 W 的梯度：**
$$\frac{\partial s_j}{\partial W_j} = X_i$$

**③ 链式法则合并，对所有样本聚合：**
$$\frac{\partial L}{\partial W} = X^T (P - Y_{\text{onehot}})$$

---

### 2.4 Vectorized 梯度实现

```python
dscores = prob.clone()                    # (N, C)，复制概率矩阵
dscores[torch.arange(N), y] -= 1         # 正确类减 1（P - Y_onehot）
dscores /= N                             # 对应 loss.mean() 的除 N

dW = X.t().mm(dscores)                   # X^T @ dscores -> (D, C)
dW += 2 * reg * W                        # 加正则梯度
```

---

### 2.5 Shape 流动

```
X         (N, D)
W         (D, C)
scores    (N, C)   = X @ W
prob      (N, C)   softmax(scores)
dscores   (N, C)   prob - Y_onehot，除以 N
X.T       (D, N)
dW        (D, C)   = X.T @ dscores
```

---

### 2.6 完整 softmax_loss_vectorized 实现

```python
def softmax_loss_vectorized(W, X, y, reg):
    N = X.shape[0]

    # 前向传播
    scores = X.mm(W)                                         # (N, C)
    scores -= scores.max(dim=1, keepdim=True).values         # 数值稳定
    exp_scores = torch.exp(scores)
    prob = exp_scores / exp_scores.sum(dim=1, keepdim=True)  # (N, C)

    # 损失
    loss = -torch.log(prob[torch.arange(N), y]).mean()
    loss += reg * torch.sum(W * W)

    # 梯度
    dscores = prob.clone()
    dscores[torch.arange(N), y] -= 1
    dscores /= N

    dW = X.t().mm(dscores)
    dW += 2 * reg * W

    return loss, dW
```

---

### 2.7 SVM vs Softmax 对比

| 对比项 | SVM (Hinge Loss) | Softmax (Cross-Entropy) |
|--------|-----------------|------------------------|
| Loss 形式 | $\max(0, s_j - s_{y_i} + 1)$ | $-\log(p_{y_i})$ |
| 梯度来源 | margin > 0 的类 | 所有类（通过概率分布）|
| 正确类梯度 | $-k$（违反类数） | $p_{y_i} - 1$ |
| 错误类梯度 | $+1$（若违反） | $p_j$ |
| 核心操作 | `torch.clamp` + mask | softmax + log |
| 数值稳定 | 不需要 | 需要减最大值 |

---

### 2.8 常见错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 不做数值稳定 | `exp` 溢出为 `inf` | 减每行最大值 |
| 梯度不除以 N | 梯度尺度错误 | `dscores /= N` |
| 忘记 `prob.clone()` | 修改 prob 影响 loss | 用 `.clone()` 拷贝 |
| 正则系数乘 1/2 | 正则梯度差 2 倍 | 直接 `reg * sum(W²)` |
| `dscores[y] -= 1` 放错位置 | 梯度完全错误 | 必须在除 N 之前修改 |

---

### 2.9 核心记忆点

1. **Softmax 梯度 = prob − onehot**
2. **代码：`dscores = prob.clone(); dscores[range(N), y] -= 1`**
3. **除以 N 对应 loss.mean()**
4. **数值稳定：减每行最大值**
5. **dW = X.T @ dscores**

---

## 3. 两层网络（`two_layer_net.py`）

### 3.1 模型与参数结构
- `W1`: (D, H)
- `b1`: (H,)
- `W2`: (H, C)
- `b2`: (C,)

### 3.2 前向函数 `nn_forward_pass`
- 计算隐层：`h1 = X.mm(W1) + b1`
- ReLU：`hidden = torch.clamp(h1, min=0)` 或 `torch.relu(h1)`
- 输出得分：`scores = hidden.mm(W2) + b2`

### 3.3 损失与反向 `nn_forward_backward`
- 用 Softmax + 交叉熵
- 数值稳定：`scores -= scores.max(dim=1, keepdim=True).values`
- 概率：`probs = torch.exp(scores) / torch.sum(torch.exp(scores), dim=1, keepdim=True)`
- 损失：`-torch.sum(torch.log(probs[torch.arange(N), y]))/N + reg*(torch.sum(W1*W1)+torch.sum(W2*W2))`
- 反向：先调整 `probs`，再计算 `W2/b2/h1/b1` 的梯度

### 3.4 训练循环 `nn_train`
- minibatch + `sample_batch`
- 每次迭代计算 loss、grads
- 参数更新：`param -= learning_rate * grads[param]`
- 每 epoch 计算精度、学习率衰减

### 3.5 预测 `nn_predict`
- 得分最大类：`torch.argmax(scores, dim=1)`

---



## 5. 快速索引

| 知识点 | 位置 | 关键公式 |
|--------|------|---------|
| Hinge Loss | 1.1 | $\max(0, s_j - s_{y_i} + 1)$ |
| Mask 构造 | 1.4 | `mask[y] = -row_sum` |
| 梯度汇总 | 1.4 | $X^T @ \text{mask}$ |
| Speed-up | 1.11 | 向量化 → 10-100x 加速 |
| Softmax | 2.3 | 交叉熵 loss + 反向 |
| 验证方法 | 1.10 | `torch.allclose()` |

---

**🎯 你的目标（"把复杂的双循环梯度转化为矩阵运算"）完整对应：**
> 章节 1 的 **mask 构造** + **$X^T$ 汇总** = 向量化
