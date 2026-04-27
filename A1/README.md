# A1 课程作业总结：KNN 与 PyTorch 基础

## 概述

A1 是整门课的真正起点。它没有一上来就讲复杂神经网络，而是先用两条最基础、但也最关键的主线建立后续所有作业的底层能力：

1. 用 `K-Nearest Neighbor` 建立对“数据、距离、分类决策”的直觉。
2. 用 `PyTorch` 基础练习建立张量操作、索引、reshape、向量化和批处理思维。

如果说后面的 A2、A3、A4、A5、A6 都是在“搭模型、训练模型、分析模型”，那么 A1 做的事情更像是先把最基本的工具磨利：

> 学会把问题写成张量运算，而不是停留在 Python 循环层面。

---

## 目录结构

```text
A1/
├── knn.ipynb
├── knn.py
├── pytorch101.ipynb
├── pytorch101.py
├── test.py
└── eecs598/
```

其中最核心的是：

- `knn.py`：KNN 分类器实现
- `pytorch101.py`：PyTorch 张量基础练习
- `knn.ipynb` / `pytorch101.ipynb`：对应 notebook 版本，便于交互式实验

---

## 1. KNN 主线：从“距离”到“分类”

### 1.1 KNN 在学什么

`knn.py` 的核心目标不是让人记住一个老模型，而是训练一种非常重要的计算思维：

- 把样本看成向量
- 用距离度量样本之间的相似性
- 用最邻近样本的标签做预测
- 把循环实现一步步改写成张量并行计算

这部分主线非常清晰：

1. 计算训练集和测试集之间的两两距离
2. 根据最近的 `k` 个样本投票预测标签
3. 用交叉验证选择最佳 `k`

---

### 1.2 三种距离计算方式

A1 最值得反复看的地方，是同一个问题被写成了三种复杂度逐步下降的版本：

#### `compute_distances_two_loops`
- 对训练样本和测试样本分别套两层循环
- 最直观
- 最容易验证逻辑是否正确
- 但效率最低

#### `compute_distances_one_loop`
- 只保留一层对训练样本的循环
- 测试样本维度改成并行计算
- 开始体现向量化思想

#### `compute_distances_no_loops`
- 完全不使用 Python 显式循环
- 使用平方展开公式：

\[
\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2x^Ty
\]

- 这是 A1 最核心的优化思想

这一部分真正训练的是：

> 同一个数学公式，如何改写成更适合张量库执行的形式。

---

### 1.3 KNN 预测与投票

`predict_labels` 做的是：

1. 对每个测试样本找到最近的 `k` 个训练样本
2. 取出这 `k` 个邻居的标签
3. 做多数投票
4. 平票时选更小的标签

这里表面是在写分类器，实际是在训练以下能力：

- `topk` 的使用
- 高维索引
- 计数与投票逻辑
- 批量处理多个测试样本

---

### 1.4 类封装与交叉验证

`KnnClassifier` 把训练数据缓存到类中，再通过：

- `predict`
- `check_accuracy`
- `knn_cross_validate`
- `knn_get_best_k`

把一个完整的机器学习小流程串了起来。

这部分的重要性在于：

- 不再只是写零散函数
- 开始理解“训练集 / 验证集 / 测试集”的角色
- 理解超参数 `k` 不是拍脑袋定，而是要通过验证集选择

虽然 KNN 本身很简单，但这里已经把后续所有模型都会用到的实验流程提前演练了一遍。

---

## 2. PyTorch 101 主线：从张量操作到向量化思维

### 2.1 `pytorch101.py` 在练什么

`pytorch101.py` 更像是一份系统化的张量操作训练单。它覆盖的内容看起来碎，但其实是按能力层次层层推进的：

#### 基础创建
- 创建张量
- 指定 shape
- 指定 dtype
- 原地修改元素

对应函数例如：
- `create_sample_tensor`
- `mutate_tensor`
- `create_tensor_of_pi`
- `multiples_of_ten`

---

### 2.2 形状与元素个数

A1 很早就让人意识到：

> 你处理的不是“数字”，而是“带 shape 的数据”。

像 `count_tensor_elements` 这样的题虽然简单，但它会强迫你意识到：

- 张量的本质是多维数组
- shape 决定了数据如何被解释
- 后面所有网络前向传播，本质上都在管理 shape

这为后续卷积、全连接、batch 处理打基础。

---

### 2.3 切片、索引与高级索引

这一部分是 PyTorch 初学者最容易出错、但也是必须过关的地方。

对应练习包括：

- `slice_indexing_practice`
- `slice_assignment_practice`
- `shuffle_cols`
- `reverse_rows`
- `take_one_elem_per_col`

它们训练的核心能力包括：

- 行列切片
- 保持维度还是降维
- 整数数组索引
- 原地赋值
- 用尽量少的操作完成复杂数据重排

后面不管是 CNN feature map 处理、sequence mask、attention 索引，还是 batch 数据重组，这些能力都会反复出现。

---

### 2.4 One-hot、布尔索引与归约

例如：

- `make_one_hot`
- `sum_positive_entries`
- `zero_row_min`

这些题的意义在于训练：

- 如何根据索引批量写入
- 如何用布尔掩码筛选数据
- 如何做 `sum / min / argmin` 之类的 reduction
- 如何将 reduction 结果再用于反向索引修改

这已经是非常接近真实深度学习代码的基本功了。

---

### 2.5 reshape / transpose / batched matrix multiply

这一组内容尤其重要，因为它直接连接到后续神经网络实现：

- `reshape_practice`
- `batched_matrix_multiply`
- 以及后续相关函数

这部分真正训练的是：

- 视图变换而不是复制数据
- 维度换位
- batch 维度的统一处理
- 用矩阵乘法表达更复杂的批量运算

很多人第一次学深度学习时，真正卡住的不是公式，而是：

- 为什么这里要 `reshape`
- 为什么这里要 `transpose`
- 为什么 batch matmul 的维度对不上

A1 就是在最早阶段先把这些坑踩一遍。

---

## 3. A1 的核心收获

如果只用一句话概括 A1，那么它训练的是：

> 把“逐个样本、逐个元素”的朴素代码，改写成“按批处理、按张量并行”的高效实现。

更具体一点，A1 为后续作业提供了这些底层能力：

### 对后续 A2 的帮助
- 能理解向量化 loss 计算
- 能理解梯度张量的 shape
- 能写出批量训练代码

### 对后续 A3 / A4 的帮助
- 能看懂卷积输出维度变化
- 能理解 feature map reshape / flatten
- 能处理更复杂的 indexing 和 broadcasting

### 对后续 A5 / A6 的帮助
- 能理解 batch-first 的序列张量
- 能看懂注意力中的矩阵乘法
- 能处理生成模型中的多维张量运算

---

## 4. A1 复习建议

推荐按下面顺序复习：

### 第一遍：先看 `knn.py`
重点看：
- 三种距离实现
- `no_loops` 版本的数学展开
- `predict_labels`
- `cross_validate`

目标：建立“向量化比循环更重要”的意识。

### 第二遍：系统刷 `pytorch101.py`
重点看：
- slice / indexing
- boolean mask
- reshape / transpose
- batched matrix multiply

目标：熟悉 PyTorch 张量操作的常见写法。

### 第三遍：回头总结易错点
尤其注意：
- shape 是否符合预期
- 索引后维度有没有丢
- `view/reshape` 前后元素顺序是否变了
- reduction 后结果如何再参与索引

---

## 5. 这份作业的定位

A1 的难点不在模型，而在“表达方式”。

它让人从：

- 用 Python 循环处理样本

过渡到：

- 用张量一次处理一批样本

这是后续所有深度学习代码真正的共同语言。

所以 A1 虽然看起来“只是 KNN + PyTorch 入门”，但它其实决定了后面写代码时你是停留在脚本层面，还是开始真正进入数值计算与深度学习实现的思维方式。
