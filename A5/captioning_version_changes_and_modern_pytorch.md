# UMICH Captioning 版本变化与现代 PyTorch 架构说明

## 适用范围

这份文档面向你当前的学习路径：

- 已完成：`2020fa/2020FA_A4/rnn_lstm_attention_captioning.py`
- 准备开始：`A5/rnn_lstm_captioning.py`

目标是回答两个问题：

1. 新旧两个版本具体变化在哪里。
2. 当前版本体现了哪些更“现代化”的 PyTorch 架构写法。

---

## 一、先给结论

### 结论 1：核心知识没有变

两份作业的核心仍然是同一条主线：

- 图像特征提取
- RNN / LSTM / attention captioning
- 词嵌入
- temporal softmax loss
- caption 生成

所以你**不用从零重新学一遍**。

### 结论 2：代码组织方式明显现代化了

当前版本不是单纯“换个文件名”，而是把很多写法改成了更接近现代 PyTorch 工程习惯的形式，例如：

- 更明确地用 `nn.Module` 封装子模块
- 用 `nn.Parameter` / 子模块注册参数
- 用 `torchvision.models.feature_extraction.create_feature_extractor`
- 用类型标注（type hints）
- 更清晰地区分“图像编码器”和“caption 模型本体”
- 更偏 device-agnostic / reusable 的设计

### 结论 3：你最适合的做法是“迁移式完成”

也就是：

- 不必把旧版全部重做
- 但要按当前版的 notebook 和 `.py` 文件把接口、结构、编码器部分重新过一遍

---

## 二、新旧版本的主要变化

## 1. 文件定位变化

旧版文件：

- `2020fa/2020FA_A4/rnn_lstm_attention_captioning.py`
- `2020fa/2020FA_A4/rnn_lstm_attention_captioning.ipynb`

当前版文件：

- `A5/rnn_lstm_captioning.py`
- `A5/rnn_lstm_captioning.ipynb`

### 变化含义

旧版把 captioning 放在 A4 里；当前版把它挪到了 A5，说明课程结构重组了。

当前版 A5 下面还有 `transformers.py`，这说明课程希望你把：

- RNN / LSTM captioning
- Transformer

看作同一条“序列建模主线”。

---

## 2. Image Encoder 的设计变化最大

### 旧版：`FeatureExtractor(object)`

旧版使用的是：

- `MobileNet v2`
- 自己手动做 preprocess
- 手动调用 backbone
- 用 `extract_mobilenet_feature` 返回特征

它更像“作业工具类”。

### 当前版：`ImageEncoder(nn.Module)`

当前版改成：

- `RegNet-X 400MF`
- `ImageEncoder` 继承自 `nn.Module`
- 用 `create_feature_extractor(...)` 抽取中间层
- 暴露 `out_channels` 属性
- `forward()` 直接处理输入并返回 feature map

### 这意味着什么

这是最典型的“现代 PyTorch 化”改造：

#### 旧版思路
- “写一个帮助类，里面包一个预训练网络”

#### 当前版思路
- “把图像编码器作为模型系统中的一个标准模块”

也就是说，当前版更强调：

> 编码器本身就是模型图中的一部分，而不是一个外部辅助脚本。

---

## 3. 预训练 backbone 从 MobileNet 变成了 RegNet

### 旧版 notebook 描述
旧版写的是：

- 使用 `MobileNet v2` 提取图像特征

### 当前版 notebook 描述
当前版写的是：

- 使用 `RegNet-X 400MF` 提取图像特征

### 含义

这不仅是换模型，更反映了课程更新后的默认风格：

- 更依赖 torchvision 官方 backbone
- 更强调从中间层抽特征图
- 更自然支持 attention captioning 需要的 spatial features

---

## 4. 从“面向作业脚本”转向“面向模块组合”

### 旧版倾向
旧版很多部分更像经典课程作业风格：

- 单个函数一个 TODO
- 工具类和模型类混在一起
- 参数初始化常常显式传 device / dtype
- 比较强调手工控制每一步

### 当前版倾向
当前版更像现代 PyTorch 项目：

- 编码器是模块
- RNN/LSTM/Attention 模块边界更明确
- `CaptioningRNN` 更像总装模型
- `ignore_index`、`cell_type`、`image_encoder_pretrained` 这些参数更像真实项目接口

---

## 5. device / dtype 处理方式更现代

### 旧版写法特点
旧版很多类初始化时都显式传：

- `device='cpu'`
- `dtype=torch.float32`

例如 `RNN(...)`、`WordEmbedding(...)`、`FeatureExtractor(...)` 都把这些参数直接写进初始化。

这种写法在教学上很好理解，但工程上稍显笨重。

### 当前版写法特点
当前版更倾向：

- 让模块先自然创建
- 通过外部统一控制 device
- 在 forward 中尽量根据输入 tensor 自动匹配 dtype / device

例如 `ImageEncoder.forward()` 里会根据 backbone 权重 dtype 来处理输入。

### 为什么这更现代

因为现在更推荐：

- 模型本身尽量少写死 device
- 通过外部 `model.to(device)` 或训练脚本统一搬运
- 新建 tensor 时尽量跟随已有 tensor 的设备和类型

这就是所谓的 **device-agnostic code**。

---

## 6. 类型标注更完整

当前版明显加入了大量 type hints，例如：

- `pretrained: bool = True`
- `verbose: bool = True`
- `images: torch.Tensor`
- `input_dim: int = 512`
- `ignore_index: Optional[int] = None`

旧版基本没有这类现代 Python 风格的标注。

### 意义

类型标注的价值主要是：

- 代码更容易读
- IDE 补全和静态检查更友好
- 模块边界更清晰

这也是现代 PyTorch / Python 项目常见风格。

---

## 7. 当前版 notebook 更像“从 captioning 过渡到 Transformer”

旧版 captioning 在 A4 里，与 style transfer、network visualization 并列。

当前版则把 captioning 单独放在 A5，并紧接着 `transformers.py`。

### 这意味着

当前版不只是让你会写 RNN/LSTM，而是在课程编排上暗示：

- 先掌握 sequence modeling 的经典路线
- 再进入 Transformer

所以当前版 `rnn_lstm_captioning` 的教学定位更像：

> “RNN/LSTM 作为进入现代序列模型的前置台阶”

而不仅仅是“视觉里的一个应用”。

---

## 三、哪些内容你可以直接复用

## 1. 可以高复用的知识点

你已经做过旧版后，下面这些几乎都可以直接迁移：

- `rnn_step_forward`
- `rnn_step_backward`
- `rnn_forward`
- `rnn_backward`
- `WordEmbedding`
- `temporal_softmax_loss`
- LSTM 单步前向/反向
- LSTM 序列前向/反向
- captioning 的训练主线
- sampling / inference 的基本逻辑

这些属于“算法本体”，不会因为版本变动而本质改变。

---

## 2. 需要重新适应的部分

### 必须重新看

#### (1) `ImageEncoder`
这是变化最大的部分。

#### (2) `CaptioningRNN.__init__` 中的模块组织
当前版更模块化，接口也更清晰。

#### (3) 当前版 notebook 中 attention 的组织方式
旧版文件名直接写了 `attention`，当前版标题更偏 `rnn_lstm_captioning`，所以要以当前版 notebook 的要求为准。

#### (4) helper / data loading 的名字变化
例如：

- 旧版：`load_COCO`
- 当前版：`load_coco_captions`

这是课程代码清理的一部分。

---

## 四、什么叫“现代化的 PyTorch 架构”

这里我专门用你这个作业的上下文来解释，不讲太抽象的概念。

## 1. 把每个功能块做成独立 `nn.Module`

现代 PyTorch 很强调模块边界。

例如一个 captioning 系统，通常应该拆成：

- `ImageEncoder`
- `WordEmbedding`
- `RNN` / `LSTM` / `AttentionLSTM`
- `output projection`
- 顶层 `CaptioningRNN`

这样做的好处：

- 更容易替换 backbone
- 更容易复用编码器
- 更容易测试单个模块
- 更容易扩展到新任务

### 不太现代的写法
- 一个大类里什么都做
- 特征提取写成外部脚本函数
- forward 之外塞很多临时逻辑

### 更现代的写法
- 每层职责明确
- 顶层模块只负责把子模块串起来

---

## 2. 参数通过模块自动注册，而不是手工散落

现代 PyTorch 中，参数通常通过这几种方式出现：

- `nn.Linear`
- `nn.Conv2d`
- `nn.Embedding`
- `nn.Parameter`

这样 `model.parameters()` 就能自动拿到所有可训练参数。

当前版整体风格更接近这个方向。

---

## 3. backbone 不是黑盒辅助函数，而是模型图中的一环

旧版的 `FeatureExtractor` 更像外部预处理器。

当前版的 `ImageEncoder(nn.Module)` 更符合现代架构习惯：

- 编码器和解码器都属于模型的一部分
- forward 里直接连起来
- 可以统一冻结 / 解冻参数
- 可以直接接入训练脚本

这在真实项目中特别重要。

---

## 4. 用 torchvision 官方工具提取中间层特征

当前版用了：

- `torchvision.models.feature_extraction.create_feature_extractor`

这是很现代的官方方案。

### 为什么它比手工切 backbone 更好

- 可读性更高
- 不需要手动切很多层
- 结构变化时更稳健
- 更适合 feature map extraction / FPN / attention 场景

这已经是现代 torchvision 项目里非常常见的套路。

---

## 5. 更少依赖写死的 device / `.cuda()` / `.to()` 散落在实现里

现代 PyTorch 强调：

- 实现块里不要到处写 `.cuda()`
- 不要把 device 逻辑散落在每个函数里
- 尽量根据输入 tensor 自动继承设备

这样代码会：

- 更可移植
- 更容易在 CPU / GPU 间切换
- 更适合 notebook 和脚本复用

---

## 6. forward 只做“本模块职责内”的事情

现代架构里，`forward()` 应该尽量清晰地表示：

> 输入是什么，经过哪些子模块，输出什么。

而不是在里面混很多：

- 数据下载
- 设备搬运
- 训练循环逻辑
- 杂项预处理

当前版相对旧版就在往这个方向靠拢。

---

## 五、你接下来应该怎么利用这些变化

## 最推荐的实际做法

### 方案：迁移式完成当前版 A5

1. 打开 `A5/rnn_lstm_captioning.ipynb`
2. 对照 `A5/rnn_lstm_captioning.py`
3. 把你旧版已经掌握的函数快速过掉
4. 把重点放在：
   - `ImageEncoder`
   - 模块初始化组织
   - 当前版要求实现的 forward / sample / attention 部分

### 不建议

- 把旧版答案机械复制过去
- 把当前版当成“完全新作业”从零死磕

最佳方式是：

> 用旧版理解做支撑，用当前版结构完成真正迁移。

---

## 六、如果你不是为了目标检测，那最优主线是什么

你已经明确说过你学习这门课的目的不是目标检测，所以推荐主线是：

1. `A5/rnn_lstm_captioning.py`
2. `A5/transformers.py`
3. `A6/vae.py`
4. `A6/gan.py`
5. 可选：`A6/network_visualization.py`
6. 可选：`A6/style_transfer.py`

检测部分：

- `A4/one_stage_detector.py`
- `A4/two_stage_detector.py`

可以后置，甚至不做也行。

---

## 七、最后给你的最短建议

### 你现在应该怎么做

- 不必重做旧版 captioning
- 直接开始当前版 `A5/rnn_lstm_captioning`
- 把它当成一次“现代 PyTorch 结构迁移练习”

### 你最该关注的变化

- `FeatureExtractor(object)` -> `ImageEncoder(nn.Module)`
- `MobileNet` -> `RegNet`
- 外部工具式特征提取 -> 模块化编码器
- 显式 device/dtype 教学风格 -> 更自然的 device-agnostic 风格
- 作业式脚本结构 -> 更接近真实项目的模块组织

---

## 八、给你的行动建议

你可以立刻按这个顺序开始：

### 第一步
通读：
- `A5/rnn_lstm_captioning.ipynb`

### 第二步
对照：
- `A5/rnn_lstm_captioning.py`
- `2020fa/2020FA_A4/rnn_lstm_attention_captioning.py`

### 第三步
优先攻克：
- `ImageEncoder`
- `CaptioningRNN.__init__`
- 当前版中任何你没见过的 attention / sampling 组织方式

---

如果你后面愿意，我还可以继续给你补一份：

# 《A5/rnn_lstm_captioning 迁移清单》

内容会具体到：

- 哪些函数可以直接复用旧版思路
- 哪些函数要重新做
- 哪些地方最体现现代 PyTorch 写法
- 你应该按什么顺序实现，效率最高

