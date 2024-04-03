## 代码说明
### 环境配置

Python 版本：3.8
PyTorch 版本：1.10
CUDA 版本：11.3

所需要的环境已经在`requirements.txt`中定义

### 数据

* 使用大赛提供的有标注数据（10万）进行模型的训练
* 使用大赛提供的无标签数据（100万）用于模型的预训练
* 没有使用任何额外数据集

### 预训练模型

* 使用了 huggingface 上提供的 `hfl/chinese-roberta-wwm-ext` 模型作为单流模型中的特征提取模型。链接为： https://huggingface.co/hfl/chinese-roberta-wwm-ext
* 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型作为预训练中的特征提取模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

### 算法描述

#### 单流模型
* 对提供的文本特征`title、asr以及ocr中的text`进行直接的拼接，长度限制为330
* 对于拼接的文本特征，使用`roberta`模型进行特征提取，然后过bert中embedding层进行映射
* 对于视觉特征，直接过一层dense后记性ReLU激活，然后过与文本相同的bert中embedding层进行映射，保证和视频特征和文本特征在相同的维度
* 视觉和文本特征的embeddings和mask均是直接拼接，然后对mask进行取反操作，去除mask中的[PAD]
* 对拼接后的文本和视频特征过bert的encoder进行编码，得到最后一层的隐层状态last hidden states,然后再将输出送入MeanPooling（）中进行平均池化
* 对池化后的结果过单层MLP结构，然后去预测二级分类的 id.

#### 模型预训练
* 使用了mlm和mfm预训练任务，并额外增加了video_embedding对视频特征进行提取，整体结构与单流模型相似

#### 模型融合
* 分别取单流模型和预训练模型的10折进行融合

### 性能

离线测试性能：0.668（A榜）
B榜测试性能：0.687（B榜）

### 训练流程
* 使用无标签数据，取1%作为验证集进行预训练
* 使用单流模型在有标签数据上进行十折交叉验证，然后获取每一折中最好的模型
* 使用预训练得到的模型在有标签任务上进行微调，也是取十折


### 测试流程

* 单流模型：采用sklearn中提供的`StratifiedKFold`进行十折交叉验证，取每一折中最好的模型，然后对获取的十折最好的模型进行等比例融合预测
* 预训练：采用单流模型同样的方式进行十折交叉验证，得到等比例融合的预测概率
* 将单流模型的结果和预训练模型的结果进行加权平均，作为最终的结果
