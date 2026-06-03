# 自监督学习大作业报告模板

## 1. 任务目标

本项目探索 Tiny ImageNet 上的自监督预训练方法。使用 ResNet-18 作为骨干网络，在无标签图像上完成前置任务预训练，再将 encoder 迁移到少量标注图像分类任务中微调，并与随机初始化模型对比。

## 2. 方法

### 2.1 Backbone

采用 ResNet-18。考虑 Tiny ImageNet 图像尺寸较小，将第一层卷积改为 `3x3, stride=1, padding=1`，并移除原始 `maxpool`，以保留更多空间信息。

### 2.2 前置任务

Rotation Prediction:
输入图像随机旋转 0、90、180 或 270 度，模型预测旋转类别。该任务鼓励模型学习物体方向、轮廓和语义结构。

Jigsaw Puzzle:
将图像切成 3x3 小块，并从固定排列集合中采样一种排列打乱图像，模型预测排列编号。该任务鼓励模型学习局部纹理与全局结构关系。

Relative Patch Position:
从同一图像中采样中心块和邻近块，模型预测邻近块相对中心块的 8 个方向类别。该任务鼓励模型学习上下文和空间关系。

## 3. 数据与训练设置

数据集：Tiny ImageNet。

预处理：随机裁剪、水平翻转、颜色扰动、归一化。

优化器：AdamW。

学习率调度：CosineAnnealingLR。

默认超参数：

| 阶段 | Epochs | Batch Size | LR | Weight Decay |
| --- | ---: | ---: | ---: | ---: |
| SSL 预训练 | 20 | 128 | 1e-3 | 1e-4 |
| 少样本微调 | 30 | 128 | 1e-3 | 1e-4 |

## 4. 超参数调优

记录不同 batch size、学习率、预训练轮数对前置任务准确率、下游 Top-1/Top-5、训练时间和显存占用的影响。

| Pretext | Batch Size | LR | Epochs | Pretext Acc | Fine-tune Top-1 | Fine-tune Top-5 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rotation | 64 | 1e-3 | 20 |  |  |  |  |
| Rotation | 128 | 1e-3 | 20 |  |  |  |  |
| Jigsaw | 128 | 1e-3 | 20 |  |  |  |  |
| Relative Patch | 128 | 1e-3 | 20 |  |  |  |  |

可根据 `outputs/experiment_summary.csv` 填写表格。

## 5. 下游分类结果

| 初始化方式 | Shots/Class | Top-1 Acc | Top-5 Acc |
| --- | ---: | ---: | ---: |
| Random Init | 10 |  |  |
| Rotation SSL | 10 |  |  |
| Jigsaw SSL | 10 |  |  |
| Relative Patch SSL | 10 |  |  |

## 6. 分析

可从以下角度分析：

- 自监督预训练是否提升少样本分类性能。
- 哪个前置任务迁移效果最好。
- batch size 增大后训练更稳定还是资源消耗更高。
- 学习率过大或过小时对收敛速度和最终指标的影响。
- 前置任务准确率和下游分类准确率是否完全一致，若不一致说明原因。

## 7. 结论

总结自监督学习在减少标注数据依赖方面的效果，并说明本实验的局限，例如训练轮数、模型规模、数据增强策略和计算资源限制。
