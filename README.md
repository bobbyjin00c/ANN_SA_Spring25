# ConvNet实现
## 实验概述
本实验在CIFAR-10数据集上实现卷积网络(ConvNet)，目标验证集准确率≥60%。通过架构改进和超参数优化，最终测试准确率达到69.2%。
## 关键实现
### 网络架构改进
- 基础架构：`ThreeLayerConvNet` (conv→ReLU→pool→fc→ReLU→fc→Softmax)
- 优化架构：`DeeperConvNet`实现VGG-like结构：
```
conv→ReLU→conv→ReLU→pool (重复块)
```
- 新增功能：
- Batch Normalization
- Dropout (最终未采用)
- Early Stopping
- 动态学习率衰减

  ### 参数优化历程
| 模型版本 | 关键改进 | Val Acc | Test Acc |
|----------|---------|---------|----------|
| 基准模型 | 双卷积层(32+64核) | ~60% | 63.2% |
| 模型二 | 同时引入BN+Dropout | 13.8% | - |
| 模型三 | 扩大规模(64+128核) | 12.8% | - |
| 模型四 | 仅用BN | 59.7% | 10.7% |
| 最终模型 | Early Stopping+LR衰减 | 72.8% | 69.2% |

## 最佳配置
```python
{
"hidden_dim": 256,
"num_filters": [32, 64],
"filter_size": 3,
"weight_scale": 1e-2,
"reg": 5e-4,
"use_batchnorm": True,
"use_dropout": False,
"learning_rate": 5e-4,
"batch_size": 100,
"num_epochs": 30
}
```
## 未来改进方向
- GPU加速训练（原环境限制强制CPU）
- 深层网络结构（VGG/ResNet）
- 正则化策略改进
- 优化学习率调度策略
  
