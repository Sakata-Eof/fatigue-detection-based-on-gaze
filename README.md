# 基于注视点估计的疲劳度评估
## 项目说明
该项目是本人本科毕业设计，收集了8个人疲劳与非疲劳状态下
的面部图像视频作为数据集，使用 [AFF-Net](https://github.com/vigil1917/AFF-Net)
得到注视点坐标并使用四种预处理方式处理坐标，
最后使用不同模型进行二分类，形成 benchmark 。

参考文献：[ Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets](https://ieeexplore.ieee.org/abstract/document/9412205)

## QUICK START
### 数据预处理
运行 `sliceVideo.py`, 等待切分完成后再执行 `process.py` 进行注视点估计。

`3Dplot.py` 可用于生成处理后的注视点坐标3维图表，`plot.py` 生成的则是二维图表。

### 模型训练与评估
运行 `train.py`, 训练结束后会自动生成benchmark图表。
训练的具体过程以及最佳超参数可以在 `model_training.log`
中查找。

