# 基于注视点估计的疲劳度评估
## 数据预处理
运行 `sliceVideo.py`, 等待切分完成后再执行 `process.py` 进行注视点估计。

`3Dplot.py` 可用于生成处理后的注视点坐标3维图表，`plot.py` 生成的则是二维图表。

## 模型训练与评估
运行 `train.py`, 训练结束后会自动生成benchmark图表。