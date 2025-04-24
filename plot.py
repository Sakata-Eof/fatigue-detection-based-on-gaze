import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv('output/1/f/gaze.txt', delim_whitespace=True, header=None, names=['frame', 'gaze_x', 'gaze_y'])

# 设置 Seaborn 风格
sns.set(style="darkgrid")

# 创建一个画布和子图
plt.figure(figsize=(10, 6))

# 绘制 gaze_x 和 gaze_y 随着帧数变化的线型图
plt.plot(data['frame'], data['gaze_x'], label='Gaze X', color='blue')
plt.plot(data['frame'], data['gaze_y'], label='Gaze Y', color='red')

# 设置图表的标题和标签
plt.title('Gaze Point Over Frames', fontsize=16)
plt.xlabel('Frame', fontsize=12)
plt.ylabel('Gaze Coordinates', fontsize=12)

# 添加图例
plt.legend()

# 显示图表
plt.show()
