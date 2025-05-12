import os

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于三维绘图

# 设置数据路径
data_path = 'output'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 存储所有数据的列表
all_data = []

# 遍历目录并加载数据
for video_id in range(10):  # 假设视频序号为 0-9
    for label in ['f', 'nf']:  # 'f' 和 'nf' 文件夹
        folder_path = os.path.join(data_path, str(video_id), label)
        gaze_file_path = os.path.join(folder_path, 'gaze.txt')

        if os.path.exists(gaze_file_path):
            # 读取数据
            data = pd.read_csv(gaze_file_path, delim_whitespace=True, header=None, names=['frame', 'gaze_x', 'gaze_y'])
            if label =='f':
                data['label'] = 'fatigue'
            else:
                data['label'] = 'not fatigue'

            # 将数据添加到总数据列表
            all_data.append(data)

# 合并所有数据
all_data = pd.concat(all_data, ignore_index=True)
zhfont1 = matplotlib.font_manager.FontProperties(fname="MapleMonoNormalNL-NF-CN-Regular.ttf")
# 绘制三维散点图
for label in ['f', 'nf']:
    # 选择数据
    if label == 'f':
        label_data = all_data[all_data['label'] == 'fatigue']
        label = '疲劳'
        x_offset = 0  # 疲劳点的 x 坐标偏移量
        y_offset = 0  # 疲劳点的 y 坐标偏移量
    else:
        label_data = all_data[all_data['label'] == 'not fatigue']
        label = '不疲劳'
        x_offset = -1  # 疲劳点的 x 坐标偏移量
        y_offset = -1  # 疲劳点的 y 坐标偏移量

    # 设置颜色：'f' 为红色，'nf' 为蓝色
    color = 'r' if label == '疲劳' else 'b'

    # 绘制散点图，添加 z 坐标偏移
    ax.scatter(label_data['gaze_x']+x_offset, label_data['gaze_y']+y_offset, label_data['frame'],
               c=color, label=label, alpha=0.6, s=10)

# 设置轴标签
ax.set_xlabel('X', fontproperties=zhfont1)
ax.set_ylabel('Y', fontproperties=zhfont1)
ax.set_zlabel('帧', fontproperties=zhfont1)

# 设置投影方向
ax.view_init(azim=150, elev = 30)  # elev 是仰角，azim 是方位角

# 设置标题
ax.set_title('注视点坐标的3D散点图', fontproperties=zhfont1)

# 添加图例
ax.legend()
plt.savefig('scatter.svg')
# 显示图形
plt.show()
