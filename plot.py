# 从文件读取得分并排序
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df_scores = pd.read_csv("scores.csv")
df_scores = df_scores.sort_values(by="F1-score", ascending=False)

names = df_scores["Model-Measure"].tolist()
scores = df_scores["F1-score"].tolist()

# 使用 Matplotlib 绘制条形图
zhfont1 = matplotlib.font_manager.FontProperties(fname="MapleMonoNormalNL-NF-CN-Regular.ttf")
plt.figure(figsize=(12, 8))
sns.barplot(x=names, y=scores, palette="Blues_d")
plt.title('模型比对 - 不同方法下 F1-score', fontsize=16, fontproperties=zhfont1)
plt.xlabel('模型-方法', fontsize=12, fontproperties=zhfont1)
plt.ylabel('F1-score', fontsize=12, fontproperties=zhfont1)
plt.xticks(rotation=45, ha="right")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontproperties=zhfont1)
plt.tight_layout()
plt.savefig('benchmark.svg')
plt.show()