# 从文件读取得分并排序
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# 从文件读取得分并排序
df_scores = pd.read_csv("scores.csv")
df_scores = df_scores.sort_values(by="Accuracy", ascending=False)

names = df_scores["Model-Measure"].tolist()
scores = df_scores["Accuracy"].tolist()
dic = {}
for i in range(len(names)):
    dic[names[i]] = scores[i]

# 使用 Matplotlib 绘制条形图
zhfont1 = matplotlib.font_manager.FontProperties(fname="MapleMonoNormalNL-NF-CN-Regular.ttf")
plt.figure(figsize=(12, 8))
sns.barplot(x=names, y=scores, palette="Blues_d")
plt.title('模型比对 - 不同方法下准确率', fontsize=16, fontproperties=zhfont1)
plt.xlabel('模型-方法', fontsize=12, fontproperties=zhfont1)
plt.ylabel('准确率', fontsize=12, fontproperties=zhfont1)
plt.xticks(rotation=45, ha="right")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontproperties=zhfont1)
plt.tight_layout()
plt.savefig('benchmark\\benchmark.svg')
plt.show()
for i in range(4):
    measure_i_dict={}
    for mm in dic.keys():
        if mm.split("-")[1] == str(i+1):
            measure_i_dict[mm.split('-')[0]] = dic[mm]
    pd_mi = pd.DataFrame(measure_i_dict.items(), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
    names = pd_mi["Model"].tolist()
    scores = pd_mi["Accuracy"].tolist()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=names, y=scores, palette="Reds")
    plt.title(f'模型比对 - 方法{i+1}下准确率', fontsize = 18, fontproperties=zhfont1)
    plt.xlabel('模型', fontsize=18, fontproperties=zhfont1)
    plt.ylabel('准确率', fontsize=18, fontproperties=zhfont1)
    plt.xticks(rotation=45, ha="right")
    for j, v in enumerate(scores):
        plt.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontproperties=zhfont1)
    plt.tight_layout()
    plt.savefig(f'benchmark\\benchmark-measure{i+1}.svg')
    plt.show()
