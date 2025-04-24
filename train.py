import os
import pandas as pd
import numpy as np
import logging
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# 配置日志
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GazeCNN(nn.Module):
    def __init__(self, in_channels):
        super(GazeCNN, self).__init__()
        # 第一个卷积层，增加通道数和卷积核大小
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=5, stride=1, padding=2)  # 更大通道数和卷积核
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)  # 更大通道数
        self.bn2 = nn.BatchNorm1d(256)  # Batch Normalization
        self.conv3 = nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2)  # 第三个卷积层，通道数更大
        self.bn3 = nn.BatchNorm1d(512)  # Batch Normalization
        self.pool = nn.AdaptiveMaxPool1d(1)  # 池化层，降维
        self.dropout = nn.Dropout(0.5)  # Dropout 防止过拟合
        self.fc1 = nn.Linear(512, 256)  # 全连接层
        self.fc2 = nn.Linear(256, 128)  # 全连接层
        self.fc3 = nn.Linear(128, 1)  # 输出层

    def forward(self, x):
        # 前向传播，逐层处理
        x = torch.relu(self.bn1(self.conv1(x)))  # 卷积 + 激活 + BN
        x = torch.relu(self.bn2(self.conv2(x)))  # 卷积 + 激活 + BN
        x = torch.relu(self.bn3(self.conv3(x)))  # 卷积 + 激活 + BN
        x = self.pool(x).squeeze(-1)  # 最大池化，提取最强特征
        x = self.dropout(x)  # Dropout
        x = torch.relu(self.fc1(x))  # 全连接 + 激活
        x = torch.relu(self.fc2(x))  # 全连接 + 激活
        return torch.sigmoid(self.fc3(x)).squeeze(1)  # 输出层，用 sigmoid 函数二分类


# 改进后的 reshape_for_cnn，支持按视频分段划窗
# video_boundaries: 每个样本属于哪个视频的分段标识，与 X、y 长度一致

def reshape_for_cnn(X, y, window_size=60, video_boundaries=None):
    X_seq, y_seq = [], []
    if video_boundaries is None:
        video_boundaries = [0] * len(X)  # 所有样本属于同一视频

    current_video = video_boundaries[0]
    start = 0
    for i in range(1, len(X)):
        if video_boundaries[i] != current_video:
            segment_X = X[start:i]
            segment_y = y[start:i]
            for j in range(len(segment_X) - window_size):
                X_seq.append(segment_X[j:j + window_size])
                y_seq.append(segment_y[j + window_size])
            current_video = video_boundaries[i]
            start = i
    # 最后一个视频段
    segment_X = X[start:]
    segment_y = y[start:]
    for j in range(len(segment_X) - window_size):
        X_seq.append(segment_X[j:j + window_size])
        y_seq.append(segment_y[j + window_size])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    X_seq = np.transpose(X_seq, (0, 2, 1))  # shape: [B, C, T]
    return X_seq, y_seq


from torch.optim.lr_scheduler import StepLR

def train_cnn(X, y, window_size=60, video_boundaries=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    X_seq, y_seq = reshape_for_cnn(X, y, window_size, video_boundaries)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.5, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = GazeCNN(in_channels=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning Rate Scheduler (StepLR)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # 每10个epoch降低学习率

    for epoch in range(100):  # 增加 epoch 数量
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            predicted = (preds > 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        train_loss /= total
        train_acc = correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)

                val_loss += loss.item() * xb.size(0)
                predicted = (preds > 0.5).float()
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        val_loss /= total
        val_acc = correct / total

        # 记录日志
        logging.info(f"Epoch [{epoch+1}/100], "
                     f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Epoch [{epoch+1}/100], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # 调整学习率
        scheduler.step()

    # 最终测试
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds) > 0.5
    acc = accuracy_score(all_labels, all_preds)
    logging.info(f"Final CNN Test Accuracy: {acc:.4f}")

    return acc


def PreProcess(X, y, measure=0, standardize=True):
    """
    X: 数据集特征
    y: 标签
    measure: 预处理方式（0-不处理，1-相对位移，2-相对位移+上一点信息，3-速度、加速度+注视点坐标）
    standardize: 是否标准化
    """
    df = pd.DataFrame(X, columns=["frame", "gaze_x", "gaze_y"])
    df["label"] = y

    # 按视频分割数据
    videos = []
    current_video = []
    last_frame = -1
    for i, row in df.iterrows():
        if row["frame"] < last_frame:
            videos.append(current_video)
            current_video = []
        current_video.append(row)
        last_frame = row["frame"]
    videos.append(current_video)

    # 划分视频边界，这里是给cnn切窗口用的
    video_boundaries = []
    video_id = 0
    last_frame = -1
    for i, row in df.iterrows():
        if row['frame'] < last_frame:
            video_id += 1
        video_boundaries.append(video_id)
        last_frame = row['frame']

    processed_videos = []

    for video in videos:
        video_df = pd.DataFrame(video)

        if measure == 0:
            features = video_df[["frame", "gaze_x", "gaze_y"]].values
            labels = video_df["label"].values
            if standardize:
                scaler = MinMaxScaler()
                features[:, 1:] = scaler.fit_transform(features[:, 1:])# 除帧数以外的数据进行归一化
            processed_videos.append((features, labels))

        elif measure == 1:
            video_df["frame_diff"] = video_df["frame"].diff().fillna(0)
            video_df["gaze_x_diff"] = video_df["gaze_x"].diff().fillna(0)
            video_df["gaze_y_diff"] = video_df["gaze_y"].diff().fillna(0)
            video_df = video_df.dropna(subset=["frame_diff", "gaze_x_diff", "gaze_y_diff"])
            features = video_df[["frame_diff", "gaze_x_diff", "gaze_y_diff"]].values
            labels = video_df["label"].values
            if standardize:
                scaler = MinMaxScaler()
                features = scaler.fit_transform(features)
            processed_videos.append((features, labels))

        elif measure == 2:
            video_df["frame_diff"] = video_df["frame"].diff().fillna(0)
            video_df["gaze_x_diff"] = video_df["gaze_x"].diff().fillna(0)
            video_df["gaze_y_diff"] = video_df["gaze_y"].diff().fillna(0)
            video_df["frame_diff_prev"] = video_df["frame_diff"].shift(1).fillna(0)
            video_df["gaze_x_diff_prev"] = video_df["gaze_x_diff"].shift(1).fillna(0)
            video_df["gaze_y_diff_prev"] = video_df["gaze_y_diff"].shift(1).fillna(0)
            features = video_df[["frame_diff", "gaze_x_diff", "gaze_y_diff", "frame_diff_prev", "gaze_x_diff_prev", "gaze_y_diff_prev"]].values
            labels = video_df["label"].values
            if standardize:
                scaler = MinMaxScaler()
                features = scaler.fit_transform(features)
            processed_videos.append((features, labels))

        elif measure == 3:
            video_df["velocity_x"] = video_df["gaze_x"].diff().fillna(0)
            video_df["velocity_y"] = video_df["gaze_y"].diff().fillna(0)
            video_df["acceleration_x"] = video_df["velocity_x"].diff().fillna(0)
            video_df["acceleration_y"] = video_df["velocity_y"].diff().fillna(0)
            features = video_df[["frame", "gaze_x", "gaze_y", "velocity_x", "velocity_y", "acceleration_x", "acceleration_y"]].values
            labels = video_df["label"].values
            if standardize:
                scaler = MinMaxScaler()
                features[:, 1:] = scaler.fit_transform(features[:, 1:])
            processed_videos.append((features, labels))

    all_features = np.vstack([features for features, _ in processed_videos])
    all_labels = np.hstack([labels for _, labels in processed_videos])

    return all_features, all_labels, video_boundaries


def LoadData(fnf, dataPath):# 加载数据
    persons = os.listdir(dataPath)
    persons.sort()
    data = []
    for person in persons:
        logging.info(f"Loading data No.{person}, {fnf}")
        inFile = open(os.path.join(dataPath, person, fnf, "gaze.txt"), 'r')
        lines = inFile.readlines()
        for line in lines:
            data.append([int(line.split(" ")[0]), float(line.split(" ")[1]), float(line.split(" ")[2])])
    return data

if __name__ == '__main__':
    dataPath = 'output'

    dataF = LoadData("f", dataPath)
    label = [1] * len(dataF) #添加标签，疲劳为1

    dataNF = LoadData("nf", dataPath)
    label += [0] * len(dataNF) #不疲劳为0
    all_accuracies = []  # 用于存储所有模型在不同方法下的准确率
    for i in range(4): # 4种数据预处理方式
        X, y, video_boundaries = PreProcess(dataF + dataNF, label, measure=i, standardize=True) # 预处理， measure取0-3，
                                                                              # 0-不处理，1-相对位移，2-相对位移+上一点信息，3-速度、加速度+注视点坐标

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)# 测试集占0.3

        models = {
            'SVC': SVC(),
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(),
            'KNeighbors': KNeighborsClassifier()
        }

        param_grids = {
            'SVC': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf']
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2']
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }

        best_models = {}
        model_accuracies = []
        for model_name, model in models.items(): # 对比不同模型
            logging.info(f"\nPerforming grid search for {model_name}...")
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', verbose=2)  # 超参数调优，verbose=2 显示进度

            grid_search.fit(X_train, y_train)

            logging.info(f"Best parameters for {model_name} in measure {i}: {grid_search.best_params_}")
            logging.info(f"Best cross-validation score for {model_name} in measure {i}: {grid_search.best_score_}")

            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model

            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Accuracy for {model_name} in measure {i}: {accuracy_score(y_test, y_pred)}")
            logging.info(f"Classification Report for {model_name} in measure {i}:\n{classification_report(y_test, y_pred)}")
            model_accuracies.append((f"{model_name}-{i}", accuracy))  # 保存模型和方法编号
        logging.info("\n--- Model Comparison ---")
        for model_name, model in best_models.items():
            y_pred = model.predict(X_test)
            logging.info(f"{model_name} Accuracy in measure {i}: {accuracy_score(y_test, y_pred)}")

        logging.info("Training CNN model...")
        print("Training CNN model...")
        # 注意：这里需要提供 video_boundaries 列表，与原始数据对齐
        # 使用预处理返回的 video_boundaries
        acc = train_cnn(X, y, window_size=10, video_boundaries=video_boundaries)
        all_accuracies.append((f"CNN-{i}", acc))
        logging.info(f"Accuracy for CNN in measure {i}: {acc}")

        all_accuracies.extend(model_accuracies)  # 将当前测量结果加入所有结果
        logging.info(f"measure {i} Training complete.")

    all_accuracies.sort(key=lambda x: x[1], reverse=True)  # 按准确率排序
    names = [x[0] for x in all_accuracies]
    accuracies = [x[1] for x in all_accuracies]

    # 使用 Seaborn 绘制条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x=names, y=accuracies, palette="Blues_d")
    plt.title('Model Comparison - Accuracy for Different Methods', fontsize=16)
    plt.xlabel('Model and Method', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    logging.info("Training complete.")
