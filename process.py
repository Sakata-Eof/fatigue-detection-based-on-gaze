import os

import cv2

import model
import numpy as np
import torch.nn as nn
import torch
picturePath = "pictures"
outputPath = "output"
#加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.model()
net = nn.DataParallel(net)
state_dict = torch.load("checkpoint/Iter_10_AFF-Net.pt")
net.load_state_dict(state_dict)
net = net.module
net.to(device)
net.eval()

# 生成rects特征
def normalize_box(box, frame_size):
    x, y, w, h = box
    return [int(w) / int(frame_size[0]), int(h)/ int(frame_size[1]),
            int(x) / int(frame_size[0]), int(y) / int(frame_size[1])]

def preprocess_image(img, target_size, flip=False):
    """图像预处理函数"""
    img = cv2.resize(img, target_size)
    if flip:
        img = cv2.flip(img, 1)  # 水平翻转
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float()

def GazeEstimationPerVideo(path):
    file = open(os.path.join(path,"rects.txt"), 'r')
    lines = file.readlines()
    for line in lines:
        frameCount = line.split(" ")[0]
        print(f"frameCount: {frameCount}, gaze: ", end="")
        rects = line.split(" ")[1:]
        rects = np.array(
            normalize_box(tuple(rects[0:4]), tuple(rects[-2:]))+
            normalize_box(tuple(rects[4:8]), tuple(rects[-2:]))+
            normalize_box(tuple(rects[8:12]), tuple(rects[-2:]))
        ).astype(np.float32)
        rects = torch.from_numpy(rects).unsqueeze(0).to(device)
        face = cv2.imread(os.path.join(path, frameCount, "face.jpg"))
        face = preprocess_image(face, target_size=(224,224)).to(device)
        left = cv2.imread(os.path.join(path, frameCount, "left.jpg"))
        left = preprocess_image(left, target_size=(112,112)).to(device)
        right = cv2.imread(os.path.join(path, frameCount, "right.jpg"))
        right = preprocess_image(right, target_size=(112,112), flip = True).to(device)#右眼翻转
        with torch.no_grad():
            gaze = net(left, right, face, rects)
            gaze = gaze[0].cpu().numpy()
        print(gaze)
        outFile = open(os.path.join(outputPath, "gaze.txt"), "a")
        outFile.write(" ".join([frameCount, str(gaze[0]), str(gaze[1])])+"\n")

if __name__ == '__main__':
    persons = os.listdir(picturePath)
    persons.sort()
    for person in persons:
        print(f"Processing person No.{person}, fatigue.")
        outputPath = os.path.join("output", person,"f")
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        GazeEstimationPerVideo(os.path.join(picturePath, person,"f"))
        print(f"Processing person No.{person}, not fatigue.")
        outputPath = os.path.join("output", person, "nf")
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        GazeEstimationPerVideo(os.path.join(picturePath, person,"nf"))
