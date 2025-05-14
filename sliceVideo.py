import os

import cv2
import face_recognition

frameInterval = 1  # 每几帧取一帧
root = "videos"  # 输入目录
outRoot = "pictures"  # 输出目录，不要更改


def CropImg(image, x, y, width, height):
    # 使用 OpenCV 根据给定的左上角坐标 (x, y) 和裁剪的宽度 (width)、高度 (height) 来裁剪图像。
    return image[y : y + height, x : x + width]


def sliceV(sourcePath, fnf):
    persons = os.listdir(sourcePath)
    persons.sort()
    for person in persons:
        # 逐人切
        person = person.split("_")[0]
        imgOutRoot = os.path.join(outRoot, person)
        if not os.path.exists(imgOutRoot):
            os.makedirs(imgOutRoot)
        print(f"Start Processing No.{person}")
        if not os.path.exists(os.path.join(imgOutRoot, fnf)):
            os.makedirs(os.path.join(imgOutRoot, fnf))
        outfile = open(os.path.join(imgOutRoot, fnf, "rects.txt"), "w")
        # 使用cv2读取视频，手机录制竖屏视频可能有问题，自行转格式
        cap = cv2.VideoCapture(os.path.join(sourcePath, person + "_" + fnf + ".mp4"))
        if not cap.isOpened():
            print("Error opening video stream or file")
            exit()
        frameCount = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frameCount % frameInterval != 0:
                # 每几帧跳过
                frameCount += 1
                continue
            (h, w) = frame.shape[:2]
            print(f"Start Processing frame {frameCount}")
            # 使用 face_recognition 识别人脸
            faces = face_recognition.face_locations(frame)
            if len(faces) == 0:
                frameCount += 1
                print("No face found")
                continue
            top, right, bottom, left = max(
                faces, key=lambda f: (f[2] - f[0]) * (f[1] - f[3])
            )  # 选取面积最大的面部
            fx = left
            fy = top
            fw = right - left
            fh = bottom - top
            if fw * fh < 500:  # 过于小的面部说明检测有误，抛弃
                frameCount += 1
                print("No face found")
                continue
            # 使用face_recognition 识别眼部特征点
            left = face_recognition.face_landmarks(frame)[0]["left_eye"]
            right = face_recognition.face_landmarks(frame)[0]["right_eye"]
            # 左眼框选
            lcx = 0
            lcy = 0
            lew = left[3][0] - left[0][0]
            lew = int(1.5 * lew)  # 放大截取范围
            leh = int((lew / 60) * 36)  # 按比例计算眼部高度
            for x, y in left:
                # 左眼中央坐标，取六个特征向量的平均
                lcx += x
                lcy += y
            lcx = int(lcx / len(left))
            lcy = int(lcy / len(left))
            lex = int(lcx - lew / 2)
            ley = int(lcy - leh / 2)
            # 右眼框选
            rcx = 0
            rcy = 0
            rew = right[3][0] - right[0][0]
            rew = int(1.5 * rew)
            reh = int((rew / 60) * 36)
            for x, y in right:
                # 右眼中央坐标，取六个特征向量的平均
                rcx += x
                rcy += y
            rcx = int(rcx / len(right))
            rcy = int(rcy / len(right))
            rex = int(rcx - rew / 2)
            rey = int(rcy - reh / 2)
            # 裁切图像并保存
            faceImage = CropImg(frame, fx, fy, fw, fh)
            ImageOutRoot = os.path.join(imgOutRoot, fnf, str(frameCount))
            if not os.path.exists(ImageOutRoot):
                os.makedirs(ImageOutRoot)
            cv2.imwrite(os.path.join(ImageOutRoot, "face.jpg"), faceImage)
            leImage = CropImg(frame, lex, ley, lew, leh)
            reImage = CropImg(frame, rex, rey, rew, reh)
            cv2.imwrite(os.path.join(ImageOutRoot, "left.jpg"), leImage)
            cv2.imwrite(os.path.join(ImageOutRoot, "right.jpg"), reImage)
            # 保存rects，格式为 帧, 面部xywh, 左眼xywh, 右眼xywh
            outfile.write(
                " ".join(
                    [
                        str(frameCount),
                        str(fx),
                        str(fy),
                        str(fw),
                        str(fh),
                        str(lex),
                        str(ley),
                        str(lew),
                        str(leh),
                        str(rex),
                        str(rey),
                        str(rew),
                        str(reh),
                        str(w),
                        str(h),
                    ]
                )
                + "\n"
            )
            frameCount += 1


if __name__ == "__main__":
    fVideoPath = os.path.join(root, "f")
    nfVideoPath = os.path.join(root, "nf")
    sliceV(fVideoPath, "f")
    sliceV(nfVideoPath, "nf")
