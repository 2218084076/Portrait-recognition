from PIL import Image, ImageGrab
import mss
import mss.tools
import time
import cv2 as cv
import pyautogui
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

model = tf.keras.models.load_model('models/my_model')
batch_size = 32
android_x0 = 160
android_y0 = 160
android_x1 = 320
android_y1 = 320
img_width = android_x1 - android_x0
img_height = android_y1 - android_y0

def get_check_result(img_now):
    filename = "./cache/record_temp.jpg"
    cv.imwrite(filename,img_now)
    # filename = "./1/1 (%s).jpg"%i
    filepath=filename
# filepath = "sg.jpg"
    img = cv2.imread(filepath)  # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色

# OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier(
        "face.xml"
    )
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    cropped = None
    if len(faceRects):  # 大于0则检测到人脸
        cropped = None
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            # cv2.rectangle(img, (x, y), (x + h, y + w), color, 1)
            cropped = img[y:y+w, x:x+h] # 裁剪坐标为[y0:y1, x0:x1]
        cropped = cv2.resize(cropped,(160,160))
        cv2.imwrite(filename, cropped)

        path_temp = "./cache/record_temp.jpg"
        sunflower_path = pathlib.Path(path_temp)

        img = keras.preprocessing.image.load_img(
            sunflower_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        class_names=["home","list","news","search"]
        names={
            "home":"xia",
            "list":"jiao",
            "news":"wang",
            "search":"feng"
        }
        k = "None"
        # print(score)
        if 100 * np.max(score)>=50:
            k = class_names[np.argmax(score)]
            result = names.get(k,"None")
            print(result,"当前页面为:",class_names[np.argmax(score)],100 * np.max(score))
        result = names.get(k,"None")
        return result
    else:
        return "None"
capture = cv2.VideoCapture(0)
i=0
while True:
    ret,frame = capture.read()
    frame = frame[0:480,80:560]   #裁剪图像
    # frame=cv.resize(frame,(160,160))
    # base_img = cv.imread(get_android_img(),0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if i%10==9:
        i=0
        get_check_result(gray)
    else:
        i+=1
    key = cv.waitKey(1)
    if key == ord("q"):
        break

cv.destroyAllWindows()
