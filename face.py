import cv2
import os 

for i in range(1,51):
    filename = "./1/1 (%s).jpg"%i
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

    if len(faceRects):  # 大于0则检测到人脸
        cropped = None
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            # cv2.rectangle(img, (x, y), (x + h, y + w), color, 1)
            cropped = img[y:y+w, x:x+h] # 裁剪坐标为[y0:y1, x0:x1]
            '''
            # 左眼
            cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                       color)
            #右眼
            cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                       color)
            #嘴巴
            cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                          (x + 5 * w // 8, y + 7 * h // 8), color)
            '''
    # cv2.imshow("image", cropped)  # 显示图像
    # k = cv2.waitKey(0)
    # if k == ord("s"):
    
        cv2.imwrite("./1/2 (%s).jpg"%i, cropped)

    # cv2.destroyAllWindows()