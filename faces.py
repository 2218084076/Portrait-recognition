import os
import cv2
image_list = []
for i in os.listdir("./temp/"):
    for image in os.listdir("./temp/%s/"%i):
        image_list.append("./temp/%s/%s"%(i,image))
print(image_list)
for image_url in image_list:
    img = cv2.imread(image_url)
    img = cv2.resize(img,(160,160))
    cv2.imwrite(image_url,img)