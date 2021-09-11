# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
from yolo_change import YOLO
from PIL import Image
import time
import cv2

yolo_strewberry = YOLO(c='strawberry')
yolo_strewberry_stem = YOLO(c='strawberry_stem')

while True:
    img = input('Input image filename:')
    # try:
    image = Image.open(img)
    # except:
    #     print('Open Error! Try again!')
    #     continue
    time1 = time.time()
    strewberry_image = yolo_strewberry.detect_image(image)
    for strewberry in yolo_strewberry.strewberry_list:
        strewberry = Image.fromarray(cv2.cvtColor(strewberry, cv2.COLOR_BGR2RGB))
        strewberry_stem_image = yolo_strewberry_stem.detect_image(strewberry)
        strewberry_stem_image.show()
    time2 = time.time()
    print(time2-time1)
