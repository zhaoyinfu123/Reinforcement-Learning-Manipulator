# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
from yolo import YOLO
from PIL import Image
import time

yolo = YOLO()

while True:
    img = input('Input image filename:')
    # try:
    image = Image.open(img)
    # except:
    #     print('Open Error! Try again!')
    #     continue
    time1 = time.time()
    r_image = yolo.detect_image(image)
    time2 = time.time()
    print(time2-time1)
    r_image.show()
