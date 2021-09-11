#!/usr/bin/env python
# -- coding:UTF-8 --s
# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
from yolo_change import YOLO
from PIL import Image
import time
import os
import cv2

yolo = YOLO()


def detect(img):
    # try:
    image = Image.open(img)
    # except:
    #     print('Open Error! Try again!')
    #     continue
    time1 = time.time()
    r_image = yolo.detect_image(image)
    time2 = time.time()
    print(time2-time1)
    # r_image.show()


def main():
    dict_name = '/home/zyf/dobot_ws/src/my_pkg/scripts/yolo3_pytorch/img'
    for file_name in os.listdir(dict_name):
        print(file_name)
        # img = cv2.imread('img/' + file_name)
        img_path = 'img/' + file_name
        detect(img_path)


if __name__ == '__main__':
    main()
