import cv2
import os


def read_dict(dict_name):
    count = 0
    for file_name in os.listdir(dict_name):
        print(file_name)
        img = cv2.imread(dict_name + '/' + file_name)
        cv2.imshow(file_name, img)
        cv2.waitKey(0)
        count += 1
        if count > 10:
            break


read_dict('/home/zyf/dobot_ws/src/my_pkg/scripts/yolo3_pytorch/img')
