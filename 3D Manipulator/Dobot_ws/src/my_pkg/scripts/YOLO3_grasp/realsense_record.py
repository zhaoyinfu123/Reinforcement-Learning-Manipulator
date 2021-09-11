#!/usr/bin/env python
# -- coding:UTF-8 --
# -------------------------------------#
#       调用摄像头检测
# -------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import pyrealsense2 as rs


yolo = YOLO()

# 声明点云对象，用于计算点云和纹理映射。
pc = rs.pointcloud()
# 我们希望点云对象是持续的，这样当有新的帧到达时，能够显示出来
points = rs.points()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

record = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'XVID'), 5, (640, 480))

fps = 0.0
while(True):
    ########
    # realsense获取深度信息
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  # 对齐

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    img_color = np.asanyarray(color_frame.get_data())  # 转换为cv2数据类型
    img_depth = np.asanyarray(depth_frame.get_data())
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    ########

    t1 = time.time()

    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()
    # 读取某一帧
    # color_image = np.asanyarray(color_frame.get_data())

    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = yolo.detect_image(frame)
    frame = np.array(frame)

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #######
    obj_info = yolo.obj_info
    yolo.obj_info = []
    # obj_info[0]和obj_info[1]为像素位置,而obj_info[2]为目标类别
    obj_xyz = []
    if obj_info != []:
        # print('obj_info', obj_info)
        for obj in obj_info:
            i = 640*obj[1]+obj[0]
            obj_x = round(np.float(vtx[i][0]), 3)
            obj_y = round(np.float(vtx[i][1]), 3)
            obj_z = round(np.float(vtx[i][2]), 3)
            obj_xyz.append([obj[2], obj_x, obj_y, obj_z])
            frame = cv2.putText(frame, '{},{},{}'.format(obj_x, obj_y, obj_z),
                                (obj[0], obj[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if obj_xyz != []:
        print('obj_xyz', obj_xyz)
    #######

    fps = (fps + (1./(time.time()-t1))) / 2
    print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video", frame)

    record.write(frame)

    c = cv2.waitKey(30) & 0xff
    if c == 27:
        cv2.destroyAllWindows()
        record.release()
        # yolo.close_session()
        pipeline.stop()
        break
