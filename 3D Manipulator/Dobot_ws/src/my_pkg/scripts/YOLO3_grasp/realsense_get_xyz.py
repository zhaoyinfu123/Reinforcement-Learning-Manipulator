import pyrealsense2 as rs
import numpy as np
import cv2

# 声明点云对象，用于计算点云和纹理映射。
pc = rs.pointcloud()
# 我们希望点云对象是持续的，这样当有新的帧到达时，能够显示出来
points = rs.points()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
pipe_profile = pipeline.start(config)

# 创建一个对其目标
# rs.aling能够让我们将深度帧和其他帧对齐
# align_to表示对齐的方式
align_to = rs.stream.color
align = rs.align(align_to)


while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  # 对齐

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    img_color = np.asanyarray(color_frame.get_data())  # 转换为cv2数据类型
    img_depth = np.asanyarray(depth_frame.get_data())

    # # Intrinsics & Extrinsics
    # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    # color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    # depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

    # # depth_scale 深度比例 转换为米
    # depth_sensor = pipe_profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    # Map depth to color
    # depth_pixel = [240, 320]   # Random pixel
    # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)

    # color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
    # color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
    # print('depth: 1', color_point)
    # print('depth: 2', color_pixel)

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    # tex = np.asanyarray(points.get_texture_coordinates())
    i = 640*200+200  # (200, 200)处的像素号
    print('depth:', 'x', [np.float(vtx[i][0]), 'y', np.float(vtx[i][1]), 'z', np.float(vtx[i][2])])
    # 在（200， 200）画圆，并过得深度信息
    cv2.circle(img_color, (200, 200), 8, [255, 0, 255], thickness=-1)
    cv2.putText(img_color, "Dis:"+str(img_depth[200, 200])+'mm', (40, 40),  
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
    cv2.putText(img_color, "X:"+str(np.float(vtx[i][0]))+'m',    (80, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
    cv2.putText(img_color, "Y:"+str(np.float(vtx[i][1]))+'m',    (80, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
    cv2.putText(img_color, "Z:"+str(np.float(vtx[i][2]))+'m',    (80, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
    cv2.imshow('depth_frame', img_color)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    '''
    npy_vtx = np.zeros((len(vtx), 3), float)
    print('len: ',len(vtx))
    for i in range(len(vtx)):
        npy_vtx[i][0] = np.float(vtx[i][0])
        npy_vtx[i][1] = np.float(vtx[i][1])
        npy_vtx[i][2] = np.float(vtx[i][2])
    npy_tex = np.zeros((len(tex), 3), float)
    for i in range(len(tex)):
        npy_tex[i][0] = np.float(tex[i][0])
        npy_tex[i][1] = np.float(tex[i][1])
    '''

pipeline.stop()

'''
pc = rs.pointcloud()
frames = pipeline.wait_for_frames()
depth = frames.get_depth_frame()
color = frames.get_color_frame()
img_color = np.asanyarray(color_frame.get_data())
img_depth = np.asanyarray(depth_frame.get_data())
pc.map_to(color)
points = pc.calculate(depth)
vtx = np.asanyarray(points.get_vertices())
tex = np.asanyarray(points.get_texture_coordinates())
npy_vtx = np.zeros((len(vtx), 3), float)
for i in range(len(vtx)):
    npy_vtx[i][0] = np.float(vtx[i][0])
    npy_vtx[i][1] = np.float(vtx[i][1])
    npy_vtx[i][2] = np.float(vtx[i][2])
npy_tex = np.zeros((len(tex), 3), float)
for i in range(len(tex)):
    npy_tex[i][0] = np.float(tex[i][0])
    npy_tex[i][1] = np.float(tex[i][1])
'''
