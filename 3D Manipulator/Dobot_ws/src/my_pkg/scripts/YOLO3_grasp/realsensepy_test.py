import pyrealsense2 as rs
pipeline = rs.pipeline()
pipe_profile = pipeline.start()
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Map depth to color
    depth_pixel = [240, 320]   # Random pixel
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
    color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
    color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
    pipeline.stop()


    pc = rs.pointcloud()
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    img_color = np.asanyarray(color.get_data())
    img_depth = np.asanyarray(depth.get_data())
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