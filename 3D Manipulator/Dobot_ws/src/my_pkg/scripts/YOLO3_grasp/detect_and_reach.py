#!/usr/bin/env python
# -- coding:UTF-8 --
import sys
import cv2
import moveit_commander
import numpy as np
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Pose
from PIL import Image
from pyfirmata import Arduino
import time
from yolo import YOLO


class control():
    def __init__(self):
        self.yolo = YOLO()
        # 声明点云对象，用于计算点云和纹理映射。
        self.pc = rs.pointcloud()
        # 我们希望点云对象是持续的，这样当有新的帧到达时，能够显示出来
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dobot_contorl')
        self.arm = moveit_commander.MoveGroupCommander('magician_arm')
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_max_velocity_scaling_factor(0.7)
        self.arm.set_max_acceleration_scaling_factor(0.7)
        self.arm.allow_replanning(True)
        self.arm.set_planning_time(10)
        self.go_home()
        # self.go_init_location()
        self.counter = 0

    def go_home(self):
        self.arm.set_named_target('home')
        self.arm.go()

    def go_init_location(self):
        current_pose = self.arm.get_current_pose()
        target_pose = Pose()
        target_pose.orientation = current_pose.pose.orientation
        target_pose.position = current_pose.pose.position
        target_pose.position.x = 0.15
        target_pose.position.y = 0.15
        target_pose.position.z = 0
        self.arm.set_pose_target(target_pose)
        self.arm.go()

    def detect(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)  # 对齐

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            img_color = np.asanyarray(color_frame.get_data())  # 转换为cv2数据类型
            self.pc.map_to(color_frame)
            points = self.pc.calculate(depth_frame)
            self.vtx = np.asanyarray(points.get_vertices())

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            frame = self.yolo.detect_image(frame)
            obj_info = self.yolo.obj_info
            self.counter += 1
            if obj_info != [] and self.counter > 10:
                break
        return obj_info

    def transform(self, obj_info):
        obj_list = []
        for obj in obj_info:
            i = 640*obj[1]+obj[0]
            obj_x = round(np.float(self.vtx[i][0]), 3)
            obj_y = round(np.float(self.vtx[i][1]), 3)
            obj_z = round(np.float(self.vtx[i][2]), 3)
        # obj_x = 0.12
        # obj_y = -0.072
        # obj_z = 0.479
            print(obj_x, obj_y, obj_z)
            # if obj_x < 0:
            #     obj_x_robot_frame = 0.18 + obj_y * 1.3  # 195
            #     obj_y_robot_frame = obj_x*1.1
            #     obj_z_robot_frame = 0.41 - obj_z
            # if obj_x >= 0:
            obj_x_robot_frame = 0.17 + obj_y * 1.2  # 195
            obj_y_robot_frame = obj_x
            obj_z_robot_frame = 0.43 - obj_z
            kind = obj[2]
            high = obj[3]
            wide = obj[4]
            obj_list.append([kind, obj_x_robot_frame, obj_y_robot_frame, obj_z_robot_frame, high, wide])
        return obj_list

    def reach(self, x, y, z):
        current_pose = self.arm.get_current_pose()
        target_pose = Pose()
        target_pose.orientation = current_pose.pose.orientation
        target_pose.position = current_pose.pose.position
        # x_offset = 0.06*np.cos(jv[0])
        # y_offset = 0.06*np.sin(jv[0])
        # print(x_offset, y_offset)
        theta = np.arctan(y/x)
        y_offset = 0.02*np.sin(theta)
        x_offset = 0.02*np.cos(theta)
        z_offset = 0.05  # real 0.05
        target_pose.position.x = x - x_offset
        target_pose.position.y = y - y_offset
        target_pose.position.z = z + z_offset
        self.arm.set_pose_target(target_pose)
        self.arm.go()
        rospy.sleep(rospy.Duration(1.0))


class sucker_control():
    def __init__(self):
        self.board = Arduino('/dev/ttyUSB1')

    def suck(self):
        self.board.servo_config(9, 0, 180, 180)
        self.board.servo_config(8, 0, 180, 0)
        time.sleep(3)
        self.board.servo_config(9, 0, 180, 0)
        self.board.servo_config(8, 0, 180, 0)
        time.sleep(1)

    def loose(self):
        self.board.servo_config(9, 0, 180, 0)
        self.board.servo_config(8, 0, 180, 180)
        time.sleep(1)
        self.board.servo_config(9, 0, 180, 0)
        self.board.servo_config(8, 0, 180, 0)
        time.sleep(1)


def main():
    c = control()
    # sucker = sucker_control()
    print('detect')
    obj_info = c.detect()
    print('transform')
    obj_info = c.transform(obj_info)
    for obj in obj_info:
        obj_type = obj[0]
        x = obj[1]
        y = obj[2]
        z = obj[3]
        high = obj[4]
        wide = obj[5]
        s = high * wide
        print('S=', s)
        # obj_info = [['apple', 0.012, -0.097, 0.38]]
        print('reach')
        c.reach(x, y, z+0.02)
        c.reach(x, y, z-0.015)
        # sucker.suck()
        # if s < 8000:
        #     c.reach(0.08, 0.12, -0.05)
        # else:
        #     c.reach(0.08, -0.12, -0.05)
        # c.reach(0.12, -0.12, -0.05)
        # sucker.loose()

        # c.reach(x, y, z)
        # sucker.suck()
        # if obj_type == 'red plum':
        #     c.reach(0.08, 0.12, -0.05)
        # elif obj_type == 'black plum':
        #     c.reach(0.08, -0.12, -0.05)
        # sucker.loose()
        c.go_home()


if __name__ == '__main__':
    main()
