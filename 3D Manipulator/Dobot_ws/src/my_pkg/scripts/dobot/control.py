#!/usr/bin/env python
#-- coding:UTF-8 --
# 机械臂+吸盘控制， 给定目标位置，机械臂移动，吸取，移动，放下。

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from pyfirmata import Arduino
import time


class dobot_contorl():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dobot_contorl')

        self.board = Arduino('/dev/ttyUSB1')

        # self.listener = tf.TransformListener()

        self.arm = moveit_commander.MoveGroupCommander('magician_arm')
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.05)
        self.arm.allow_replanning(True)
        self.arm.set_planning_time(10)

        self.main()

    def go_home(self): 
        self.arm.set_named_target('home')
        self.arm.go()
        rospy.sleep(rospy.Duration(1.0))

    def go_target_position(self, x, y, z):
        current_pose = self.arm.get_current_pose()
        # print(current_pose)
        #   position: 
        #     x: 0.165931380919
        #     y: 9.05112419306e-06
        #     z: 0.0414030955311
        target_pose = Pose()
        target_pose.orientation = current_pose.pose.orientation
        target_pose.position = current_pose.pose.position
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        self.arm.set_pose_target(target_pose)
        self.arm.go()
        rospy.sleep(rospy.Duration(1.0))

    def suck(self):
        self.board.servo_config(9, 0, 255, 180)
        self.board.servo_config(8, 0, 255, 0)
        time.sleep(2)
        self.board.servo_config(9, 0, 255, 0)
        self.board.servo_config(8, 0, 255, 0)
        time.sleep(1)

    def loose(self):
        self.board.servo_config(9, 0, 255, 0)
        self.board.servo_config(8, 0, 255, 180)
        time.sleep(1)
        self.board.servo_config(9, 0, 255, 0)
        self.board.servo_config(8, 0, 255, 0)
        time.sleep(1)

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    def main(self):
        print('Go home')
        self.go_home()
        
        print('Go to target position')
        self.go_target_position(80e-3, -184e-3, -50e-3)
        self.go_target_position(80e-3, -184e-3, -70e-3)

        print('suck')
        self.suck()

        print('Go to target position')
        self.go_target_position(80e-3, 184e-3, -50e-3)

        print('loose')
        self.loose()

        print('Go home')
        self.go_home()

        print('shut down')
        self.shut_down()


if __name__ == '__main__':
    dobot_contorl()

