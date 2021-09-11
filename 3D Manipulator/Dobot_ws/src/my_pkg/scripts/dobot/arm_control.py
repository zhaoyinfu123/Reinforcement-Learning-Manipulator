#!/usr/bin/env python
#-- coding:UTF-8 --
# 单纯机械臂控制，给定目标位置。


import sys
import rospy
import moveit_commander
import tf2_ros
import tf
from geometry_msgs.msg import Pose, PoseStamped
import time
import random
import numpy as np
import math

class arm_control():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)  
        rospy.init_node('dobot_contorl')  
        self.arm = moveit_commander.MoveGroupCommander('magician_arm')
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.05)
        self.arm.allow_replanning(True)
        self.arm.set_planning_time(10)

    def get_eof_location(self):
        pose = self.arm.get_current_pose()
        eof_x = pose.pose.position.x
        eof_y = pose.pose.position.y
        eof_z = pose.pose.position.z
        return eof_x, eof_y, eof_z


    def init_object_location(self):
        x = random.uniform(0.14, 0.23)
        y = random.uniform(-0.22, 0.22)
        z = random.uniform(-0.06, 0.032)
        return x, y, z

    def verify_executable(self):
        x, y, z = self.init_object_location()
        traj = self.get_traj(x, y, z)
        while traj.joint_trajectory.joint_names == []:
            x, y, z = self.init_object_location()
            traj = self.get_traj(x, y, z)
        return traj

    def go_home(self):
        self.arm.set_named_target('home')
        self.arm.go()

    def set_joint_value(self, joint_values):
        try:
            self.arm.set_joint_value_target(joint_values)
            self.arm.go(wait=True)
            m = 1
        except:
            m = 0
            pass
        return m

    def get_joint_value(self):
        joint_values = self.arm.get_current_joint_values()
        return joint_values

    def get_traj(self, x, y, z):
        current_pose = self.arm.get_current_pose()
        target_pose = Pose()
        target_pose.orientation = current_pose.pose.orientation
        target_pose.position = current_pose.pose.position
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        self.arm.set_pose_target(target_pose)
        traj = self.arm.plan()
        return traj
    
    def execut(self, traj):
        self.arm.execute(traj)
        rospy.sleep(rospy.Duration(1.0))
    
    def dist_calculate(self, x1, y1, z1, x2, y2, z2):
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        return dist
    
    def get_state_and_reward(self, obj_x, obj_y, obj_z):
        [j1, j2, _, _, _, _] = self.get_joint_value()
        mid_z = 0.135*np.cos(j2)
        l = 0.135*np.sin(j2)
        mid_x = l*np.cos(j1)
        mid_y = l*np.sin(j1)
        print(mid_x, mid_y, mid_z)
        end_x, end_y, end_z = self.get_eof_location()
        dist = self.dist_calculate(end_x, end_y, end_z, obj_x, obj_y, obj_z)
        if dist < 0.1:
            goal = 1
        else:
            goal = 0
        state = np.array([mid_x-obj_x, mid_y-obj_y, mid_z-obj_z, mid_x, mid_y, mid_z,\
                          end_x-mid_x, end_y-mid_y, end_z-mid_z, end_x-obj_x, end_y-obj_y, end_z-obj_z,\
                          end_x, end_y, end_z, goal])
     
        return state

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


if __name__ == '__main__':
    dobot = arm_control()
    dobot.execut(dobot.get_traj(0.12, -0.12, 0))
    print(dobot.get_joint_value())
    dobot.get_state(0.12, -0.12, -0.05)
    # dobot.get_state()
