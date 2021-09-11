#!/usr/bin/env python
#-- coding:UTF-8 --

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import time
import random


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

    def verify_executable(self):
        x, y, z = self.init_object_location()
        traj = self.get_traj(x, y, z)
        while traj.joint_trajectory.joint_names == []:
            x, y, z = self.init_object_location()
            traj = self.get_traj(x, y, z)
        print('find a executable object location')
        return x, y, z

    def set_joint_value(self, j1, j2, j3):
        self.arm.set_joint_value_target([j1, j2, 999, j3, 999, 999])
        self.arm.go()

    def get_joint_value(self):
        print(self.arm.get_current_joint_values())

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)



def main():
    for i in range(10):
        j1 = random.uniform(-1.57, 1.57)
        j2 = random.uniform(0, 1.48)
        j3 = random.uniform(-0.17, 1.57)
        dobot_arm = arm_control()
        try:
            dobot_arm.set_joint_value(j1, j2, j3)
        except:
            pass
        print('111')
        
    dobot_arm.shut_down()


if __name__ == '__main__':
    main()
