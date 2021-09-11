#!/usr/bin/env python
# -- coding:UTF-8 --

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
import random
import numpy as np


class arm_control():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dobot_contorl')
        self.arm = moveit_commander.MoveGroupCommander('magician_arm')
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.05)
        self.arm.allow_replanning(True)
        self.arm.set_planning_time(10)

    def set_joint_value(self, joint_values):
        self.arm.set_joint_value_target(joint_values)
        self.arm.go(wait=True)

    def get_joint_value(self):
        joint_values = self.arm.get_current_joint_values()
        return joint_values


def main():
    arm = arm_control()
    jv = arm.get_joint_value()
    print(jv)
    arm.set_joint_value([0, 0.7834531597773875, 0, 0.7834417538970921, 0, 0])
    # arm.set_joint_value([0, 0.5, 0, 0.5, 0, 0])


if __name__ == '__main__':
    main()
