#!/usr/bin/env python
#-- coding:UTF-8 --
# 获得机械臂末端的xyz坐标

import sys
import rospy
import moveit_commander
import tf
from geometry_msgs.msg import Pose
import time
import random


moveit_commander.roscpp_initialize(sys.argv)  
rospy.init_node('dobot_contorl')  

arm = moveit_commander.MoveGroupCommander('magician_arm')
print(arm.get_current_pose())
print(type(arm.get_current_pose().pose.position.x))
print(arm.get_current_joint_values())
moveit_commander.roscpp_shutdown()
moveit_commander.os._exit(0)