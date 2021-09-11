#!/usr/bin/env python
# -- coding:UTF-8 --

import moveit_commander
from geometry_msgs.msg import PoseStamped
import sys
import rospy


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('add_object')
arm = moveit_commander.MoveGroupCommander('magician_arm')
sense = moveit_commander.PlanningSceneInterface()

sense.remove_world_object('box')

# box_size = [0.05, 0.05, 0.05]
# box_pose = PoseStamped()
# box_pose.header.frame_id = 'world'
# box_pose.pose.position.x = 0.5
# box_pose.pose.position.y = 0.0
# box_pose.pose.position.z = -0.07
# box_pose.pose.orientation.w = 1.0
# sense.add_box('box', box_pose, box_size)

# moveit_commander.roscpp_shutdown()
# moveit_commander.os._exit(0)
