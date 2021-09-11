import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('dobot_contorl')
arm = moveit_commander.MoveGroupCommander('magician_arm')
arm.set_goal_position_tolerance(0.01)
arm.set_goal_orientation_tolerance(0.05)
arm.allow_replanning(True)
arm.set_planning_time(10)

current_pose = arm.get_current_pose()
target_pose = Pose()
target_pose.orientation = current_pose.pose.orientation
target_pose.position = current_pose.pose.position
target_pose.position.x = 0.12
target_pose.position.y = 0.12
target_pose.position.z = 0
arm.set_pose_target(target_pose)
traj = arm.plan()
arm.execute(traj)
