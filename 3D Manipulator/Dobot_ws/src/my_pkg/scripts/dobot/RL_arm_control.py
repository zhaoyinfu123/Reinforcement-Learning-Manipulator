#!/usr/bin/env python
# -- coding:UTF-8 --

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
import random
import numpy as np


BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
NUM_ACTIONS = 4  # 输出的action为4个轴的改变量
NUM_STATES = 3 + 4  # 3为目标xyz， 4为4个轴的当前的角度
ENV_A_SHAPE = 0


class arm_control():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dobot_contorl')
        self.arm = moveit_commander.MoveGroupCommander('magician_arm')
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.05)
        self.arm.allow_replanning(True)
        self.arm.set_planning_time(10)

        self.sense = moveit_commander.PlanningSceneInterface()

        self.end_effector_link = self.arm.get_end_effector_link()
        rospy.sleep(2)

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
        return traj, x, y, z

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

    def dist_calculate(self, x, y, z, obj_x, obj_y, obj_z):
        dist = np.sqrt((x-obj_x)**2 + (y-obj_y)**2 + (z-obj_z)**2)
        return dist

    def get_state(self, obj_x, obj_y, obj_z):
        [j1, j2, _, _, _, _] = self.get_joint_value()
        mid_z = 0.135*np.cos(j2)
        link = 0.135*np.sin(j2)
        mid_x = link*np.cos(j1)
        mid_y = link*np.sin(j1)
        end_x, end_y, end_z = self.get_eof_location()
        end_to_obj = self.dist_calculate(end_x, end_y, end_z, obj_x, obj_y, obj_z)
        mid_to_obj = self.dist_calculate(mid_x, mid_y, mid_z, obj_x, obj_y, obj_z)
        if end_to_obj < 0.05:
            goal = 1.
        else:
            goal = 0.
        state = np.array([end_x, end_y, end_x, end_to_obj, obj_x-end_x, obj_y-end_y, obj_z-end_z,
                          mid_x, mid_y, mid_x, mid_to_obj, obj_x-mid_x, obj_y-mid_y, obj_z-mid_z])
        state = (state-state.mean())/state.std()
        state = np.hstack((state, goal))
        return state

    def add_object(self):
        box_size = [1, 1, 1]
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'world'
        box_pose.pose.position.x = 0.2
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = -0.07
        box_pose.pose.orientation.w = 1.0
        self.sense.add_box('box', box_pose, box_size)
        rospy.sleep(2)

    def remove_box(self):
        self.sense.remove_attached_object(self.end_effector_link, 'box')
        self.sense.remove_world_object('box')

    def attach_object(self):
        self.sense.attach_box(self.end_effector_link, 'box')

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


def main():
    dobot = arm_control()
    dobot.go_home()
    # dobot.remove_box()
    # dobot.add_object()
    # traj = dobot.get_traj(0.2, 0.0, -0.05)
    traj = dobot.get_traj(0.15, 0.1, -0.05)
    dobot.execut(traj)
    end_x, end_y, end_z = dobot.get_eof_location()
    print(end_x, end_y, end_z)
    dist = dobot.dist_calculate(end_x, end_y, end_z, 0.15, 0.1, -0.05)
    print(dist)
    # dobot.attach_object()
    # dobot.go_home()
    # dobot.remove_box()
    dobot.shut_down()


if __name__ == '__main__':
    main()
