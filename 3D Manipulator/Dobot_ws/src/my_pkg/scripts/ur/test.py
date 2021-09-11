#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import tf


tau = 0.005  # target smoothing coefficient
gamma = 0.99  # discounted factor
max_episode = 50000
capacity = 50000  # replay buffer size
"""batch size原来为100"""
batch_size = 128  # mini batch size
# optional parameters
EPISILO = 0.9
log_interval = 10  # 每隔多少个回合将神经网络的参数保存
update_iteration = 1
action_dim = 5  # 机械臂三轴
state_dim = 29
max_action = 6.
directory = '/home/zyf/dobot_ws/models_and_rewards/models/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 700)
        self.l2 = nn.Linear(700, 500)
        self.l3 = nn.Linear(500, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 700)
        self.l2 = nn.Linear(700, 500)
        self.l3 = nn.Linear(500, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = Replay_buffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        if np.random.randn() <= EPISILO:
            # state = torch.FloatTensor(state.reshape(1, -1))
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state).cpu().data.numpy().flatten()
        else:
            action = np.random.uniform(-0.5, 0.5, action_dim)
        return action

    def update(self, eps):
        # print('learning..')
        learning_rate = 0.0005/(1+0.007*eps)
        learning_rate = np.clip(learning_rate, 0.0001, 0.001)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        for it in range(update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), '/home/zyf/dobot_ws/models_and_rewards/models/anno_DDPG_actor.pth')
        torch.save(self.critic.state_dict(), '/home/zyf/dobot_ws/models_and_rewards/models/anno_DDPG_critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load('/home/zyf/dobot_ws/models_and_rewards/models/anno_DDPG_actor.pth'))
        self.critic.load_state_dict(torch.load('/home/zyf/dobot_ws/models_and_rewards/models/anno_DDPG_critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


class ur_control():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ur_control')

        self.arm = moveit_commander.MoveGroupCommander('manipulator')
        self.arm.set_max_acceleration_scaling_factor(1)
        self.arm.set_max_velocity_scaling_factor(1)
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.05)
        self.arm.allow_replanning(True)
        self.arm.set_planning_time(10)

        self.arm.set_pose_reference_frame('base_link')

        self.listener = tf.TransformListener()

    def go_home(self):
        self.arm.set_named_target('home')
        self.arm.go()

    def get_eof_locaton(self):
        pose = self.arm.get_current_pose()
        end_x = pose.pose.position.x
        end_y = pose.pose.position.y
        end_z = pose.pose.position.z
        end_z -= 1
        return end_x, end_y, end_z

    def init_object_location(self):
        # x = random.uniform(-0.5, 0.5)
        # y = random.uniform(-0.5, 0.5)
        # z = random.uniform(0.1, 0.5)
        r = random.uniform(0, 0.80)
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2*np.pi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

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

    def verify_executable(self):
        x, y, z = self.init_object_location()
        traj = self.get_traj(x, y, z)
        while traj.joint_trajectory.joint_names == []:
            x, y, z = self.init_object_location()
            traj = self.get_traj(x, y, z)
        return traj, x, y, z

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

    def execut(self, traj):
        self.arm.execute(traj)
        rospy.sleep(rospy.Duration(1.0))

    def dist_calculate(self, x, y, z, obj_x, obj_y, obj_z):
        dist = np.sqrt((x-obj_x)**2 + (y-obj_y)**2 + (z-obj_z)**2)
        return dist

    def get_state(self, obj_x, obj_y, obj_z):
        joint_location = np.zeros([5, 3])
        joint_location = pd.DataFrame(joint_location, columns=['x', 'y', 'z'],
                                      index=['upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'])
        for joint in ['upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']:
            obj_joint_frame = PoseStamped()
            obj_joint_frame.header.stamp = rospy.Time()
            obj_joint_frame.header.frame_id = joint
            obj_joint_frame.pose.position.x = 0
            obj_joint_frame.pose.position.y = 0
            obj_joint_frame.pose.position.z = 0
            x = 0
            while x == 0:
                try:
                    obj_robot_frame = self.listener.transformPose('base_link', obj_joint_frame)
                    x = 1
                except:
                    x = 0
            joint_location['x'][joint] = obj_robot_frame.pose.position.x
            joint_location['y'][joint] = obj_robot_frame.pose.position.y
            joint_location['z'][joint] = obj_robot_frame.pose.position.z
        # print(joint_location)
        #                      x         y         z
        # upper_arm_link -0.068939  0.117058  0.089159
        # forearm_link    0.282409  0.185061  0.347778
        # wrist_1_link    0.542330  0.338136  0.097042
        # wrist_2_link    0.495136  0.418272  0.097042
        # wrist_3_link    0.498133  0.420037  0.002455

        upper_link_to_obj = self.dist_calculate(joint_location['x']['upper_arm_link'], joint_location['y']['upper_arm_link'],
                                                joint_location['z']['upper_arm_link'], obj_x, obj_y, obj_z)
        forearm_link_to_obj = self.dist_calculate(joint_location['x']['forearm_link'], joint_location['y']['forearm_link'],
                                                  joint_location['z']['forearm_link'], obj_x, obj_y, obj_z)
        wrist_1_link_to_obj = self.dist_calculate(joint_location['x']['wrist_1_link'], joint_location['y']['wrist_1_link'],
                                                  joint_location['z']['wrist_1_link'], obj_x, obj_y, obj_z)
        wrist_2_link_to_obj = self.dist_calculate(joint_location['x']['wrist_2_link'], joint_location['y']['wrist_2_link'],
                                                  joint_location['z']['wrist_2_link'], obj_x, obj_y, obj_z)
        wrist_3_link_to_obj = self.dist_calculate(joint_location['x']['wrist_3_link'], joint_location['y']['wrist_3_link'],
                                                  joint_location['z']['wrist_3_link'], obj_x, obj_y, obj_z)

        end_x, end_y, end_z = self.get_eof_locaton()
        end_to_obj = self.dist_calculate(end_x, end_y, end_z, obj_x, obj_y, obj_z)
        if end_to_obj < 0.05:
            goal = 1.
        else:
            goal = 0.
        state = np.array([end_x, end_y, end_z, end_to_obj, obj_x-end_x, obj_y-end_y, obj_z-end_z,

                          joint_location['x']['upper_arm_link'], joint_location['y']['upper_arm_link'],
                          joint_location['z']['upper_arm_link'], upper_link_to_obj,
                          obj_x-joint_location['x']['upper_arm_link'],
                          obj_y-joint_location['y']['upper_arm_link'], obj_z-joint_location['z']['upper_arm_link'],

                          joint_location['x']['forearm_link'], joint_location['y']['forearm_link'],
                          joint_location['z']['forearm_link'], forearm_link_to_obj,
                          obj_x-joint_location['x']['forearm_link'],
                          obj_y-joint_location['y']['forearm_link'], obj_z-joint_location['z']['forearm_link'],

                          joint_location['x']['wrist_1_link'], joint_location['y']['wrist_1_link'],
                          joint_location['z']['wrist_1_link'], wrist_1_link_to_obj,
                          obj_x-joint_location['x']['wrist_1_link'],
                          obj_y-joint_location['y']['wrist_1_link'], obj_z-joint_location['z']['wrist_1_link'],

                          joint_location['x']['wrist_2_link'], joint_location['y']['wrist_2_link'],
                          joint_location['z']['wrist_2_link'], wrist_2_link_to_obj,
                          obj_x-joint_location['x']['wrist_2_link'],
                          obj_y-joint_location['y']['wrist_2_link'], obj_z-joint_location['z']['wrist_2_link'],

                          joint_location['x']['wrist_3_link'], joint_location['y']['wrist_3_link'],
                          joint_location['z']['wrist_3_link'], wrist_3_link_to_obj,
                          obj_x-joint_location['x']['wrist_3_link'],
                          obj_y-joint_location['y']['wrist_3_link'], obj_z-joint_location['z']['wrist_3_link']])
        # state = (state - state.mean())/state.std()
        state = np.hstack((state, goal))
        return state

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


def main():
    ur = ur_control()
    # joint = anno.get_joint_value()
    # print(joint)
    obj_x, obj_y, obj_z = 0.5, 0.5, 0.5
    traj = ur.get_traj(obj_x, obj_y, obj_z)
    ur.execut(traj)
    print(ur.get_eof_locaton())
    state = ur.get_state(obj_x, obj_y, obj_z)
    print(state)


if __name__ == '__main__':
    main()
