#!/usr/bin/env python
# -- coding:UTF-8 --

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import xlwt


tau = 0.005  # target smoothing coefficient
gamma = 0.99  # discounted factor
max_episode = 500
capacity = 50000  # replay buffer size
"""batch size原来为100"""
batch_size = 128  # mini batch size
# optional parameters
EPISILO = 0.9
log_interval = 10  # 每隔多少个回合将神经网络的参数保存
update_iteration = 1
action_dim = 3  # 机械臂三轴
state_dim = 15
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
        print('learning..')
        learning_rate = 0.0007/(1+0.007*eps)
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
        torch.save(self.actor.state_dict(), '/home/zyf/dobot_ws/models_and_rewards/models/DDPG_actor.pth')
        torch.save(self.critic.state_dict(), '/home/zyf/dobot_ws/models_and_rewards/models/DDPG_critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load('/home/zyf/dobot_ws/models_and_rewards/models/DDPG_actor.pth'))
        self.critic.load_state_dict(torch.load('/home/zyf/dobot_ws/models_and_rewards/models/DDPG_critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


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

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


def main():
    dobot = arm_control()
    agent = DDPG(state_dim, action_dim, max_action)
    dt = 0.2
    # agent.load()
    workbook = xlwt.Workbook('utf-8')
    worksheet = workbook.add_sheet('eps_rewards and running steps')
    for eps in range(max_episode):
        print(time.ctime())
        eps_rewards = 0
        eps_steps = 0

        # dobot.go_home()
        # 初始化为随机位置
        print('Go to a random location')
        traj = dobot.verify_executable()
        dobot.execut(traj)

        obj_x, obj_y, obj_z = 0.12, -0.12, -0.05
        state = dobot.get_state(obj_x, obj_y, obj_z)
        while True:
            action = agent.select_action(state)
            # print(action)
            joint_values = dobot.get_joint_value()
            joint_values = [joint_values[0]+action[0]*dt, joint_values[1]+action[1]*dt, joint_values[2],
                            joint_values[3]+action[2]*dt, joint_values[4], joint_values[5]]
            # joint_values[0] = np.clip(joint_values[0], -1.5, 1.5)
            # joint_values[1] = np.clip(joint_values[1], 0.2, 1.2)
            # joint_values[3] = np.clip(joint_values[3], 0.2, 1.2)
            moved = dobot.set_joint_value(joint_values)
            if moved == 1:
                next_eof_x, next_eof_y, next_eof_z = dobot.get_eof_location()
                next_state = dobot.get_state(obj_x, obj_y, obj_z)
                end_x, end_y, end_z = dobot.get_eof_location()
                distance = dobot.dist_calculate(end_x, end_y, end_z, obj_x, obj_y, obj_z)
                reward = -distance*10

                if distance < 0.02:
                    done = True
                else:
                    done = False

                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                eps_steps += 1
                eps_rewards += reward

                if done or eps_steps >= 200:
                    break

        print("episode: {}, took {} steps, the episode reward is {}"
              .format(eps, eps_steps, round(eps_rewards, 3)))
        agent.update(eps)
        if eps % log_interval == 0 and eps != 0:
            agent.save()
        worksheet.write(eps, 0, round(eps_rewards, 3))
        worksheet.write(eps, 1, eps_steps)
        workbook.save('/home/zyf/dobot_ws/models_and_rewards/plot_reward/DDPG_rewards.xls')


if __name__ == '__main__':
    main()
