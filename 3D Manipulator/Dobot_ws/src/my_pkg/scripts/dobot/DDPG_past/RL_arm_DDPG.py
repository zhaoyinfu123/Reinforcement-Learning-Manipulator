#!/usr/bin/env python
#-- coding:UTF-8 --

import sys
import os
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
import os
from tensorboardX import SummaryWriter
from itertools import count

# hyper-parameters
mode ='train' # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
env_name = "Pendulum-v0"
tau = 0.005 # target smoothing coefficient
target_update_interval = 1
test_iteration = 10

learning_rate = 1e-4
gamma = 0.99 # discounted factor
capacity = 1000000 # replay buffer size
batch_size = 100 # mini batch size
seed = False
random_seed = 9527
# optional parameters
EPISILO = 0.9
sample_frequency = 2000
render =False  # show UI or not
log_interval = 50  # 每隔多少个回合将神经网络的参数保存
load = False # load model
render_interval = 100 # after render_interval, the env.render() will work
exploration_noise = 0.1
max_episode = 10000 # num of games
print_log = 5
update_iteration = 200
max_length_of_trajectory = 200
action_dim = 3  # 机械臂三轴
state_dim = 3 + 3  # 机械臂末端坐标xyz，目标物品坐标xyz
max_action = 1.5
min_Val = torch.tensor(1e-7).float()
script_name = os.path.basename(__file__)
directory = './exp' + script_name + env_name +'./'


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

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        if np.random.randn() <= EPISILO:
            # state = torch.FloatTensor(state.reshape(1, -1))
            state = torch.FloatTensor(state.reshape(1, -1))
            action = self.actor(state).data.numpy().flatten()
        else:
            action = np.random.uniform(-0.5, 0.5, action_dim)
        return action

    def update(self):
        print('learning..')
        for it in range(update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1-d)
            reward = torch.FloatTensor(r)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

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
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
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
        print('find a executable object location')
        return x, y, z

    def go_home(self):
        self.arm.set_named_target('home')
        self.arm.go()
        rospy.sleep(rospy.Duration(1.0))

    def set_joint_value(self, joint_values):
        try:
            # print('Joint values are executable')
            self.arm.set_joint_value_target(joint_values)
            self.arm.go(wait=True)
            push = 1
            # print('moved')
            # rospy.sleep(rospy.Duration(1.0))
        except:
            # print('Joint values are not executable')
            push = 0
            pass
        return push

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
    
    # def execut(self, traj):
    #     self.arm.execute(traj)
    #     rospy.sleep(rospy.Duration(1.0))

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


def main():
    dobot = arm_control()
    agent = DDPG(state_dim, action_dim, max_action)
    agent.load()
    workbook = xlwt.Workbook('utf-8')
    worksheet = workbook.add_sheet('eps_reward')
    for eps in range(max_episode):
        total_reward = 0
        step = 0
        dobot.go_home()
        obj_x, obj_y, obj_z = dobot.verify_executable()
        eof_x, eof_y, eof_z = dobot.get_eof_location()
        state = np.array([eof_x, eof_y, eof_z, obj_x, obj_y, obj_z])
        # for step in count():
        for step in range(200):
            state = state * 1000
            # print(state)
            action = agent.select_action(state)
            # print(action)
            joint_values = dobot.get_joint_value()
            joint_values = [joint_values[0]+action[0], joint_values[1]+action[1], joint_values[2]\
                          , joint_values[3]+action[2], joint_values[4], joint_values[5]]
            joint_values[0] = np.clip(joint_values[0], -1.5, 1.5)
            joint_values[1] = np.clip(joint_values[1], 0.4, 1.1)
            joint_values[3] = np.clip(joint_values[3], 0.4, 1.1)
            push = dobot.set_joint_value(joint_values)
            next_eof_x, next_eof_y, next_eof_z = dobot.get_eof_location()
            next_state = np.array([next_eof_x, next_eof_y, next_eof_z, obj_x, obj_y, obj_z])
            distance_reward = -((next_eof_x-obj_x)**2+(next_eof_y-obj_y)**2+(next_eof_z-obj_z)**2)**0.5*10
            step_reward = - step * 0.1
            # print(distance_reward)
            # print(step_reward)
            if -distance_reward < 0.05:
                done = True
                done_reward = 300
            else:
                done = False
                done_reward = 0
            if push == 1:
                reward = distance_reward + step_reward + done_reward
                # print(state, next_state*1000, action, reward, np.float(done))
                agent.replay_buffer.push((state, next_state*1000, action, reward, np.float(done)))
            else:
                reward = 0
            state = next_state
            step += 1
            total_reward += reward
            if done:
                break
        print("Episode: \t{} Take step:\t{} Total Reward: \t{:0.2f}".format(eps, step, total_reward))
        agent.update()
        # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

        if eps % log_interval == 0:
            agent.save()
        worksheet.write(eps, 0, round(total_reward, 3))
        workbook.save('eps_reward.xls')


if __name__ == '__main__':
    main()