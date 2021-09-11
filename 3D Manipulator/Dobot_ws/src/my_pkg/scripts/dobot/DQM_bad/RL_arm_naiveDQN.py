#!/usr/bin/env python3
#-- coding:UTF-8 --
# RL_arm_control+naiveDQN控制
# naive似乎不能用 
# 放弃


import sys
import rospy
import moveit_commander
import tf
from geometry_msgs.msg import Pose
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 20000
Q_NETWORK_ITERATION = 100
episodes = 400
num_steps = 50
# env = gym.make("CartPole-v0")
# env = env.unwrapped
NUM_ACTIONS = 3  # 机械臂三轴
NUM_STATES = 3 + 3  # 机械臂末端坐标xyz，目标物品坐标xyz
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape




class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        # 将1插入torch.FloatTensor(state)的维度的第0位，即得到一个一维的向量
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action = self.eval_net.forward(state)
            '''
            action_value = self.eval_net.forward(state)
            # torch.max输出action_value中以行为准（第二个参数）的最大值，返回较大值和较大值的索引，在转换为numpy类型
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            '''
        else: # random policy
            action = np.random.uniform(0, 6.28, NUM_ACTIONS)
            # action = np.random.randint(0,NUM_ACTIONS)
            # action = action  if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
        x = random.uniform(0.1, 0.24)
        y = random.uniform(0.1, 0.24)
        z = random.uniform(-0.125, 0.32)
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
        self.arm.set_joint_value_target(joint_values)
        self.arm.go()

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

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


def main():
    dobot = arm_control()
    dqn = DQN()
    for eps in range(episodes):
        ep_reward = 0
        dobot.go_home()
        # 初始化state
        obj_x, obj_y, obj_z = dobot.verify_executable()
        eof_x, eof_y, eof_z = dobot.get_eof_location()
        state = (eof_x, eof_y, eof_z, obj_x, obj_y, obj_z)
        for step in range(num_steps):
            # 获得action
            action = dqn.choose_action(state).numpy()
            
            # 执行action
            joint_values = dobot.get_joint_value()
            joint_values = joint_values + [action[0], action[1], 0, action[2], 0, 0]
            dobot.set_joint_value()
            
            # 获得next_state
            next_eof_x, next_eof_y, next_eof_z = dobot.get_eof_location()
            next_state = (next_eof_x, next_eof_y, next_eof_z, obj_x, obj_y, obj_z)
            
            # 计算reward
            # 到达回合最大步数惩罚
            if step == num_steps-1:
                max_step_reward = -50
            # 机械臂末端触碰地面惩罚
            if next_eof_z < -0.05:
                touch_ground_reward = -50
            # 以机械臂末端到达目标物品的距离的负数作为奖励
            distance_reward = -((next_eof_x-obj_x)**2+(next_eof_y-obj_y)**2+(next_eof_z-obj_z)**2)**0.5
            if distance_reward == 0:
                done =True
            # 步数惩罚
            step_reward = - step * 0.05
            reward = distance_reward + step_reward + max_step_reward + touch_ground_reward

            # 保存数据
            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(eps, round(ep_reward, 3)))
            if done:
                break
            state = next_state