#!/usr/bin/env python
#-- coding:UTF-8 --
# 更改了NN的大小、MEMORY_CAPACITY、BATCH_SIZE， 增加距离计算函数， state中增加了距离和0/1(用于指示末端是否接近了目标)。
# 更改choose_action的方法，每次选择一个轴改变

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xlwt
import os


# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 3000
Q_NETWORK_ITERATION = 100
EPISODES = 10000
NUM_STEP = 200
NUM_ACTIONS = 3 * 2  # 机械臂三轴
NUM_STATES = 3 + 3 + 1 + 1 + 1 # 机械臂末端坐标xyz，目标物品坐标xyz，与目标之间的距离， 末端到坐标原点的距离， indecater



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 300)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(300,300)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(300,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN(object):
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + NUM_ACTIONS + 1))
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def save(self):
        torch.save(self.eval_net.state_dict(), '/home/zyf/dobot_ws/src/my_pkg/models/RL_arm_DQN.pth')

    def load(self):
        self.eval_net.load_state_dict(torch.load('/home/zyf/dobot_ws/src/my_pkg/models/RL_arm_DQN.pth'))
        print('=========================')
        print('Weight has been loaded...')
        print('=========================')

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        if np.random.randn() <= EPISILO:  # greedy policy
            
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            '''
            action = self.eval_net.forward(state)
            action = action.detach().numpy()
            action = action[0]
            '''
        else:  # random policys
            
            action = np.random.randint(0, NUM_ACTIONS)
            '''
            action = np.random.uniform(-0.5, 0.5, NUM_ACTIONS)
            '''
            # clip action j1[-0.7, 0.7] j2[0.6, 1.00] j3[0.40, 0.80]
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, reward, next_state))
        # print(transition)
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # 将eval_net的参数 复制给target_net
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 从MC中取出BS个数
        batch_memory = self.memory[sample_index, :]  # 从memory中选择出来这几个数对应的行
        # memory为 state action reward next_state 从batch中选出对应的
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        # q_eval
        # batch中包含了许多行 通过net输出行们对应的action的值
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # loss = self.loss_func(q_eval, q_target)
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
            m = 1
            # print('moved')
            # rospy.sleep(rospy.Duration(1.0))
        except:
            # print('Joint values are not executable')
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
    
    # def execut(self, traj):
    #     self.arm.execute(traj)
    #     rospy.sleep(rospy.Duration(1.0))

    def shut_down(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


def distance_culculate(x1, y1, z1, x2, y2, z2):
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return dist


def get_state(x1, y1, z1, x2, y2, z2):
    dist1 = distance_culculate(x1, y1, z1, x2, y2, z2)
    dist2 = distance_culculate(x1, y1, z1, 0, 0, 0)
    if dist1 > 0.05:
        nearby = 0
    else:
        nearby = 1
    state = torch.Tensor([x1, y1, z1, x2, y2, z2, dist1, dist2, nearby])
    # state = torch.Tensor([x1, y1, z1, x2, y2, z2, dist1 ,nearby])
    return state


def main():
    dobot = arm_control()
    dqn = DQN()
    # dqn.load()
    workbook = xlwt.Workbook('utf-8')
    worksheet = workbook.add_sheet('eps_rewards and running steps')
    for eps in range(EPISODES):
        ep_reward = 0
        dobot.go_home()
        # 初始化state
        # obj_x, obj_y, obj_z = dobot.verify_executable()
        obj_x, obj_y, obj_z = 0.121, -0.119, -0.069
        # print(obj_x, obj_y, obj_z)
        eof_x, eof_y, eof_z = dobot.get_eof_location()
        # state = torch.Tensor([eof_x, eof_y, eof_z, obj_x, obj_y, obj_z, 1, 1])
        state = get_state(eof_x, eof_y, eof_z, obj_x, obj_y, obj_z)
        eps_steps = 0
        while True:
            # 获得action
            action = dqn.choose_action(state)
            # 执行action
            joint_values = dobot.get_joint_value()
            joint_values = [joint_values[0]+action[0], joint_values[1]+action[1], joint_values[2]\
                          , joint_values[3]+action[2], joint_values[4], joint_values[5]]
            joint_values[0] = np.clip(joint_values[0], -1.5, 1.5)
            joint_values[1] = np.clip(joint_values[1], 0.2, 1.2)
            joint_values[3] = np.clip(joint_values[3], 0.2, 1.2)
            moved = dobot.set_joint_value(joint_values)
            
            if moved == 1:
                # 获得next_state
                next_eof_x, next_eof_y, next_eof_z = dobot.get_eof_location()
                next_state = get_state(next_eof_x, next_eof_y, next_eof_z, obj_x, obj_y, obj_z)
                
                # 计算rewar1
                # 到达回合最大步数惩罚
                # if step == num_steps-1:
                #     max_step_reward = -5
                # else:
                #     max_step_reward = 0
                # 机械臂末端触碰地面惩罚
                # if next_eof_z < -0.05:
                #     touch_ground_reward = -5
                # else:
                #     touch_ground_reward = 0
                # 以机械臂末端到达目标物品的距离的负数作为奖励
                distance = distance_culculate(next_eof_x, next_eof_y, next_eof_z, obj_x, obj_y, obj_z)
                if distance < 0.01:
                    done = True
                else:
                    done = False
                # 步数惩罚
                # step_reward = - step * 0.05
                # reward = distance_reward + step_reward + max_step_reward + touch_ground_reward
                if (next_eof_y<0 and obj_y>0) or (next_eof_y>0 and obj_y<0):
                    area_reward = -10
                else:
                    area_reward = 0
                reward = -distance  + area_reward

                # 保存数据
                # print(state, action, reward, next_state)
                dqn.store_transition(state, action, reward, next_state)
                ep_reward += reward

                if dqn.memory_counter >= MEMORY_CAPACITY:
                    try:
                        dqn.learn()
                        if eps_steps % 50 == 0 and eps_steps != 0:
                            print('learning')
                    except:
                        print('runtime error')
                        pass
                if done:
                    break
                state = next_state
                eps_steps += 1

            if eps_steps >= NUM_STEP:
                break

        dqn.save()
        worksheet.write(eps, 0, round(ep_reward, 3))
        worksheet.write(eps, 1, eps_steps)
        workbook.save('/home/zyf/dobot_ws/src/plot_reward/DQN_rewards.xls')
        print("episode: {}, took {} steps, the episode reward is {}"\
                .format(eps, eps_steps, round(ep_reward, 3)))

if __name__ == '__main__':
    main()

