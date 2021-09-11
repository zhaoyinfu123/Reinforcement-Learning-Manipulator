#!/usr/bin/env python
# -- coding:UTF-8 --
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from pyfirmata import Arduino
from PIL import Image
import time
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from yolo import YOLO


tau = 0.005  # target smoothing coefficient
gamma = 0.99  # discounted factor
capacity = 50000  # replay buffer size
batch_size = 128  # mini batch size
EPISILO = 0.9
update_iteration = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
action_dim = 3  # 机械臂三轴
state_dim = 15
max_action = 6.


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
        torch.save(self.actor.state_dict(), '/home/zyf/dobot_ws/models_and_rewards/models/DDPG_actor.pth')
        torch.save(self.critic.state_dict(), '/home/zyf/dobot_ws/models_and_rewards/models/DDPG_critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(
            '/home/zyf/dobot_ws/models_and_rewards/models/DDPG_actor_random_best.pth'))
        self.critic.load_state_dict(torch.load(
            '/home/zyf/dobot_ws/models_and_rewards/models/DDPG_critic_random_best.pth'))
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
        x = round(random.uniform(0.14, 0.23), 3)
        y = round(random.uniform(-0.22, 0.22), 3)
        z = round(random.uniform(-0.06, 0.032), 3)
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

    def go_down(self, diff):
        diff = diff - 0.05
        end_x, end_y, end_z = self.get_eof_location()
        traj = self.get_traj(end_x, end_y, end_z-0.07)
        self.execut(traj)

    def go_up(self):
        end_x, end_y, end_z = self.get_eof_location()
        traj = self.get_traj(end_x, end_y, end_z+0.035)
        self.execut(traj)

    def shut_down(self):
        self.go_home()
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


class yolov3():
    def __init__(self):
        self.yolo = YOLO()
        # 声明点云对象，用于计算点云和纹理映射。
        self.pc = rs.pointcloud()
        # 我们希望点云对象是持续的，这样当有新的帧到达时，能够显示出来
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.counter = 0

    def detect(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)  # 对齐

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            img_color = np.asanyarray(color_frame.get_data())  # 转换为cv2数据类型
            self.pc.map_to(color_frame)
            points = self.pc.calculate(depth_frame)
            self.vtx = np.asanyarray(points.get_vertices())

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            frame = self.yolo.detect_image(frame)
            obj_info = self.yolo.obj_info
            self.counter += 1
            if obj_info != [] and self.counter > 5:
                break
        return obj_info

    def transform(self, obj_info):
        obj_list = []
        for obj in obj_info:
            i = 640*obj[1]+obj[0]
            obj_x = round(np.float(self.vtx[i][0]), 3)
            obj_y = round(np.float(self.vtx[i][1]), 3)
            obj_z = round(np.float(self.vtx[i][2]), 3)
        # obj_x = 0.12
        # obj_y = -0.072
        # obj_z = 0.479

            obj_x_robot_frame = 0.17 + obj_y * 1.3  # 195
            obj_y_robot_frame = obj_x*1.1
            obj_z_robot_frame = 0.41 - obj_z
            kind = obj[2]
            high = obj[3]
            wide = obj[4]
            obj_list.append([kind, obj_x_robot_frame, obj_y_robot_frame, obj_z_robot_frame, high, wide])
        return obj_list

    # def reach(self, obj_list):
    #     print(obj_list)
    #     obj_list = obj_list[0]
    #     current_pose = self.arm.get_current_pose()
    #     target_pose = Pose()
    #     target_pose.orientation = current_pose.pose.orientation
    #     target_pose.position = current_pose.psose.position
    #     target_pose.position.x = obj_list[1] - 0.05
    #     target_pose.position.y = obj_list[2] - 0.05
    #     target_pose.position.z = obj_list[3]
    #     self.arm.set_pose_target(target_pose)
    #     self.arm.go()
    #     rospy.sleep(rospy.Duration(1.0))


class sucker_control():
    def __init__(self):
        self.board = Arduino('/dev/ttyUSB1')

    def suck(self):
        self.board.digital[8].write(0)
        self.board.digital[9].write(1)
        time.sleep(3)

    def inflate(self):
        self.board.digital[8].write(1)
        self.board.digital[9].write(0)
        time.sleep(1)
        self.board.digital[8].write(0)
        self.board.digital[9].write(0)

    def loose(self):
        self.board.digital[8].write(0)
        self.board.digital[9].write(0)

        self.board.digital[8].write(1)
        self.board.digital[9].write(0)
        time.sleep(1)
        self.board.digital[8].write(0)
        self.board.digital[9].write(0)


def main():
    v3 = yolov3()
    agent = DDPG(state_dim, action_dim, max_action)
    agent.load()
    dobot = arm_control()
    traj = dobot.get_traj(0, 0.15, 0.02)
    dobot.execut(traj)
    sucker = sucker_control()
    dt = 0.2

    input()

    obj_info = v3.detect()
    obj_info = v3.transform(obj_info)
    # print(obj_info)
    # [['red jujube', 0.13490000000000002, 0.056100000000000004, -0.11900000000000005, 58, 50],
    # ['red jujube', 0.15960000000000002, -0.06380000000000001, -0.12100000000000005, 63, 56]]

    # while True:
    # obj_info = v3.detect()
    # if obj_info == []:
    #     break
    # obj_info = v3.transform(obj_info)
    # obj = obj_info[0]
    for obj in obj_info:
        dobot.go_home()
        kind = obj[0]
        x = obj[1]
        y = obj[2]
        z = obj[3]
        # z = -0.12
        theta = np.arctan(y/x)
        x_offset = 0.07*np.cos(theta)
        y_offset = 0.07*np.sin(theta)
        z_offset = 0.1  # real 0.05
        obj_x = x - x_offset
        obj_y = y - y_offset
        # obj_z = 0.01
        obj_z = z + z_offset + 0.03
        # print(obj_z)
        # 0.01
        while True:
            # step = 0
            # while True:
            state = dobot.get_state(obj_x, obj_y, obj_z)
            action = agent.select_action(state)
            joint_values = dobot.get_joint_value()
            joint_values = [joint_values[0]+action[0]*dt, joint_values[1]+action[1]*dt, joint_values[2],
                            joint_values[3]+action[2]*dt, joint_values[4], joint_values[5]]
            moved = dobot.set_joint_value(joint_values)
            if moved == 1:
                # step += 1
                end_x, end_y, end_z = dobot.get_eof_location()
                distance = dobot.dist_calculate(end_x, end_y, end_z, obj_x, obj_y, obj_z)
                # xy_distance = np.sqrt((end_x-obj_x)**2 + (end_y-obj_y)**2)
                # z_ditance = end_z - obj_z

                if distance < 0.015:
                    done = True
                else:
                    done = False
                if done:
                    break
        z_diff = z - end_z
        sucker.inflate()
        dobot.go_down(z_diff)
        sucker.suck()
        dobot.go_up()
        dobot.go_home()
        if kind == 'red jujube':
            dobot.execut(dobot.get_traj(0, 0.15, -0.015))
        elif kind == 'green jujube':
            dobot.execut(dobot.get_traj(0, -0.15, -0.015))
        sucker.loose()
        dobot.go_up()

    dobot.shut_down()


if __name__ == '__main__':
    main()
