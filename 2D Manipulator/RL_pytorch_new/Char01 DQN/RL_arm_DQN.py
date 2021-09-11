#!/usr/bin/env python
#-- coding:UTF-8 --
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gym


# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
episodes = 3000
num_steps = 100
env = gym.make('MountainCar-v0')
env = env.unwrapped
NUM_STATES = env.observation_space.shape[0] # 2
NUM_ACTIONS = env.action_space.n


class Net(nn.Module):
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


class DQN(object):
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))  # MC行， NS*2+2列
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def save_weights(self, eps, rewards):
        torch.save(self.eval_net.state_dict(), \
            './dobot_ws/src/my_pkg/models/eps:{}+reward:{}.h5'.format(eps, rewards))

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
        # action = np.clip(action, -0.1, 0.1)
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
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    dqn = DQN()
    step_count_list = []

    render = True
    for eps in range(10000):
        state = env.reset()
        step_counter = 0
        eps_reward = 0
        if render:
            env.render()
        for step in range(1000):
            step_counter += 1
            if render:
                env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            position, vilocity = next_state
            reward = abs(position - (-0.5))
            dqn.store_transition(state, action, reward, next_state)
            eps_reward += reward
            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
            if done:
                break
            state = next_state
        print(step_counter)

if __name__ == '__main__':
    main()