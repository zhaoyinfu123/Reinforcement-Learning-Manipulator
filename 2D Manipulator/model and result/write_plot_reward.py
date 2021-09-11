#!/usr/bin/env python
#-- coding:UTF-8 --

import xlrd
from matplotlib import pyplot as plt
import numpy as np

sheet = xlrd.open_workbook('DDPG_rewards.xls')
sheet = sheet.sheets()[0]
reward = sheet.col_values(0)
step = sheet.col_values(1)

reward_mean = np.mean(reward)
reward_std = np.std(reward)
step_mean = np.mean(step)
step_std = np.std(step)
print('mean:{}, std:{}'.format(reward_mean, reward_std))
print('mean:{}, std:{}'.format(step_mean, step_std))

plt.subplot(211)
plt.ylabel('Rewards')
plt.xlabel('Episodes')
# plt.text(10, 10, 'mean:{}, std:{}'.format(reward_mean, reward_std), fontdict = {'size': 12, 'color': 'red'})
plt.plot(reward, 'r')
# plt.plot(reward_sqrt, 'b')

plt.subplot(212)
plt.ylabel('Steps')
plt.xlabel('Episodes')
# plt.text(10, 210, 'mean:{}, std:{}'.format(step_mean, step_std), fontdict = {'size': 12, 'color': 'red'})
plt.plot(step, 'g')
# plt.plot(step_sqrt, 'b')

plt.subplots_adjust(hspace=0.3)
plt.show()
