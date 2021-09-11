import xlrd
from matplotlib import pyplot as plt


sheet_spilt = xlrd.open_workbook('DDPG_rewards.xls')
sheet_spilt = sheet_spilt.sheets()[0]
reward_spilt = sheet_spilt.col_values(0)
step_spilt = sheet_spilt.col_values(1)

# sheet_sqrt = xlrd.open_workbook('DDPG_rewards_sqrt.xls')
# sheet_sqrt = sheet_sqrt.sheets()[0]
# reward_sqrt = sheet_sqrt.col_values(0)
# step_sqrt = sheet_sqrt.col_values(1)

plt.subplot(211)
plt.plot(reward_spilt, 'r')
# plt.plot(reward_sqrt, 'b')

plt.subplot(212)
plt.plot(step_spilt, 'g')
# plt.plot(step_sqrt, 'b')
plt.show()
