#!/usr/bin/env python
# -- coding:UTF-8 --

"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env_s_endmidxy_endmidtoobj_i_n_rr import ArmEnv
from rl2 import DDPG
import numpy as np
from itertools import count

MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(s_dim, a_dim, a_bound)
# rl.load()


def main():
    # start training
    for i in range(1001):
        s = env.reset()
        ep_r = 0.

        # for j in range(MAX_EP_STEPS):
        for j in count():
            env.render()

            a = rl.select_action(s)
            # print(a)
            # print(type(a))
            s_, r, done = env.step(a)

            rl.replay_buffer.push((s, s_, a, r, np.float(done)))
            if len(rl.replay_buffer.storage) > 100:
                rl.update()
            s = s_
            ep_r += r

            if done or j == MAX_EP_STEPS-1:
                break
        # if len(rl.replay_buffer.storage) > 1000:
        #     rl.update()
        print('Episode:{}, Step:{}, Total reward:{}'.format(i, j, ep_r))
        rl.save(i, ep_r, j)


if __name__ == '__main__':
    main()
