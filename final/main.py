"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl2 import DDPG
import torch

MAX_EPISODES = 900
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)
print(rl)
steps = []
def main():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()
            s = torch.Tensor(s)
            s = s.view(s.size(0), -1)
            a = rl.select_action(s)

            s_, r, done = env.step(a)

            rl.replay_buffer.push((s, s_, a, r, np.float(done)))

            s = s_
            ep_r += r

            if done or j == MAX_EP_STEPS-1:
                break
        print('Episode:{}, Step:{}, Total reward:{}'.format(i, j, ep_r))
        rl.update()
    rl.save()


if __name__ == '__main__':
    main()



