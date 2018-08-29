"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG
import matplotlib.pyplot as plt
from noise import noise

MAX_EPISODES = 1000
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)


noise_mean = 0
noise_std_dev = 0.5
noise_theta = 0.25
noise = noise(a_dim, noise_mean, noise_std_dev, noise_theta)
steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        noise.reset()
        ep_r = 0.
        
        for j in range(MAX_EP_STEPS):
        #j = 0
        #while not (s[0] > 400 or s[1] > 400 or s[0] < 0 or s[1] < 0):

            #env.render()

            a = rl.choose_action(s) + noise.sample_noise()

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_

            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i | position: %i, %i, %i' % (i, '---' if not done else 'done', ep_r, j, int(env.uav_pos[0]), int(env.uav_pos[1]), int(env.uav_pos[2])))
                break
    #j += 1
    #plt.plot(x,range(MAX_EPISODES))
    #plt.plot(x,range(MAX_EPISODES))
    #plt.plot(x,range(MAX_EPISODES))
    #plt.show
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(MAX_EP_STEPS):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()
