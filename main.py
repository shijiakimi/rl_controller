"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env1 import ArmEnv
from rl import DDPG
import matplotlib.pyplot as plt
from noise import noise

MAX_EPISODES = 1500
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
a_scale = 500
rl = DDPG(a_dim, s_dim, a_scale)


noise_mean = 0
noise_std_dev = 5
noise_theta = 2.5
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

            a = rl.choose_action(s)
            #print "action: ", a
            a += noise.sample_noise()
            s_, r, done = env.step(a)
            #if(abs(s_[6]) > env.goal['x'] or s_[7] > env.goal['y'] or s_[8] > env.goal['z']):
                #continue
            if(abs(s_[6]) < env.goal['x'] and abs(s_[7]) < env.goal['y'] and abs(s_[8]) < env.goal['z']):
                rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                # print "rl.learn"
                rl.learn()

            s = s_

            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i | %.2f, %.2f, %.2f | %.2f, %.2f, %.2f' % (i, '---' if not done else 'done', ep_r, j, env.uav_init_pos[0], env.uav_init_pos[1], env.uav_init_pos[2], env.uav_pos[0], env.uav_pos[1], env.uav_pos[2]))
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
