
from env1 import ArmEnv
from rl_init import DDPG
import matplotlib.pyplot as plt
from noise import noise
import numpy as np

MAX_EPISODES = 3000
MAX_EP_STEPS = 100
ON_TRAIN = True

# set env
env = ArmEnv([0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1000., 1000., 1000., 1000.])
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
a_scale = [-1.,1.]
#a_scale = [0, 10]
#print a_scale
sz = max(env.goal['x'], max(env.goal['y'], env.goal['z']))
linear_upper = 300
angular_upper = np.pi
prop_speed_low = 1
prop_speed_high = 3000
s_bound_upper = [sz, np.pi / 2, linear_upper, angular_upper, prop_speed_high]
s_bound_lower = [-sz, -np.pi / 2, -linear_upper, -angular_upper, prop_speed_low]
rl = DDPG(a_dim, s_dim, a_scale, s_bound_upper, s_bound_lower)
#print "after DDPG init"

noise_mean = 0
noise_std_dev = 0.2
noise_theta = 0.15
noise_dt = env.dt
noise = noise(a_dim, noise_mean, noise_std_dev, noise_theta, noise_dt)
steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        print i
        #s = env.reset()
        #noise.reset()
        #ep_r = 0.
        rl.learn()
        s = env.reset()
        for j in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            #s, r, done = env.step(a)
            s[:3] += a[:3]
        print s[:3]
            #j = 0
            #while not (s[0] > 400 or s[1] > 400 or s[0] < 0 or s[1] < 0):

            #env.render()

            #a = rl.choose_action(s)
            #print "action: ", a
            #n = noise.sample_noise()
            #print n
            #a = a + n
            #s_, r, done = env.step(a)
            #if(abs(s_[6]) > env.goal['x'] or s_[7] > env.goal['y'] or s_[8] > env.goal['z']):
                #continue
            #if(abs(s_[8]) < 2 * (env.goal['z'] + env.goal['l'])):
            #rl.store_transition(s, a, r, s_)

            #ep_r += r
            #if rl.memory_full:
                # start to learn once has fulfilled the memory

            #s = s_

            #if done or j == MAX_EP_STEPS-1:
                #print('Ep: %i | %s | ep_r: %.1f | step: %i | %.2f, %.2f, %.2f | %.2f, %.2f, %.2f | %.2f, %.2f, %.2f'% (i, '---' if not done else 'done', ep_r, j, env.uav_init_pos[0], env.uav_init_pos[1], env.uav_init_pos[2], env.uav_pos[0], env.uav_pos[1], env.uav_pos[2], env.uav_pos[0] - env.uav_init_pos[0], env.uav_pos[1] - env.uav_init_pos[1], env.uav_pos[2] - env.uav_init_pos[2]))
                #break
    #j += 1
    #plt.plot(x,range(MAX_EPISODES))
    #plt.plot(x,range(MAX_EPISODES))
    #plt.plot(x,range(MAX_EPISODES))
    #plt.show
    rl.save()


def eval():
    rl.restore()
    #env.render()
    #env.viewer.set_vsync(True)
    i = 0
    while i < 200:
        s = env.reset()
        for _ in range(MAX_EP_STEPS):
            #env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
        print s[:3]


if ON_TRAIN:
    train()
else:
    eval()
