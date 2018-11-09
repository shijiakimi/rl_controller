import sys
import numpy as np
from RLDDPG import DDPG
from task import Landing as Task
from OUNoise import OUNoise

max_episodes = 1000
target_pos = np.array([0., 0., 0.])
task = Task(target_pos=target_pos, init_pose=[0., 0., 10.0, 0., 0., 0.])
agent = DDPG(task)
scores = []
grades = []
avg_reward = []
best_score = -np.inf
noise = OUNoise(task.action_size, mu=0.0, theta=0.2, sigma=0.1)
noise_annealing = 0.001
noise_min_sigma = 0.01

for i in range(max_episodes):
    state = agent.reset_episode()
    step = 0
    score = 0.
    noise.reset(noise_annealing, noise_min_sigma)
    while True:
        step += 1
        action = agent.act(state)
        action += noise.sample()
        action = np.clip(action, -1, 1)

        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state

        score += reward
        if done:
            print('Ep: %i | %s | ep_r: %.1f | step: %i | %.2f, %.2f, %.2f | %.2f, %.2f, %.2f | %.2f, %.2f, %.2f'% (i, '---' if not done else 'done', score, step, task.sim.init_pose[0], task.sim.init_pose[1], task.sim.init_pose[2], task.sim.pose[0], task.sim.pose[1], task.sim.pose[2], task.sim.pose[0] - task.sim.init_pose[0], task.sim.pose[1] - task.sim.init_pose[1], task.sim.pose[2] - task.sim.init_pose[2]))
            break
