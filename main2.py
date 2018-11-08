import sys
import numpy as np
from RLDDPG import DDPG
from newTask import Task

max_episodes = 1000
target_pos = np.array([0., 0., 15.])
task = Task(target_pos=target_pos, init_pose=[0., 0., 0.01, 0., 0., 0.])
agent = DDPG(task)

for i in range(max_episodes):
    state = agent.reset_episode()
    step = 0
    while True:
        step += 1
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print('Ep: %i | %s | ep_r: %.1f | step: %i | %.2f, %.2f, %.2f | %.2f, %.2f, %.2f | %.2f, %.2f, %.2f'% (i, '---' if not done else 'done', reward, step, task.sim.init_pose[0], task.sim.init_pose[1], task.sim.init_pose[2], task.sim.pose[0], task.sim.pose[1], task.sim.pose[2], task.sim.pose[0] - task.sim.init_pos[0], task.sim.pose[1] - task.sim.init_pos[1], task.sim.pose[2] - task.sim.init_pos[2]))
            break
