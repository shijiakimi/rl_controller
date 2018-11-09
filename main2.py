import sys
import numpy as np
from RLDDPG import DDPG
from newTask import Task
from OUNoise import OUNoise
import matplotlib.pyplot as plt



init_pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
target_position = np.array([0.0, 0.0, 10.0])
task = Task(init_pose=init_pose, target_pos=target_position, pos_noise=0.25, ang_noise=None,
            vel_noise=0.15, ang_vel_noise=None)

agent = DDPG(task)
scores = []
grades = []
avg_reward = []
best_score = -np.inf
noise = OUNoise(task.action_size, mu=0.0, theta=0.2, sigma=0.1)

def train(number_episodes, runtime=5.0, init_pos=None, target_position=None, printOut=True):
    global agent, scores, grades, avg_reward, best_score, noise

    noise_annealing = 0.001
    noise_min_sigma = 0.01

    # Handle each episode w/ state
    for i_episode in range(1, number_episodes+1):
        state = agent.reset_episode()
        noise.reset(noise_annealing, noise_min_sigma)

        steps = 0
        score = 0.

        while True:
            action = agent.act(state)
            action += noise.sample()
            action = np.clip(action, -1, 1)

            next_state, reward, done = agent.task.step(action)
            agent.step(action, reward, next_state, done)

            state = next_state
            score += reward
            steps += 1
            if done:
                break

        avg_reward.append(score/max(1, steps))
        scores.append(score)

        #Keep track of best score
        if score > best_score:
            best_score = score

        if(printOut):
            text = "\r"
            text += "Episodes: {:4d}, ".format(len(scores))
            text += "Score: {:.1f}, ".format(score)
            text += "Average Score: {:.1f}, ".format(np.mean(scores[-25:]))
            text += "Best Score: {:.1f}, ".format(best_score)
            text += "Average Reward: {:.1f}, ".format(avg_reward[-1])
            text += "  "
            print(text)
            sys.stdout.flush()

train(100)
#train(100)
#train(100)
plt.figure(figsize=(15,5))
plt.plot(scores, '.', alpha=0.5, color='red')
plt.plot(np.convolve(scores, np.ones(21)/21)[(21-1)//2:-21], color='red', label='Reward')
plt.ylabel('Reward')
plt.legend(loc=2)
plt.grid(True)

plt.xlabel("Episode #")
plt.xlim(0, len(scores))
plt.savefig('rewards.png')
plt.show()
