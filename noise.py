import numpy as np


class noise:
    def __init__(self, state_size, mean, std_dev, theta, dt):
        self.state = [mean for i in range(state_size)]
        self.mean = [mean for i in range(state_size)]
        self.std_dev = std_dev
        self.theta = theta
        self.state_size = state_size
        self.dt = dt


    def sample_noise(self):
        x = self.state
        dx = self.theta * (np.array(self.mean) - np.array(x))*self.dt + self.std_dev * np.random.randn(self.state_size) * np.sqrt(self.dt)
        self.state = x + dx
        return list(self.state)


    def reset(self):
        self.state = self.mean
