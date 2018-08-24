import numpy as np


class noise:
	def __init__(self, state_size, mean, std_dev, theta):
		self.state = [mean for i in range(state_size)]
		self.mean = [mean for i in range(state_size)]
		self.std_dev = std_dev
		self.theta = theta
		self.state_size = state_size

	
	def sample_noise(self):
		self.state += self.theta * (np.array(self.mean) - np.array(self.state)) + self.std_dev * np.random.randn(self.state_size)
		return list(self.state)
	
	
	
	def reset(self):
		self.state = self.mean 
