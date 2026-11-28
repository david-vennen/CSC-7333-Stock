#dqn_replay.py
#Fixed-size ring buffer of (state, action, reward, next_state, done) transitions.

import numpy as np


class ReplayBuffer:
	def __init__(self, capacity: int, state_dim: int, seed: int | None = None):
		self.capacity = capacity
		self.states = np.zeros((capacity, state_dim), dtype=np.float32)
		self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
		self.actions = np.zeros(capacity, dtype=np.int64)
		self.rewards = np.zeros(capacity, dtype=np.float32)
		self.dones = np.zeros(capacity, dtype=np.float32)
		self.idx = 0
		self.size = 0
		self.rng = np.random.default_rng(seed)

	def push(self, state, action, reward, next_state, done):
		i = self.idx
		self.states[i] = state
		self.actions[i] = action
		self.rewards[i] = reward
		self.next_states[i] = next_state
		self.dones[i] = float(done)
		self.idx = (i + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size: int):
		ix = self.rng.integers(0, self.size, size=batch_size)
		return (
			self.states[ix],
			self.actions[ix],
			self.rewards[ix],
			self.next_states[ix],
			self.dones[ix],
		)

	def __len__(self):
		return self.size
