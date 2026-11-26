#tabularq_agent.py
#this is where we create the agent.
#we will need to import tabularq_state.py and tabularq_reward.py
#this file should implement the bellman equation and our selection policy (greedy, epsilon greedy,etc)

import random
import numpy as np
from .tabularq_state import TabularQState
from .tabularq_reward import TabularQReward

class TabularQAgent:
	def __init__(self, state_builder_cfg, reward_cfg, alpha, gamma, epsilon, actions=("buy", "hold", "sell"), q_init=0.0, seed =42):
		self.state = TabularQState(**(state_builder_cfg or {}))
		self.reward = TabularQReward(**reward_cfg)
		self.q_table = {}
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.actions = actions
		self.q_init = q_init
		random.seed(seed)
		np.random.seed(seed)

	def _get_q(self, state):
		if state not in self.q_table:
			self.q_table[state] = {a: self.q_init for a in self.actions}
		return self.q_table[state]
		
	def select_action(self, state):
		if random.random() < self.epsilon:
			return random.choice(self.actions)
		q_values = self._get_q(state)
		return max(q_values, key=q_values.get)

	def update(self, state, action, reward, next_state):
		q_sa = self._get_q(state)[action]
		max_next = max(self._get_q(next_state).values())
		target = reward + self.gamma * max_next
		self.q_table[state][action] = q_sa + self.alpha * (target - q_sa)

	def get_q_value(self, state, action):
		return self._get_q(state).get(action, self.q_init)

	