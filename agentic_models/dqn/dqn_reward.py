#dqn_reward.py
#Reward for the DQN agent. Kept deliberately simple (compare: tabular_q/tabularq_reward.py, which
#adds realized-profit bonuses and log loss penalties). Swapping in TabularQReward is possible later
#if the group wants the two agents to share a reward; this one is the textbook baseline.

import numpy as np


class DQNReward:
	"""
	reward_t = position_t * log(close_{t+1}/close_t) - cost_bps/1e4 * |position_t - position_{t-1}|

	The action chosen at bar t sets position_t, which then earns the t -> t+1 return.
	"""

	def __init__(self, cost_bps: float = 5.0, scale: float = 100.0):
		"""
		cost_bps: transaction cost per side, in basis points (5 bps = 0.05%)
		scale: multiply rewards so daily log returns (~0.01) are O(1) for the Q-network. Pure rescale;
		       does not change the optimal policy.
		"""
		self.cost_bps = cost_bps
		self.scale = scale

	def compute(self, position: float, prev_position: float, next_log_return: float) -> float:
		cost = self.cost_bps / 1e4 * abs(position - prev_position)
		return float(self.scale * (position * next_log_return - cost))

	def compute_vectorized(self, positions: np.ndarray, prev_positions: np.ndarray, next_log_returns: np.ndarray) -> np.ndarray:
		cost = self.cost_bps / 1e4 * np.abs(positions - prev_positions)
		return self.scale * (positions * next_log_returns - cost)
