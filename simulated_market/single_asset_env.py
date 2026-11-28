#single_asset_env.py
#Learner-agnostic daily trading environment for the RL agents (DQN and tabularq).
#
#Each episode = one ticker from a random start bar. The agent chooses a target position each bar;
#the position chosen at bar t earns the return from close t to close t+1.
#Gymnasium-style API (reset -> (obs, info), step -> (obs, reward, terminated, truncated, info))
#without requiring gymnasium as a dependency.
#
#The observation is produced by a `state_fn(ticker, t, position)` the caller supplies, so:
#  DQN:      state_fn returns DQNState's continuous vector
#  tabularq: state_fn can return TabularQState.build_from_row(frames[ticker].iloc[t]) (+ position)

import numpy as np
import pandas as pd


class SingleAssetTradingEnv:
	def __init__(
		self,
		frames: dict[str, pd.DataFrame],
		state_fn,
		reward,
		positions=(0.0, 1.0),
		episode_length: int | None = 252,
		min_start: int = 0,
		seed: int | None = None,
	):
		"""
		frames: {ticker: DataFrame with a 'close' column, indexed by date, sorted ascending}
		state_fn: callable (ticker, t, position) -> observation
		reward: object with compute(position, prev_position, next_log_return) -> float (e.g. DQNReward)
		positions: position for each discrete action (action i -> positions[i])
		episode_length: bars per episode (None = run to the end of the series)
		min_start: earliest bar an episode may start on (skip feature warm-up rows)
		"""
		self.tickers = [t for t, df in frames.items() if len(df) > min_start + 2]
		if not self.tickers:
			raise ValueError("SingleAssetTradingEnv: no ticker has enough bars")
		self.frames = frames
		self.state_fn = state_fn
		self.reward = reward
		self.positions = tuple(float(p) for p in positions)
		self.n_actions = len(self.positions)
		self.episode_length = episode_length
		self.min_start = min_start
		self.rng = np.random.default_rng(seed)

		#precompute t -> t+1 log returns per ticker (last bar has no next return)
		self.next_log_ret = {}
		for t in self.tickers:
			close = frames[t]["close"].to_numpy(dtype=np.float64)
			self.next_log_ret[t] = np.log(close[1:] / close[:-1])

		self.ticker = None
		self.t = 0
		self.start = 0
		self.position = 0.0

	def reset(self, ticker: str | None = None, start: int | None = None, seed: int | None = None):
		if seed is not None:
			self.rng = np.random.default_rng(seed)
		self.ticker = ticker if ticker is not None else self.tickers[self.rng.integers(len(self.tickers))]
		last_decision_bar = len(self.next_log_ret[self.ticker]) - 1   #need t+1 to exist
		if start is None:
			hi = last_decision_bar - (self.episode_length or 1) + 1
			start = int(self.rng.integers(self.min_start, max(self.min_start + 1, hi + 1)))
		self.start = self.t = int(start)
		self.position = 0.0
		return self.state_fn(self.ticker, self.t, self.position), self._info(0.0, 0.0)

	def step(self, action: int):
		prev = self.position
		self.position = self.positions[int(action)]
		r_next = float(self.next_log_ret[self.ticker][self.t])
		reward = self.reward.compute(self.position, prev, r_next)
		turnover = abs(self.position - prev)
		info = self._info(r_next, turnover)

		self.t += 1
		at_data_end = self.t >= len(self.next_log_ret[self.ticker])
		hit_limit = self.episode_length is not None and (self.t - self.start) >= self.episode_length
		#Markets don't "end", so running out of data or hitting the episode limit is a truncation:
		#the agent should still bootstrap from the next state. terminated is always False.
		terminated = False
		truncated = at_data_end or hit_limit
		obs = self.state_fn(self.ticker, self.t, self.position)
		return obs, reward, terminated, truncated, info

	def _info(self, next_log_return: float, turnover: float) -> dict:
		return {
			"ticker": self.ticker,
			"t": self.t,
			"date": self.frames[self.ticker].index[self.t],
			"position": self.position,
			"next_log_return": next_log_return,
			"turnover": turnover,
		}
