#dqn_agent.py
#Double DQN agent. Same loop as the tabularq skeleton describes:
#  calculate state -> select action (epsilon greedy) -> reward -> next state -> update agent
#The only difference from tabular Q-learning is that Q(s, a) is a neural network instead of a table,
#and updates come from minibatches sampled out of a replay buffer.

import copy
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn.functional as F

from .dqn_network import QNetwork
from .dqn_replay import ReplayBuffer


@dataclass
class DQNConfig:
	"""All tunable DQN hyperparameters, so an experiment config can set them by name."""
	n_actions: int = 2                    #0 = flat, 1 = long
	hidden_sizes: tuple = (128, 64)
	dropout: float = 0.1
	lr: float = 1e-4
	weight_decay: float = 1e-5
	gamma: float = 0.99
	batch_size: int = 256
	buffer_size: int = 200_000
	warmup_steps: int = 5_000             #env steps before learning starts
	train_every: int = 1                  #gradient step every N env steps
	target_sync_every: int = 1_000        #hard update period (ignored when tau > 0)
	tau: float = 0.0                      #soft update rate; 0 = use hard updates
	eps_start: float = 1.0
	eps_end: float = 0.05
	eps_decay_steps: int = 50_000
	grad_clip: float = 10.0
	seed: int = 0
	device: str = "cpu"
	extra: dict = field(default_factory=dict)   #free-form slot for experiment metadata

	def to_dict(self) -> dict:
		d = asdict(self)
		d["hidden_sizes"] = list(self.hidden_sizes)
		return d

	@classmethod
	def from_dict(cls, d: dict) -> "DQNConfig":
		d = dict(d)
		if "hidden_sizes" in d:
			d["hidden_sizes"] = tuple(d["hidden_sizes"])
		known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
		return cls(**known)


class DQNAgent:
	def __init__(self, state_dim: int, config: DQNConfig | None = None):
		self.config = config or DQNConfig()
		c = self.config
		self.state_dim = state_dim
		self.device = torch.device(c.device)

		torch.manual_seed(c.seed)
		self.rng = np.random.default_rng(c.seed)

		self.q_net = QNetwork(state_dim, c.n_actions, c.hidden_sizes, c.dropout).to(self.device)
		self.target_net = copy.deepcopy(self.q_net).eval()
		for p in self.target_net.parameters():
			p.requires_grad_(False)

		self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
		self.buffer = ReplayBuffer(c.buffer_size, state_dim, seed=c.seed)

		self.env_steps = 0
		self.grad_steps = 0

		#harness-facing attributes (see "experiment_harness interface" below); unused by train_dqn.py
		self.actions = tuple(range(c.n_actions))
		self.state = None     #DQNState, set by for_harness()
		self.reward = None    #QReward, set by for_harness()
		self._epsilon_override = None
		self._q_cache = (None, None)

	# acting
	@property
	def epsilon(self) -> float:
		"""Linear decay over env steps, unless something (e.g. the harness at test time) set it directly."""
		if self._epsilon_override is not None:
			return self._epsilon_override
		c = self.config
		frac = min(1.0, self.env_steps / max(1, c.eps_decay_steps))
		return c.eps_start + frac * (c.eps_end - c.eps_start)

	@epsilon.setter
	def epsilon(self, value):
		"""Setting epsilon freezes it at that value; set None to go back to the decay schedule."""
		self._epsilon_override = None if value is None else float(value)

	def q_values(self, states: np.ndarray) -> np.ndarray:
		"""Q-values for a batch (or single) state, dropout off."""
		x = torch.as_tensor(np.atleast_2d(states), dtype=torch.float32, device=self.device)
		was_training = self.q_net.training
		self.q_net.eval()
		with torch.no_grad():
			q = self.q_net(x).cpu().numpy()
		self.q_net.train(was_training)
		return q

	def act(self, state: np.ndarray, greedy: bool = False) -> int:
		"""Epsilon-greedy action. greedy=True for validation/backtest."""
		if not greedy and self.rng.random() < self.epsilon:
			return int(self.rng.integers(self.config.n_actions))
		return int(self.q_values(state)[0].argmax())

	# learning
	def remember(self, state, action, reward, next_state, terminated: bool):
		"""Store one transition. Pass `terminated`, not `truncated`: an episode cut short by a time
		limit should still bootstrap from next_state."""
		self.buffer.push(state, action, reward, next_state, terminated)
		self.env_steps += 1

	def learn(self) -> float | None:
		"""One Double-DQN gradient step if warm. Returns the loss, or None if no update happened."""
		c = self.config
		if len(self.buffer) < max(c.warmup_steps, c.batch_size):
			return None
		if self.env_steps % c.train_every != 0:
			return None

		s, a, r, s2, d = self.buffer.sample(c.batch_size)
		s = torch.as_tensor(s, device=self.device)
		a = torch.as_tensor(a, device=self.device)
		r = torch.as_tensor(r, device=self.device)
		s2 = torch.as_tensor(s2, device=self.device)
		d = torch.as_tensor(d, device=self.device)

		self.q_net.train()
		q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

		with torch.no_grad():
			#Double DQN: online net picks a*, target net evaluates it (reduces overestimation)
			self.q_net.eval()
			a_star = self.q_net(s2).argmax(dim=1, keepdim=True)
			self.q_net.train()
			q_next = self.target_net(s2).gather(1, a_star).squeeze(1)
			target = r + c.gamma * (1.0 - d) * q_next

		loss = F.smooth_l1_loss(q_sa, target)   #Huber
		self.optimizer.zero_grad()
		loss.backward()
		self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), c.grad_clip))
		self.optimizer.step()
		self.grad_steps += 1

		self._update_target()
		return float(loss.item())

	def _update_target(self):
		c = self.config
		if c.tau > 0:
			with torch.no_grad():
				for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
					tp.mul_(1 - c.tau).add_(p, alpha=c.tau)
		elif self.grad_steps % c.target_sync_every == 0:
			self.target_net.load_state_dict(self.q_net.state_dict())

	# experiment_harness interface (same methods as TabularQAgent) 
	#
	# experiment_harness._run_model_loop calls, per ticker per bar:
	#     state  = agent.state.build_from_row(row)
	#     action = agent.select_action(state)          -> one of agent.actions ("buy"/"hold"/"sell")
	#     score  = agent.get_q_value(state, action)    -> used by RankedAllocator to rank tickers
	#     agent.update(state, action, reward, next_state)   (training only; reward from QReward)
	# and sets agent.epsilon = 0.0 during testing.
	#
	# Use ONE shared agent for all tickers: {t: agent for t in tickers}. Each update() call is one
	# transition, so eps_decay_steps / warmup_steps / buffer_size count transitions (bars x tickers).

	@classmethod
	def for_harness(
		cls,
		train_frames,
		config: DQNConfig | None = None,
		state_cfg: dict | None = None,
		reward_cfg: dict | None = None,
		actions=("buy", "hold", "sell"),
	) -> "DQNAgent":
		"""
		Build a DQN ready for the harness loop.
		train_frames: {ticker: DataFrame} from the TRAINING period (the feature scaler is fit on these only).
		state_cfg: DQNState kwargs (e.g. feature_cols). lookback must stay 1 for row-by-row harness use.
		reward_cfg: QReward kwargs (the reward shared with tabular Q).
		"""
		from .dqn_state import DQNState
		from ..q_reward import QReward

		state_cfg = {"include_position": False, **(state_cfg or {})}
		if state_cfg.get("lookback", 1) != 1:
			raise ValueError("the harness builds states one row at a time; use lookback=1")
		state = DQNState(**state_cfg).fit(train_frames)

		config = config or DQNConfig()
		config.n_actions = len(actions)
		agent = cls(state.state_dim, config)
		agent.actions = tuple(actions)
		agent.state = state
		agent.reward = QReward(**(reward_cfg or {}))
		return agent

	def _action_index(self, action) -> int:
		return self.actions.index(action)

	def _cached_q(self, state) -> np.ndarray:
		"""select_action then get_q_value on the same state -> one forward pass, not two."""
		cached_state, cached_q = self._q_cache
		if cached_state is state:
			return cached_q
		q = self.q_values(state)[0]
		self._q_cache = (state, q)
		return q

	def select_action(self, state):
		"""Epsilon-greedy over self.actions. Returns the action label (e.g. "buy")."""
		if self.rng.random() < self.epsilon:
			return self.actions[int(self.rng.integers(len(self.actions)))]
		return self.actions[int(self._cached_q(state).argmax())]

	def get_q_value(self, state, action) -> float:
		return float(self._cached_q(state)[self._action_index(action)])

	def update(self, state, action, reward, next_state) -> float | None:
		"""Store the transition and take a gradient step (when warm). Market steps never terminate."""
		self.remember(state, self._action_index(action), reward, next_state, terminated=False)
		self._q_cache = (None, None)   #weights may change below
		return self.learn()

	# checkpointing 
	def get_weights(self) -> dict:
		return copy.deepcopy(self.q_net.state_dict())

	def set_weights(self, weights: dict):
		self.q_net.load_state_dict(weights)
		self.target_net.load_state_dict(weights)

	def save(self, path: str, state_builder=None, extra: dict | None = None):
		"""Save weights + config (+ the fitted DQNState scaler, which the model is useless without)."""
		torch.save({
			"config": self.config.to_dict(),
			"state_dim": self.state_dim,
			"q_net": self.q_net.state_dict(),
			"target_net": self.target_net.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"env_steps": self.env_steps,
			"grad_steps": self.grad_steps,
			"state_builder": state_builder.to_dict() if state_builder is not None else None,
			"extra": extra or {},
		}, path)

	@classmethod
	def load(cls, path: str, device: str | None = None):
		"""Returns (agent, state_builder_or_None, extra)."""
		ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
		config = DQNConfig.from_dict(ckpt["config"])
		if device:
			config.device = device
		agent = cls(ckpt["state_dim"], config)
		agent.q_net.load_state_dict(ckpt["q_net"])
		agent.target_net.load_state_dict(ckpt["target_net"])
		agent.optimizer.load_state_dict(ckpt["optimizer"])
		agent.env_steps = ckpt["env_steps"]
		agent.grad_steps = ckpt["grad_steps"]

		state_builder = None
		if ckpt.get("state_builder"):
			from .dqn_state import DQNState
			state_builder = DQNState.from_dict(ckpt["state_builder"])
		return agent, state_builder, ckpt.get("extra", {})
