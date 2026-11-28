#dqn_network.py
#Q-network: small MLP mapping a state vector to one Q-value per action.

import torch
import torch.nn as nn


class QNetwork(nn.Module):
	def __init__(self, state_dim: int, n_actions: int, hidden_sizes=(128, 64), dropout: float = 0.1):
		super().__init__()
		layers = []
		prev = state_dim
		for h in hidden_sizes:
			layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
			prev = h
		layers.append(nn.Linear(prev, n_actions))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
