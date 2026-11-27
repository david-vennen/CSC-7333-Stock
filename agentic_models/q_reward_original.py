#tabularq_reward.py
#This file wil contain the code necessary for calculating the reward for a given state for the tabularq model

import numpy as np

class QReward:
	"""
	Immediate reward :
		mark-to-market return
		+bonus on realized profits in PnL
		- log-scalled penalty on losses
		- small penalty on turnover cost (commission fees, bid-ask-spread slippage, etc.)
	"""
	def __init__(
		self,
		realized_bonus : float = .5,
		loss_penalty_scale: float = 1.0,
		cost_penalty: float = .05,
		clip: float = .15,
	):
		"""
		realized_bonus: weight on realized PnL /value_before. 
		loss_penalty_scale: multiplier on the log loss penalty
		cost_penalty: penalty per unit turnover
		clip: absolute clip on final reward.  clips reward/penalty preventing single reward from dominating
		"""

		self.realized_bonus = realized_bonus
		self.loss_penalty_scale = loss_penalty_scale
		self.cost_penalty = cost_penalty
		self.clip = clip

	def compute(
		self,
		value_before: float,
		value_after: float,
		realized_pnl: float,
		turnover: float = 0.0,      #turnover is fraction of portofoio bought or sold during step. 
	) -> float :
		if value_before <=0:
			return 0.0
		#mark-to-market and realized_return equations	
		mtm_return = (value_after - value_before)/value_before
		realized_return = realized_pnl / value_before

		#log-scaled penalty applied only to the negative portion of mtm_return
		#small losses -> small extra penalty; large losses -> much harsher penalty
		loss = min(mtm_return, 0.0)
		log_penalty = self.loss_penalty_scale *np.log1p(-loss)       #log(1 + |loss|)
		reward = (
		mtm_return
		+ self.realized_bonus *realized_return
		-log_penalty
		-self.cost_penalty *turnover
		)
		return float(np.clip(reward, -self.clip, self.clip))