#interfaces.py

"""
Standardized interfaces for experiment and trading monitoring.

This module defines the contracts that both legacy code and new plug-and-play 
models can implement. It ensures monitoring systems remain agnostic to model/agent architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import torch
from datetime import datetime


@dataclass
class ExperimentSnapshot:
	"""Standardized snapshot of experiment start state (model-agnostic)"""
    
	timestamp: str = None  # ISO format datetime
	seed: int = 0
	batch_id: int = 0
	system_ram_gb: float = 0.0
	system_cpu_temp_c: Optional[float] = None
    
	def __post_init__(self):
		if self.timestamp is None:
			self.timestamp = datetime.now().isoformat()


@dataclass
class EarlyTrainingMetrics:
	"""Features for bad start prediction (model-agnostic)"""
    
	loss_epoch_1: float  # Loss after first epoch
	gradient_norm_mean: Optional[float] = None  # Mean gradient norm across layers
	weight_variance: float = 0.0  # Variance of all model weights
	learning_rate: float = 0.0005
    
	def to_list(self) -> List[float]:
		"""Convert to list for classifier input (handles Optional fields)"""
		return [
			self.loss_epoch_1,
			self.gradient_norm_mean or 0.0,
			self.weight_variance,
			self.learning_rate,
		]


@dataclass
class TradingDayState:
	"""Features for bad day prediction (agent-agnostic)"""
    
	vix_level: float = 20.0  # CBOE VIX index value

	# predicted VIX from other stock market project
	# TODO: add predicted Vix based on other project's neural net output
	#predicted_volatility_forecast: Optional[float] = None
	
	hour_of_day: int = 9  # Market open hour (0-23)
	day_of_week: int = 1  # Day of week (0=Monday, 6=Sunday)
    
	rolling_returns_5d: List[float] = None  # Last 5 days returns
	rolling_volatility_5d: float = 0.02  # Rolling volatility estimate
    
	system_load_at_open: float = 1.0  # OS load average at market open
	memory_available_gb: float = 8.0  # RAM available in GB
    
	model_confidence_avg: float = 0.5  # Average confidence from last session
	model_prediction_variance: float = 0.0  # Variance of predictions
    
	def __post_init__(self):
		if self.rolling_returns_5d is None:
			self.rolling_returns_5d = [0.01] * 5


@dataclass
class ExperimentResult:
	"""Standardized result from completed experiment"""
    
	experiment_id: int
	seed: int
	final_loss: float
	validation_sharpe: Optional[float] = None
	max_drawdown: Optional[float] = None
	training_time_seconds: float = 0.0
    
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON logging"""
		return asdict(self)


class ExperimentObserver(ABC):
	"""Interface for experiment monitoring - model/agent agnostic"""
    
	@abstractmethod
	def on_experiment_start(self, snapshot: ExperimentSnapshot) -> None:
		"""Called at the start of training"""
		pass
    
	@abstractmethod
	def on_epoch_complete(self, epoch: int, metrics: EarlyTrainingMetrics) -> bool:
		"""
		Called after each epoch completes.
        
		Returns True if training should continue (False = prune/early stop).
		"""
		pass
    
	@abstractmethod
	def on_experiment_end(self, result: ExperimentResult) -> None:
		"""Called when experiment completes"""
		pass


class TradingObserver(ABC):
	"""Interface for trading day monitoring - agent agnostic"""
    
	@abstractmethod
	def evaluate_day_risk(self, state: TradingDayState) -> float:
		"""
		Evaluate probability that today is a bad trading day.
        
		Returns float in range [0.0, 1.0] where higher = worse day expected.
		"""
		pass
    
	@abstractmethod
	def should_prune_day(self, state: TradingDayState, threshold: float = 0.7) -> bool:
		"""
		Return True if trading session should be skipped due to high risk.
        
		Args:
			state: Current market/system state
			threshold: Risk probability threshold (default 0.7 = prune if >70% bad day risk)
            
		Returns:
			True if trading should be pruned/skipped, False otherwise
		"""
		pass


class NoOpObserver(ExperimentObserver):
	"""No-op observer for legacy code compatibility"""
    
	def on_experiment_start(self, snapshot: ExperimentSnapshot) -> None:
		pass
    
	def on_epoch_complete(self, epoch: int, metrics: EarlyTrainingMetrics) -> bool:
		return True  # Always continue training
    
	def on_experiment_end(self, result: ExperimentResult) -> None:
		pass


class NoOpTradingObserver(TradingObserver):
	"""No-op observer for legacy code compatibility"""
    
	def evaluate_day_risk(self, state: TradingDayState) -> float:
		return 0.5  # Neutral risk
    
	def should_prune_day(self, state: TradingDayState, threshold: float = 0.7) -> bool:
		return False  # Never prune by default


__all__ = [
	'ExperimentSnapshot',
	'EarlyTrainingMetrics', 
	'TradingDayState',
	'ExperimentResult',
	'ExperimentObserver',
	'TradingObserver',
	'NoOpObserver',
	'NoOpTradingObserver'
]
