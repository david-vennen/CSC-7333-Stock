#metics_tracker.py
"""
Standardized tracking of model outputs during experiments.

This module is agent/model agnostic — it only requires that models 
produce predictions conforming to a standard interface (e.g., softmax weights).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import numpy as np


@dataclass
class ModelOutput:
    """Standardized representation of any model's prediction output"""
    
    raw_output: torch.Tensor  # Raw logits or probabilities from model
    action_space_size: int = None  # Number of possible actions/stocks
    
    def __post_init__(self):
        if self.action_space_size is None and isinstance(self.raw_output, torch.Tensor):
            self.action_space_size = self.raw_output.shape[-1]


class ModelOutputTracker(ABC):
    """Interface for tracking model outputs across any agent type"""
    
    @abstractmethod
    def record_prediction(self, output: ModelOutput) -> None:
        """Record a single prediction from the model"""
        pass
    
    @abstractmethod
    def get_session_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics for current session (e.g., confidence avg/variance)"""
        pass
    
    @abstractmethod
    def reset_session(self) -> None:
        """Clear all tracked data for new trading/training session"""
        pass


class SoftmaxWeightTracker(ModelOutputTracker):
    """Tracks confidence and variance from softmax-weighted predictions"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.confidences: List[float] = []
        
    def record_prediction(self, output: ModelOutput) -> None:
        """Record confidence from softmax-weighted predictions"""
        
        # Confidence = max probability across actions (higher = more certain)
        if isinstance(output.raw_output, torch.Tensor):
            probs = output.raw_output.softmax(dim=-1)
            confidence = float(torch.max(probs).item())
            
            self.confidences.append(confidence)
            
    def get_session_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics for current session"""
        
        if not self.confidences:
            return {
                "confidence_avg": 0.5,
                "prediction_variance": 0.0,
                "num_predictions": 0,
            }
            
        return {
            "confidence_avg": float(np.mean(self.confidences)),
            "prediction_variance": float(np.var(self.confidences)),
            "num_predictions": len(self.confidences),
        }
    
    def reset_session(self) -> None:
        """Clear all tracked data for new session"""
        self.confidences = []


class QValueTracker(ModelOutputTracker):
    """Tracks confidence from Q-value based agents (e.g., DQN, SARSA variants)"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.q_values: List[torch.Tensor] = []
        
    def record_prediction(self, output: ModelOutput) -> None:
        """Record Q-values from the model"""
        if isinstance(output.raw_output, torch.Tensor):
            self.q_values.append(output.raw_output.clone())
    
    def get_session_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics for current session (Q-value confidence proxy)"""
        
        if not self.q_values:
            return {
                "confidence_avg": 0.5,
                "prediction_variance": 0.0,
                "num_predictions": 0,
            }
            
        # Q-value confidence = max Q-value / (max + min) - normalized range
        q_tensor = torch.stack(self.q_values)
        max_q = float(torch.max(q_tensor).item())
        min_q = float(torch.min(q_tensor).item())
        
        if max_q != min_q:
            avg_confidence = np.mean([(q.item() - min_q) / (max_q - min_q + 1e-8) 
                                      for q in self.q_values])
        else:
            avg_confidence = 0.5
            
        return {
            "confidence_avg": float(avg_confidence),
            "prediction_variance": float(torch.var(q_tensor).item()),
            "num_predictions": len(self.q_values),
        }
    
    def reset_session(self) -> None:
        """Clear all tracked data for new session"""
        self.q_values = []


class EnsembleTracker(ModelOutputTracker):
    """Tracks outputs from ensemble of models (averages confidence across members)"""
    
    def __init__(self, num_models: int, tracker_class=SoftmaxWeightTracker):
        self.num_models = num_models
        self.trackers = [tracker_class() for _ in range(num_models)]
        
    def record_prediction(self, output: ModelOutput) -> None:
        """Record prediction from ensemble (distribute across members)"""
        # In real implementation, you'd route to appropriate member tracker
        for tracker in self.trackers:
            tracker.record_prediction(output)
    
    def get_session_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics from all model trackers"""
        
        if not self.trackers[0].confidences:  # Check first tracker as proxy
            return {
                "confidence_avg": 0.5,
                "prediction_variance": 0.0,
                "num_predictions": 0,
            }
            
        all_metrics = [t.get_session_metrics() for t in self.trackers]
        
        # Average across members
        avg_confidence = np.mean([m["confidence_avg"] for m in all_metrics])
        var_variance = np.var([m["prediction_variance"] for m in all_metrics])
        
        return {
            "confidence_avg": float(avg_confidence),
            "prediction_variance": float(var_variance),
            "num_predictions": sum(m["num_predictions"] for m in all_metrics),
        }
    
    def reset_session(self) -> None:
        """Clear all tracked data for new session"""
        for tracker in self.trackers:
            tracker.reset_session()


__all__ = [
    'ModelOutput',
    'ModelOutputTracker',
    'SoftmaxWeightTracker',
    'QValueTracker',
    'EnsembleTracker'
]
