#observers.py
"""
Concrete implementations of Experiment and Trading Observers.

These observers are model/agent agnostic — they only care about standardized metrics,
not the internal architecture of your neural networks or trading agents.

This module integrates with experiments/metrics_tracker.py for unified metric collection.
"""

import os
import json
from typing import Dict, Any, Optional
import joblib
import psutil
import platform
import torch
import numpy as np
from datetime import datetime

from .interfaces import (
    ExperimentObserver, 
    TradingObserver, 
    EarlyTrainingMetrics, 
    TradingDayState,
    NoOpObserver,
    NoOpTradingObserver,
    ExperimentResult,
    ExperimentSnapshot
)
from .metrics_tracker import (
    ModelOutputTracker,
    SoftmaxWeightTracker,
    QValueTracker,
    EnsembleTracker,
    ModelOutput
)


class SystemMonitorObserver(ExperimentObserver):
    """Monitors system state during training - model agnostic"""
    
    def __init__(self, log_file: str = "experiments/training_log.json"):
        self.log_file = log_file
        self.batch_counter = 0
        self._load_batch_counter()
        
    def _load_batch_counter(self):
        """Load batch ID from persistent file"""
        counter_file = os.path.join(os.path.dirname(self.log_file), "batch_counter.txt")
        if os.path.exists(counter_file):
            with open(counter_file, 'r') as f:
                self.batch_counter = int(f.read().strip()) + 1
        else:
            self.batch_counter = 0
            
    def _save_batch_counter(self):
        """Save batch ID to persistent file"""
        counter_file = os.path.join(os.path.dirname(self.log_file), "batch_counter.txt")
        with open(counter_file, 'w') as f:
            f.write(str(self.batch_counter))
            
    def _get_system_state(self) -> Dict[str, Any]:
        """Capture current system state"""
        return {
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_load_1min": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
            "python_version": platform.python_version(),
        }
    
    def on_experiment_start(self, snapshot: ExperimentSnapshot) -> None:
        """Log experiment start with system state"""
        
        # Update batch counter for this run
        self.batch_counter += 1
        
        snapshot.system_ram_gb = psutil.virtual_memory().available / (1024**3)
        snapshot.batch_id = self.batch_counter
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        entry = {
            "event": "experiment_start",
            **snapshot.__dict__,
            "system_state": self._get_system_state(),
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
            
    def on_epoch_complete(self, epoch: int, metrics: EarlyTrainingMetrics) -> bool:
        """Check if training should continue based on early metrics"""
        
        # Log this epoch's metrics
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        entry = {
            "event": "epoch_complete",
            "experiment_id": self.batch_counter,
            "epoch": epoch + 1,
            **metrics.__dict__,
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
            
        # Check for bad start pattern (after first few epochs)
        if epoch >= 2 and metrics.gradient_norm_mean is not None:
            if metrics.gradient_norm_mean > 10.0:  # Exploding gradients threshold
                print(f"⚠️ Pruning - High Gradient Norm: {metrics.gradient_norm_mean:.4f}")
                return False
                
        return True
        
    def on_experiment_end(self, result: ExperimentResult) -> None:
        """Log experiment completion"""
        
        entry = {
            "event": "experiment_complete",
            **result.to_dict(),
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')


class BadDayTradingObserver(TradingObserver):
    """Monitors market/system state before trading - agent agnostic"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 output_tracker: Optional[ModelOutputTracker] = None):
        self.model = None
        self.output_tracker = output_tracker or SoftmaxWeightTracker()
        self._load_model(model_path)
        
    def _load_model(self, path: Optional[str]):
        """Load classifier from disk if available"""
        if path and os.path.exists(path):
            try:
                self.model = joblib.load(path)
                print(f"Loaded BadDayClassifier from {path}")
            except Exception as e:
                print(f"Failed to load classifier: {e}, using default behavior")
                self.model = None
        else:
            print("No classifier model found, running with default risk assessment")
            
    def evaluate_day_risk(self, state: TradingDayState) -> float:
        """Return probability that today is a bad trading day"""
        
        # Build feature vector in fixed order for classifier
        features = [
            state.vix_level,
            state.hour_of_day,
            state.day_of_week,
            np.mean(state.rolling_returns_5d) if state.rolling_returns_5d else 0.01,
            max(state.rolling_returns_5d) - min(state.rolling_returns_5d) if state.rolling_returns_5d and len(state.rolling_returns_5d) > 1 else 0.02,  # Volatility proxy
            state.system_load_at_open,
            state.memory_available_gb,
            self.output_tracker.get_session_metrics()["confidence_avg"],
            self.output_tracker.get_session_metrics()["prediction_variance"],
        ]
        
        if self.model is None:
            # Default risk assessment based on simple heuristics
            return self._simple_risk_assessment(state)
            
        try:
            prediction = self.model.predict_proba([features])[0][0]  # BAD_DAY probability
            return float(prediction)
        except Exception as e:
            print(f"Classifier prediction failed: {e}, using default risk")
            return self._simple_risk_assessment(state)
            
    def _simple_risk_assessment(self, state: TradingDayState) -> float:
        """Fallback risk assessment when classifier unavailable"""
        
        risk_score = 0.5
        
        # High VIX increases risk
        if state.vix_level > 25:
            risk_score += 0.1
            
        # Low memory availability increases risk
        if state.memory_available_gb < 4.0:
            risk_score += 0.1
            
        # Unusual trading hours increase risk
        if state.hour_of_day < 9 or state.hour_of_day > 16:
            risk_score += 0.05
            
        return min(risk_score, 1.0)
        
    def should_prune_day(self, state: TradingDayState, threshold: float = 0.7) -> bool:
        """Return True if trading session should be skipped due to high risk"""
        
        risk_prob = self.evaluate_day_risk(state)
        
        if risk_prob > threshold:
            print(f"⚠️ Pruning day - Risk Probability: {risk_prob:.2%}")
            return True
            
        return False


class EnsembleObserver(ExperimentObserver):
    """Combines multiple observers for comprehensive monitoring"""
    
    def __init__(self, observers: list[ExperimentObserver] = None):
        self.observers = observers or [NoOpObserver()]
        
    def on_experiment_start(self, snapshot: ExperimentSnapshot) -> None:
        for observer in self.observers:
            observer.on_experiment_start(snapshot)
            
    def on_epoch_complete(self, epoch: int, metrics: EarlyTrainingMetrics) -> bool:
        # Continue only if ALL observers agree to continue
        return all(obs.on_epoch_complete(epoch, metrics) for obs in self.observers)
        
    def on_experiment_end(self, result: ExperimentResult) -> None:
        for observer in self.observers:
            observer.on_experiment_end(result)


class EnsembleTradingObserver(TradingObserver):
    """Combines multiple trading observers"""
    
    def __init__(self, observers: list[TradingObserver] = None):
        self.observers = observers or [NoOpTradingObserver()]
        
    def evaluate_day_risk(self, state: TradingDayState) -> float:
        # Average risk across all observers
        risks = [obs.evaluate_day_risk(state) for obs in self.observers]
        return np.mean(risks)
        
    def should_prune_day(self, state: TradingDayState, threshold: float = 0.7) -> bool:
        # Prune if ANY observer thinks it's a bad day (conservative approach)
        return any(obs.should_prune_day(state, threshold=threshold) for obs in self.observers)


__all__ = [
    'SystemMonitorObserver',
    'BadDayTradingObserver', 
    'EnsembleObserver',
    'EnsembleTradingObserver'
]
