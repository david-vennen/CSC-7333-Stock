# experiment_harness.py
"""
Experiment dispatcher.

Usage:
	experiment = prep_experiment(config)
	result = run(experiment)

Payday scheduling lives on Portfolio (set_payday_schedule / apply_due_paydays).
This module only passes the frequency through.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentic_models.state_builder import build_state_frame
from simulated_market.allocator import RankedAllocator, attribute_rewards
from simulated_market.trading_environment import Portfolio
from agentic_models.tabular_q.tabularq_agent import TabularQAgent
from agentic_models.tabular_q.tabularq_state import TabularQState
from data_prep.stock_data_loader import StockDataLoader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
	train_start: str
	train_end: str
	test_start: str
	test_end: str
	tickers: List[str]
	models_to_test: List[str]
	data_to_include: List[str] = field(default_factory=list)
	starting_cash: float = 10_000.0
	starting_holdings: Dict[str, float] = field(default_factory=dict)
	# payday schedule handed to Portfolio
	payday_frequency: Optional[str] = None     # weekly | biweekly | twice_monthly | monthly
	payday_amount: float = 0.0
	payday_anchor: Optional[str] = None
	# tabular-Q knobs
	alpha: float = 0.1
	gamma: float = 0.99
	epsilon: float = 0.1
	epsilon_decay: float = 1.0
	epsilon_min: float = 0.01
	bar_size: str = "5min"
	seed: int = 42
	output_dir: str = "experiments/results"
	spy_ticker: str = "SPY"


@dataclass
class PreparedExperiment:
	config: ExperimentConfig
	train_frames: Dict[str, pd.DataFrame]
	test_frames: Dict[str, pd.DataFrame]
	spy_train: Optional[pd.DataFrame]
	spy_test: Optional[pd.DataFrame]
	experiment_id: str


@dataclass
class ModelResult:
	model_type: str
	final_value: float
	total_return: float
	sharpe: Optional[float]
	max_drawdown: Optional[float]
	num_trades: int
	equity_curve: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentResult:
	experiment_id: str
	config_summary: Dict[str, Any]
	model_results: List[ModelResult]
	training_time_seconds: float

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------

MODEL_STATE_CLASS = {
	"tabularq": TabularQState,
	# "dqn": DQNState,
}


def resolve_data_to_include(cfg: ExperimentConfig) -> List[str]:
	"""Union of required_features across models_to_test. Writes onto cfg."""
	needed = set(cfg.data_to_include or [])
	for model in cfg.models_to_test:
		state_cls = MODEL_STATE_CLASS.get(model)
		if state_cls is None:
			raise ValueError(f"No state class registered for model '{model}'")
		features = getattr(state_cls, "required_features", None)
		if not features:
			raise AttributeError(f"{state_cls.__name__} must define required_features")
		needed.update(features)
	cfg.data_to_include = sorted(needed)
	return cfg.data_to_include


def _load_frames(loader: StockDataLoader,cfg: ExperimentConfig, start: str, end: str) -> Dict[str, pd.DataFrame]:
	state_cls = TabularQState
	if cfg.data_to_include:
		state_cls.required_features = list(cfg.data_to_include)
	frames = {}
	for ticker in cfg.tickers:
		df = build_state_frame(
			loader, state_cls, ticker=ticker, start=start, end=end, bar_size=cfg.bar_size,
		)
		if not df.empty:
			frames[ticker] = df
	return frames


def _load_spy(loader: StockDataLoader, cfg: ExperimentConfig, start: str, end: str) -> Optional[pd.DataFrame]:
	try:
		df = build_state_frame(
			loader, TabularQState, ticker=cfg.spy_ticker,
			start=start, end=end, bar_size=cfg.bar_size,
		)
		return df if not df.empty else None
	except Exception:
		return None


def prep_experiment(config: ExperimentConfig) -> PreparedExperiment:
	
	resolve_data_to_include(config)
	loader = StockDataLoader()
	return PreparedExperiment(
		config=config,
		train_frames=_load_frames(loader, config, config.train_start, config.train_end),
		test_frames=_load_frames(loader,config, config.test_start, config.test_end),
		spy_train=_load_spy(loader, config, config.train_start, config.train_end),
		spy_test=_load_spy(loader, config, config.test_start, config.test_end),
		experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
	)


# ---------------------------------------------------------------------------
# Model switch and loop
# ---------------------------------------------------------------------------

def _make_portfolio(cfg: ExperimentConfig) -> Portfolio:
	p = Portfolio(initial_cash=cfg.starting_cash)
	p._starting_holdings = dict(cfg.starting_holdings)
	if cfg.payday_frequency and cfg.payday_amount:
		p.set_payday_schedule(cfg.payday_frequency, cfg.payday_amount, cfg.payday_anchor)
	return p


def _create_agents(cfg: ExperimentConfig, model_type: str) -> Dict[str, Any]:
	if model_type == "tabularq":
		return {
			t: TabularQAgent(
				state_builder_cfg={}, reward_cfg={},
				alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon, seed=cfg.seed,
			)
			for t in cfg.tickers
		}
	if model_type == "dqn":
		raise NotImplementedError("DQN factory not yet wired")
	raise ValueError(f"Unknown model: {model_type}")


def _align(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
	if not frames:
		return {}
	clock = max((df.index for df in frames.values()), key=len)
	aligned = {}
	for ticker, df in frames.items():
		if df.empty:
			aligned[ticker] = df
			continue
		aligned[ticker] = df.reindex(clock, method="ffill")
	return aligned

def _price_at(df: pd.DataFrame, ts)->float:
	if df.empty or "close" not in df.columns:
		return np.nan
	else:
		price =df.loc[ts,"close"]
	try:
		return float(price)
	except (TypeError, ValueError):
		return np.nan
def _run_model_loop(
	cfg: ExperimentConfig,
	agents: Dict[str, Any],
	frames: Dict[str, pd.DataFrame],
	portfolio: Portfolio,
	allocator: RankedAllocator,
	train: bool,
) -> None:
	frames = _align(frames)
	if not frames:
		return
	timestamps = next(iter(frames.values())).index
	if len(timestamps) < 2:
		return
	saved_eps = {}
	if not train:
		for t, agent in agents.items():
			if hasattr(agent, "epsilon"):
				saved_eps[t] = agent.epsilon
				agent.epsilon = 0.0
				
	last_period = None
	last_day = None
	
	for i in range(len(timestamps) - 1):
		ts, next_ts = timestamps[i], timestamps[i + 1]
		portfolio.apply_due_paydays(ts)
		
		prices = {t: _price_at(frames[t], ts) for t in frames}
		next_prices = {t: _price_at(frames[t], ts) for t in frames}
		if any(np.isnan(v) for v in list(prices.values()) + list(next_prices.values())):
			continue

		value_before = portfolio.get_current_value(prices)
		trades_before = len(portfolio.trade_history)

		states, signals, scores = {}, {}, {}
		for ticker, agent in agents.items():
			row = frames[ticker].loc[ts]
			state = agent.state.build_from_row(row) if hasattr(agent, "state") else row
			action = agent.select_action(state)
			states[ticker] = state
			signals[ticker] = action
			scores[ticker] = float(agent.get_q_value(state, action))

		allocator.allocate(signals, scores, portfolio, prices)

		reward_fn = getattr(next(iter(agents.values())), "reward", None)
		local_rewards, _ = attribute_rewards(
			portfolio=portfolio,
			prices_before=prices,
			prices_after=next_prices,
			value_before=value_before,
			agents=agents,
			signals=signals,
			results={},
			reward_fn=reward_fn,
			trades_before=trades_before,
		)

		if train and reward_fn is not None:
			for ticker, agent in agents.items():
				next_state = (
					agent.state.build_from_row(frames[ticker].loc[next_ts])
					if hasattr(agent, "state") else frames[ticker].loc[next_ts]
				)
				agent.update(
					states[ticker], signals[ticker],
					local_rewards.get(ticker, 0.0), next_state,
				)
				if hasattr(agent, "epsilon") and cfg.epsilon_decay < 1.0:
					agent.epsilon = max(cfg.epsilon_min, agent.epsilon * cfg.epsilon_decay)

		portfolio.record_equity(ts, portfolio.get_current_value(prices))

	for t, eps in saved_eps.items():
		agents[t].epsilon = eps


def _run_spy_control(spy_frame: Optional[pd.DataFrame], portfolio: Portfolio, spy_ticker: str) -> None:
	if spy_frame is None or spy_frame.empty or "close" not in spy_frame.columns:
		return
	for ts in spy_frame.index:
		portfolio.apply_due_paydays(ts)
		price = _price_at(spy_frame, ts)
		if np.isnan(price) or price <= 0:
			continue
		if portfolio.cash > 0:
			portfolio.buy(ts, spy_ticker, portfolio.cash / price, price)
		portfolio.record_equity(ts, portfolio.get_current_value({spy_ticker: price}))


# ---------------------------------------------------------------------------
# Metrics, save, run
# ---------------------------------------------------------------------------

def _metrics(portfolio: Portfolio, initial_cash: float) -> Dict[str, Any]:
	equity = portfolio.get_equity_curve()
	basis = initial_cash + portfolio.total_paydays_received()
	if equity.empty:
		return {
			"final_value": basis, "total_return": 0.0, "sharpe": None,
			"max_drawdown": None, "num_trades": len(portfolio.trade_history),
			"equity_curve": [],
		}
	values = equity["portfolio_value"].astype(float).values
	rets = pd.Series(values).pct_change().dropna()
	sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else None
	peak = np.maximum.accumulate(values)
	dd = (values - peak) / np.where(peak == 0, 1, peak)
	final = float(values[-1])
	return {
		"final_value": final,
		"total_return": (final - basis) / basis if basis else 0.0,
		"sharpe": sharpe,
		"max_drawdown": float(dd.min()) if len(dd) else None,
		"num_trades": len(portfolio.trade_history),
		"equity_curve": equity.to_dict(orient="records"),
	}


def _save_and_plot(result: ExperimentResult, output_dir: str) -> None:
	out = Path(output_dir)
	out.mkdir(parents=True, exist_ok=True)
	with (out / f"{result.experiment_id}.json").open("w") as f:
		json.dump(result.to_dict(), f, indent=2, default=str)

	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	ax = axes[0]
	for mr in result.model_results:
		if mr.equity_curve:
			df = pd.DataFrame(mr.equity_curve)
			ax.plot(pd.to_datetime(df["datetime"]), df["portfolio_value"], label=mr.model_type)
	ax.set_title("Equity Curves (models + SPY control)")
	ax.legend()
	ax.tick_params(axis="x", rotation=45)

	ax = axes[1]
	labels = [mr.model_type for mr in result.model_results]
	x = np.arange(len(labels))
	w = 0.35
	ax.bar(x - w / 2, [mr.total_return for mr in result.model_results], w, label="Total Return")
	ax.bar(x + w / 2, [mr.sharpe or 0.0 for mr in result.model_results], w, label="Sharpe")
	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=15)
	ax.set_title("Return and Sharpe vs Control")
	ax.legend()

	plt.tight_layout()
	fig.savefig(out / f"{result.experiment_id}_comparison.png", dpi=150, bbox_inches="tight")
	plt.close(fig)


def run(experiment: PreparedExperiment) -> ExperimentResult:
	cfg = experiment.config
	model_results: List[ModelResult] = []
	t0 = time.time()

	for model_type in cfg.models_to_test:
		agents = _create_agents(cfg, model_type)
		allocator = RankedAllocator()
		_run_model_loop(cfg, agents, experiment.train_frames, _make_portfolio(cfg), allocator, train=True)
		test_port = _make_portfolio(cfg)
		_run_model_loop(cfg, agents, experiment.test_frames, test_port, allocator, train=False)
		model_results.append(ModelResult(model_type=model_type, **_metrics(test_port, cfg.starting_cash)))

	spy_port = _make_portfolio(cfg)
	_run_spy_control(experiment.spy_test, spy_port, cfg.spy_ticker)
	model_results.append(ModelResult(model_type="spy_control", **_metrics(spy_port, cfg.starting_cash)))

	result = ExperimentResult(
		experiment_id=experiment.experiment_id,
		config_summary={
			"tickers": cfg.tickers,
			"models_to_test": cfg.models_to_test,
			"data_to_include": cfg.data_to_include,
			"train": [cfg.train_start, cfg.train_end],
			"test": [cfg.test_start, cfg.test_end],
			"starting_cash": cfg.starting_cash,
			"payday_frequency": cfg.payday_frequency,
			"payday_amount": cfg.payday_amount,
			"payday_anchor": cfg.payday_anchor,
		},
		model_results=model_results,
		training_time_seconds=time.time() - t0,
	)
	_save_and_plot(result, cfg.output_dir)
	return result


if __name__ == "__main__":
	config = ExperimentConfig(
		train_start="2023-01-01",
		train_end="2023-12-31",
		test_start="2024-01-01",
		test_end="2024-06-30",
		tickers=["AAPL", "MSFT"],
		models_to_test=["tabularq"],
		starting_cash=10_000.0,
		payday_frequency="twice_monthly",
		payday_amount=500.0,
		payday_anchor="2023-01-15",
	)
	experiment = prep_experiment(config)
	result = run(experiment)
	for mr in result.model_results:
		print(mr.model_type, mr.total_return, mr.sharpe)
