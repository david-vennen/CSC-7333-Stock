#test_dqn_harness.py
#Runs DQNAgent through the REAL experiment_harness._run_model_loop (same loop, allocator, Portfolio and QReward as tabular Q) on synthetic data, without touching experiment_harness.py.

import numpy as np
import pytest

pytest.importorskip("pandas_datareader") 

import experiment_harness as eh
from simulated_market.allocator import RankedAllocator
from agentic_models.dqn import DQNAgent, DQNConfig, DQNState
from agentic_models.dqn.train_dqn import add_features, split_by_date
from utilities.synthetic_market import make_universe


@pytest.fixture(scope="module")
def frames():
	full = add_features(make_universe(tickers=["AAA", "BBB", "CCC"], n_days=700, seed=3))
	cols = ["close"] + [c for c in DQNState.required_features if c != "ten_year_treasury"]
	full = {t: df[cols] for t, df in full.items()}
	return split_by_date(full, "2006-12-31", "2007-12-31")


def _cfg():
	return eh.ExperimentConfig(
		train_start="2005-01-01", train_end="2006-12-31", test_start="2007-01-01", test_end="2007-12-31",
		tickers=["AAA", "BBB", "CCC"], models_to_test=["dqn"], starting_cash=10_000.0,
	)


def test_build_from_row_matches_batch_transform(frames):
	train, _, _ = frames
	s = DQNState(include_position=False).fit(train)
	df = train["AAA"]
	batch = s.transform(df)
	for i in (0, 10, len(df) - 1):
		np.testing.assert_allclose(s.build_from_row(df.iloc[i]), batch[i], rtol=1e-6)


def test_required_features_route_through_state_builder():
	from agentic_models.state_builder import _route_features
	routing = _route_features(DQNState.required_features)   #raises if any feature has no producer
	assert "ten_year_treasury" in routing["fred"]


def test_shared_dqn_runs_in_harness_loop(frames):
	train, test, _ = frames
	cfg = _cfg()
	agent = DQNAgent.for_harness(
		train, DQNConfig(warmup_steps=200, batch_size=32, eps_decay_steps=1000, seed=0),
	)
	assert agent.actions == ("buy", "hold", "sell")
	agents = {t: agent for t in cfg.tickers}   #one network shared by every ticker

	train_port = eh._make_portfolio(cfg)
	eh._run_model_loop(cfg, agents, train, train_port, RankedAllocator(), train=True)
	n_bars = len(next(iter(train.values())))
	assert agent.env_steps == (n_bars - 1) * len(cfg.tickers)   #one transition per ticker per bar
	assert agent.grad_steps > 0
	assert agent.epsilon < 0.1                                  #decayed on DQN's own schedule

	test_port = eh._make_portfolio(cfg)
	steps_before = agent.env_steps
	eh._run_model_loop(cfg, agents, test, test_port, RankedAllocator(), train=False)
	assert agent.env_steps == steps_before                       #no learning on test data
	m = eh._metrics(test_port, cfg.starting_cash)
	assert len(m["equity_curve"]) > 0 and np.isfinite(m["final_value"])
