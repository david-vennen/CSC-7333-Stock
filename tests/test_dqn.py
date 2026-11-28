#test_dqn.py
#Fast checks for the DQN pieces (run from repo root:  python -m pytest tests -q)

import numpy as np
import pandas as pd
import pytest
import torch

from agentic_models.dqn import DQNAgent, DQNConfig, DQNState, DQNReward, ReplayBuffer
from agentic_models.dqn.train_dqn import add_features, split_by_date, make_env, run_policy, evaluate_positions, train_dqn
from experiments.performance_metrics import portfolio_from_positions, summarize, next_simple_returns
from utilities.synthetic_market import make_universe


@pytest.fixture(scope="module")
def splits():
	frames = add_features(make_universe(tickers=["AAA", "BBB"], n_days=1500, seed=7))
	return split_by_date(frames, "2009-12-31", "2010-12-31")


def test_state_scaler_fit_on_train_only(splits):
	train, val, _ = splits
	s = DQNState().fit(train)
	x = np.concatenate([s.transform(df) for df in train.values()])
	assert np.abs(x.mean(axis=0)).max() < 0.2        #roughly standardized on train
	assert np.abs(x).max() <= s.clip
	assert "ten_year_treasury" not in s.feature_cols  #optional column skipped when absent
	s2 = DQNState.from_dict(s.to_dict())
	np.testing.assert_allclose(s2.transform(val["AAA"]), s.transform(val["AAA"]))


def test_observation_lookback_and_batch_match(splits):
	train, _, _ = splits
	s = DQNState(lookback=3).fit(train)
	f = s.transform(train["AAA"])
	allobs = s.observations_all(f, 1.0)
	assert allobs.shape == (len(f), s.state_dim)
	for t in (0, 1, 2, 50):
		np.testing.assert_allclose(allobs[t], s.observation(f, t, 1.0))
	assert s.observation(f, 0, 1.0)[-1] == 1.0


def test_reward_timing_and_cost():
	r = DQNReward(cost_bps=10, scale=1.0)
	assert r.compute(1.0, 0.0, 0.02) == pytest.approx(0.02 - 0.001)
	assert r.compute(0.0, 0.0, 0.02) == 0.0
	assert r.compute(0.0, 1.0, 0.02) == pytest.approx(-0.001)


def test_env_action_earns_next_bar_return(splits):
	train, _, _ = splits
	s = DQNState().fit(train)
	env = make_env(train, s, DQNReward(cost_bps=0, scale=1.0), episode_length=5, seed=0)
	obs, info = env.reset(ticker="AAA", start=10)
	close = train["AAA"]["close"].to_numpy()
	_, rew, term, trunc, info = env.step(1)
	assert rew == pytest.approx(np.log(close[11] / close[10]))
	for _ in range(4):
		_, _, term, trunc, _ = env.step(1)
	assert trunc and not term


def test_portfolio_equal_weight_and_costs():
	idx = pd.bdate_range("2020-01-01", periods=3)
	pos = pd.DataFrame({"A": [1, 1, 0], "B": [0, 1, 0]}, index=idx, dtype=float)
	nr = pd.DataFrame({"A": [0.01, 0.02, 0.0], "B": [0.03, -0.02, 0.0]}, index=idx)
	d = portfolio_from_positions(pos, nr, cost_bps=0)
	assert d["return"].tolist() == pytest.approx([0.01, 0.0, 0.0])
	assert d["turnover"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_double_dqn_learns_contextual_bandit():
	"""Action 1 pays +1 when x>0 else -1; action 0 pays 0. A working agent must learn sign(x)."""
	cfg = DQNConfig(gamma=0.0, warmup_steps=256, batch_size=64, lr=1e-3, eps_decay_steps=500, dropout=0.0, seed=0)
	agent = DQNAgent(2, cfg)
	rng = np.random.default_rng(0)
	for _ in range(3000):
		s = np.array([rng.normal(), 0.0], dtype=np.float32)
		a = agent.act(s)
		r = (1.0 if s[0] > 0 else -1.0) if a == 1 else 0.0
		agent.remember(s, a, r, s, True)
		agent.learn()
	test = np.stack([np.linspace(-2, 2, 41), np.zeros(41)], 1).astype(np.float32)
	pred = agent.q_values(test).argmax(1)
	assert (pred == (test[:, 0] > 0)).mean() > 0.9


def test_save_load_roundtrip(tmp_path, splits):
	train, val, _ = splits
	s = DQNState().fit(train)
	agent = DQNAgent(s.state_dim, DQNConfig(seed=3))
	p = tmp_path / "m.pt"
	agent.save(str(p), state_builder=s)
	agent2, s2, _ = DQNAgent.load(str(p))
	x = s2.observations_all(s2.transform(val["AAA"]), 0.0)
	np.testing.assert_allclose(agent.q_values(x), agent2.q_values(x), rtol=1e-6)
