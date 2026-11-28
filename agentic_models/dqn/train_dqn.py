#train_dqn.py
#Training / evaluation entry point for the DQN agent.
#
#Library use (for the experiment protocol):
#    frames = add_features(load_daily_frames(tickers, start, end))
#    train, val, test = split_by_date(frames, train_end="2017-12-31", val_end="2020-12-31")
#    agent, state, history = train_dqn(train, val, DQNConfig(seed=0), total_steps=200_000)
#    results = evaluate_agent(agent, state, test, benchmark_close=spy_close)
#
#Command line (run from the repo root):
#    python -m agentic_models.dqn.train_dqn --synthetic --steps 30000          #smoke test, no data needed
#    python -m agentic_models.dqn.train_dqn --tickers AAPL MSFT --train-start 2005-01-01 \
#        --train-end 2017-12-31 --val-end 2020-12-31 --test-end 2026-03-31 --seeds 5

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .dqn_agent import DQNAgent, DQNConfig
from .dqn_reward import DQNReward
from .dqn_state import DQNState, DEFAULT_FEATURES, OPTIONAL_FEATURES
from simulated_market.single_asset_env import SingleAssetTradingEnv
from experiments.performance_metrics import portfolio_from_positions, summarize, next_simple_returns


# ============================== data ==============================
def load_daily_frames(tickers, start, end, data_path=None) -> dict[str, pd.DataFrame]:
	"""1-minute finnhub CSVs -> daily OHLCV bars, one DataFrame per ticker."""
	from data_prep.stock_data_loader import StockDataLoader
	from data_prep.feature_engineering import MultiTimeFrameFeatures
	loader = StockDataLoader(base_path=data_path)
	mtf = MultiTimeFrameFeatures(loader)
	frames = {}
	for t in tickers:
		raw = loader.load1min(t, start=start, end=end)
		if raw.empty:
			continue
		daily = mtf.resample_ohlcv(raw, "1D")
		daily.index = daily.index.tz_localize(None).normalize()
		frames[t.upper()] = daily
	return frames


def add_features(daily_frames: dict[str, pd.DataFrame], extra: dict[str, pd.DataFrame] | None = None):
	"""
	Add RollingFeatures('1day') to each ticker and drop warm-up rows.
	extra: optional {name: DataFrame indexed by date} of macro columns (e.g. FRED, ALREADY lagged by its
	publication delay) to as-of join onto every ticker.
	"""
	from data_prep.rolling_features import RollingFeatures
	rf = RollingFeatures()
	out = {}
	for t, df in daily_frames.items():
		f = rf.add_rolling_features(df, "1day")
		if extra:
			for macro in extra.values():
				f = pd.merge_asof(f.sort_index(), macro.sort_index(), left_index=True, right_index=True)
		required = [c for c in DEFAULT_FEATURES if c not in OPTIONAL_FEATURES and c in f.columns]
		out[t] = f.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
	return out


def split_by_date(frames, train_end, val_end, test_end=None):
	"""Chronological split. train: <= train_end, val: (train_end, val_end], test: (val_end, test_end]."""
	train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
	test_end = pd.Timestamp(test_end) if test_end else None
	train, val, test = {}, {}, {}
	for t, df in frames.items():
		train[t] = df[df.index <= train_end]
		val[t] = df[(df.index > train_end) & (df.index <= val_end)]
		te = df[df.index > val_end]
		test[t] = te[te.index <= test_end] if test_end is not None else te
	drop_empty = lambda d: {k: v for k, v in d.items() if len(v) > 2}
	return drop_empty(train), drop_empty(val), drop_empty(test)


# ============================== evaluation ==============================
def run_policy(agent: DQNAgent, state: DQNState, frames, positions=(0.0, 1.0)):
	"""
	Greedy backtest of the agent on every ticker over its whole series.
	Returns (positions_df, next_returns_df), both {date x ticker}.

	Since the position is the only part of the state that depends on past actions, Q-values are
	computed in one batch for each possible position, then the policy is walked forward in time.
	"""
	pos_cols, ret_cols = {}, {}
	for t, df in frames.items():
		feats = state.transform(df)
		q_by_pos = [agent.q_values(state.observations_all(feats, p)) for p in positions]
		pos = np.zeros(len(df))
		current = 0.0
		for i in range(len(df) - 1):           #last bar has no next return -> stays flat
			q = q_by_pos[positions.index(current)][i]
			current = positions[int(q.argmax())]
			pos[i] = current
		pos_cols[t] = pd.Series(pos, index=df.index)
		ret_cols[t] = next_simple_returns(df["close"])
	return pd.DataFrame(pos_cols), pd.DataFrame(ret_cols)


def evaluate_positions(positions_df, next_returns_df, cost_bps):
	daily = portfolio_from_positions(positions_df, next_returns_df, cost_bps)
	daily = daily.iloc[:-1]   #final date has no next return for anyone
	m = summarize(daily["return"], daily["turnover"])
	m["pct_days_invested"] = float((daily["n_long"] > 0).mean())
	return m, daily


def evaluate_agent(agent, state, frames, cost_bps=5.0, benchmark_close: pd.Series | None = None, seed=0):
	"""
	Agent vs. benchmark vs. random policy on the same dates and cost model.
	benchmark_close: e.g. SPY close. If None, benchmark = equal-weight buy-and-hold of `frames`.
	"""
	pos, nret = run_policy(agent, state, frames)
	results = {}
	results["agent"], daily = evaluate_positions(pos, nret, cost_bps)

	if benchmark_close is not None:
		bc = benchmark_close.loc[(benchmark_close.index >= pos.index.min()) & (benchmark_close.index <= pos.index.max())]
		nb = next_simple_returns(bc).to_frame("BENCH")
		results["buy_and_hold"], _ = evaluate_positions(pd.DataFrame(1.0, index=nb.index, columns=["BENCH"]), nb, cost_bps)
	else:
		results["buy_and_hold_equal_weight"], _ = evaluate_positions(pd.DataFrame(1.0, index=nret.index, columns=nret.columns), nret, cost_bps)

	rng = np.random.default_rng(seed)
	rand = pd.DataFrame(rng.integers(0, 2, pos.shape).astype(float), index=pos.index, columns=pos.columns)
	results["random"], _ = evaluate_positions(rand, nret, cost_bps)
	return {"metrics": results, "daily": daily, "positions": pos}


# ============================== training ==============================
def make_env(frames, state: DQNState, reward: DQNReward, episode_length=252, seed=0):
	feats = {t: state.transform(df) for t, df in frames.items()}
	state_fn = lambda ticker, t, position: state.observation(feats[ticker], t, position)
	return SingleAssetTradingEnv(frames, state_fn, reward, episode_length=episode_length,
	                             min_start=state.lookback - 1, seed=seed)


def train_dqn(
	train_frames,
	val_frames,
	config: DQNConfig | None = None,
	state: DQNState | None = None,
	reward: DQNReward | None = None,
	total_steps: int = 200_000,
	eval_every: int = 10_000,
	episode_length: int = 252,
	observer=None,
	log=lambda *a: print(*a, flush=True),
):
	"""
	Train on train_frames; every eval_every steps run a greedy backtest on val_frames and keep the
	weights with the best validation Sharpe. The test set is never touched here.

	observer: optional old_experiments.ExperimentObserver (on_experiment_start / on_epoch_complete /
	          on_experiment_end). An "epoch" here = one eval period. Returning False stops training.
	Returns (agent, state, history_df).
	"""
	config = config or DQNConfig()
	reward = reward or DQNReward()
	state = (state or DQNState()).fit(train_frames)   #scaler fit on TRAINING data only

	env = make_env(train_frames, state, reward, episode_length, seed=config.seed)
	agent = DQNAgent(state.state_dim, config)

	if observer is not None:
		from old_experiments.interfaces import ExperimentSnapshot
		observer.on_experiment_start(ExperimentSnapshot(seed=config.seed))

	t0 = time.time()
	history, losses, ep_rewards = [], [], []
	best = {"sharpe": -np.inf, "weights": agent.get_weights(), "step": 0}
	obs, _ = env.reset()
	ep_r = 0.0

	for step in range(1, total_steps + 1):
		action = agent.act(obs)
		next_obs, r, terminated, truncated, _ = env.step(action)
		agent.remember(obs, action, r, next_obs, terminated)
		loss = agent.learn()
		if loss is not None:
			losses.append(loss)
		ep_r += r
		obs = next_obs
		if terminated or truncated:
			ep_rewards.append(ep_r)
			ep_r = 0.0
			obs, _ = env.reset()

		if step % eval_every == 0 or step == total_steps:
			val_m, _ = evaluate_positions(*run_policy(agent, state, val_frames), reward.cost_bps)
			row = {
				"step": step,
				"epsilon": agent.epsilon,
				"loss": float(np.mean(losses)) if losses else np.nan,
				"train_ep_reward": float(np.mean(ep_rewards)) if ep_rewards else np.nan,
				"val_sharpe": val_m["sharpe"],
				"val_cum_return": val_m["cumulative_return"],
				"val_turnover": val_m["avg_daily_turnover"],
				"val_pct_invested": val_m["pct_days_invested"],
			}
			history.append(row)
			log(f"step {step:>7} | eps {row['epsilon']:.2f} | loss {row['loss']:.4f} | "
			    f"ep_reward {row['train_ep_reward']:+.2f} | val sharpe {row['val_sharpe']:+.2f} "
			    f"cum {row['val_cum_return']:+.1%} invested {row['val_pct_invested']:.0%}")
			#only checkpoint once learning has started (pre-warmup weights are random)
			if loss is not None or losses:
				if val_m["sharpe"] > best["sharpe"]:
					best = {"sharpe": val_m["sharpe"], "weights": agent.get_weights(), "step": step}
			losses, ep_rewards = [], []

			if observer is not None:
				from old_experiments.interfaces import EarlyTrainingMetrics
				keep_going = observer.on_epoch_complete(len(history) - 1, EarlyTrainingMetrics(
					loss_epoch_1=row["loss"] if np.isfinite(row["loss"]) else 0.0,
					gradient_norm_mean=getattr(agent, "last_grad_norm", None),
					learning_rate=config.lr,
				))
				if not keep_going:
					log("observer requested early stop")
					break

	agent.set_weights(best["weights"])
	log(f"restored best weights from step {best['step']} (val sharpe {best['sharpe']:+.2f})")
	history = pd.DataFrame(history)

	if observer is not None:
		from old_experiments.interfaces import ExperimentResult
		last = history.iloc[-1]
		observer.on_experiment_end(ExperimentResult(
			experiment_id=config.seed, seed=config.seed, final_loss=float(last["loss"]),
			validation_sharpe=float(best["sharpe"]), training_time_seconds=time.time() - t0,
		))
	return agent, state, history


# ============================== CLI ==============================
def _parse_args(argv=None):
	p = argparse.ArgumentParser(description="Train / evaluate the DQN trading agent")
	p.add_argument("--synthetic", action="store_true", help="use synthetic regime data (no dataset needed)")
	p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
	p.add_argument("--benchmark", default="SPY")
	p.add_argument("--data-path", default=None)
	p.add_argument("--train-start", default="2005-01-01")
	p.add_argument("--train-end", default="2017-12-31")
	p.add_argument("--val-end", default="2020-12-31")
	p.add_argument("--test-end", default="2026-03-31")
	p.add_argument("--run-test", action="store_true", help="evaluate on the TEST period (do this once, at the end)")
	p.add_argument("--steps", type=int, default=200_000)
	p.add_argument("--eval-every", type=int, default=10_000)
	p.add_argument("--seeds", type=int, default=1, help="number of seeds (0..N-1)")
	p.add_argument("--seed-offset", type=int, default=0)
	p.add_argument("--cost-bps", type=float, default=5.0)
	p.add_argument("--lookback", type=int, default=1)
	p.add_argument("--config", default=None, help="JSON file of DQNConfig overrides")
	p.add_argument("--out", default="results/dqn")
	return p.parse_args(argv)


def main(argv=None):
	args = _parse_args(argv)
	os.makedirs(args.out, exist_ok=True)

	benchmark_close = None
	if args.synthetic:
		from utilities.synthetic_market import make_universe
		daily = make_universe(tickers=[t.upper() for t in args.tickers], n_days=3000, seed=1000)
		args.train_end, args.val_end, args.test_end = "2013-12-31", "2015-12-31", None
	else:
		daily = load_daily_frames(args.tickers + [args.benchmark], args.train_start, args.test_end, args.data_path)
		bench = daily.pop(args.benchmark.upper(), None)
		if bench is not None:
			benchmark_close = bench["close"]
		else:
			print(f"warning: benchmark {args.benchmark} not loaded; using equal-weight buy-and-hold")
	frames = add_features(daily)
	train, val, test = split_by_date(frames, args.train_end, args.val_end, args.test_end)
	print(f"tickers: {list(train)} | train bars {sum(map(len, train.values()))} | "
	      f"val bars {sum(map(len, val.values()))} | test bars {sum(map(len, test.values()))}")

	overrides = {}
	if args.config:
		with open(args.config) as f:
			overrides = json.load(f)

	all_results = []
	for seed in range(args.seed_offset, args.seed_offset + args.seeds):
		print(f"\n===== seed {seed} =====")
		cfg = DQNConfig.from_dict({**overrides, "seed": seed})
		agent, state, history = train_dqn(
			train, val, cfg, DQNState(lookback=args.lookback), DQNReward(cost_bps=args.cost_bps),
			total_steps=args.steps, eval_every=args.eval_every,
		)
		history.to_csv(os.path.join(args.out, f"history_seed{seed}.csv"), index=False)
		agent.save(os.path.join(args.out, f"dqn_seed{seed}.pt"), state_builder=state, extra=vars(args))

		split_name, split = ("test", test) if args.run_test else ("val", val)
		res = evaluate_agent(agent, state, split, args.cost_bps, benchmark_close, seed=seed)
		res["daily"].to_csv(os.path.join(args.out, f"{split_name}_daily_seed{seed}.csv"))
		row = {"seed": seed, "split": split_name}
		for who, m in res["metrics"].items():
			for k, v in m.items():
				row[f"{who}_{k}"] = v
		all_results.append(row)
		print(json.dumps(res["metrics"], indent=2))

	summary = pd.DataFrame(all_results)
	summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
	if len(summary) > 1:
		cols = [c for c in summary.columns if c.endswith(("_sharpe", "_cumulative_return", "_max_drawdown"))]
		print("\nmean ± std over seeds:")
		for c in cols:
			print(f"  {c:<45} {summary[c].mean():+.3f} ± {summary[c].std():.3f}")
	return summary


if __name__ == "__main__":
	main()
