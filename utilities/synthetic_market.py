#synthetic_market.py
#Fake daily OHLCV with a hidden, learnable structure, for smoke-testing agents without the 310 GB dataset.
#
#Each ticker's daily drift switches between an "up" and a "down" regime (a slow Markov chain).
#Regimes persist for ~1/p_switch days, so momentum features (return_20, price_ema_ratio_*) carry real
#information about tomorrow's return. A working agent should go long in up regimes, flat in down
#regimes, and beat both a random policy and buy-and-hold. On real data, no such guarantee exists.

import numpy as np
import pandas as pd


def make_regime_ohlcv(
	n_days: int = 2500,
	drift: float = 0.003,
	vol: float = 0.01,
	p_switch: float = 0.02,
	start: str = "2005-01-03",
	seed: int = 0,
) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	regime = np.empty(n_days)
	state = 1.0
	for i in range(n_days):
		if rng.random() < p_switch:
			state = -state
		regime[i] = state
	log_ret = regime * drift + vol * rng.standard_normal(n_days)
	close = 100.0 * np.exp(np.cumsum(log_ret))
	open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(0.002 * rng.standard_normal(n_days))
	high = np.maximum(open_, close) * (1 + np.abs(0.004 * rng.standard_normal(n_days)))
	low = np.minimum(open_, close) * (1 - np.abs(0.004 * rng.standard_normal(n_days)))
	volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
	idx = pd.bdate_range(start, periods=n_days)
	return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume,
	                     "regime": regime}, index=idx)


def make_universe(tickers=("AAA", "BBB"), seed: int = 0, **kwargs) -> dict[str, pd.DataFrame]:
	return {t: make_regime_ohlcv(seed=seed + i, **kwargs) for i, t in enumerate(tickers)}
