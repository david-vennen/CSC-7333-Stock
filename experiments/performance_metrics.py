#performance_metrics.py
#Agent-agnostic backtest metrics: cumulative return, Sharpe, max drawdown, turnover.
#Any agent that can produce a {date x ticker} table of target positions can be scored here,
#so DQN, tabularq and the buy-and-hold benchmark are all measured the same way.

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_from_positions(
	positions: pd.DataFrame,
	next_returns: pd.DataFrame,
	cost_bps: float = 5.0,
) -> pd.DataFrame:
	"""
	positions: {date x ticker} target position chosen at the close of `date` (0 = flat, 1 = long)
	next_returns: {date x ticker} SIMPLE return from close `date` to the next close
	Capital is split equally across the tickers marked long; the rest sits in cash (0 return).
	Rebalanced to equal weight daily; cost_bps is charged per side on the change in weights.

	Returns a DataFrame indexed by date with columns: return, turnover, n_long, equity.
	"""
	positions = positions.reindex_like(next_returns).fillna(0.0)
	#a ticker with no return on a date (not trading / not listed yet) can't be held that day
	positions = positions.where(next_returns.notna(), 0.0)
	next_returns = next_returns.fillna(0.0)

	n_long = positions.sum(axis=1)
	weights = positions.div(n_long.where(n_long > 0, 1.0), axis=0)
	turnover = weights.diff().abs().sum(axis=1)
	turnover.iloc[0] = weights.iloc[0].abs().sum()   #initial entry is a trade too

	gross = (weights * next_returns).sum(axis=1)
	net = gross - cost_bps / 1e4 * turnover
	return pd.DataFrame({
		"return": net,
		"turnover": turnover,
		"n_long": n_long,
		"equity": (1.0 + net).cumprod(),
	})


def summarize(daily_returns: pd.Series, turnover: pd.Series | None = None) -> dict:
	r = pd.Series(daily_returns).dropna()
	if r.empty:
		return {"cumulative_return": 0.0, "annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
		        "avg_daily_turnover": 0.0, "n_days": 0}
	equity = (1.0 + r).cumprod()
	cum = float(equity.iloc[-1] - 1.0)
	years = len(r) / TRADING_DAYS
	ann = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else -1.0
	std = r.std()
	sharpe = float(np.sqrt(TRADING_DAYS) * r.mean() / std) if std > 0 else 0.0
	drawdown = equity / equity.cummax() - 1.0
	return {
		"cumulative_return": cum,
		"annual_return": ann,
		"sharpe": sharpe,
		"max_drawdown": float(drawdown.min()),
		"avg_daily_turnover": float(turnover.mean()) if turnover is not None else 0.0,
		"n_days": int(len(r)),
	}


def next_simple_returns(close: pd.Series) -> pd.Series:
	"""Return from close `date` to the next close, indexed by `date` (same convention as positions)."""
	return close.shift(-1) / close - 1.0


def buy_and_hold(close: pd.Series, cost_bps: float = 5.0) -> pd.DataFrame:
	"""Buy-and-hold one asset (e.g. SPY) on the same date convention and cost model as the agents."""
	nr = next_simple_returns(close).dropna().to_frame("asset")
	return portfolio_from_positions(pd.DataFrame(1.0, index=nr.index, columns=["asset"]), nr, cost_bps)
