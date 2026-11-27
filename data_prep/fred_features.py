#fred_features.py
from __future__ import annotations

#necessary for state_builder argument parsing
PROVIDES = {
	"T5YIE",
	"DGS2",
	"DGS10",
	"USEPUINDXD",
	#used in older project. included now so I can still run that project using this new code. DJV
	"ten_year_treasury",
	"two_year_treasury",
	"inflation_expectation",
	"economic_uncertainty",
}
ALIASES = {
	"ten_year_treasury":"DGS10",
	"two_year_treasury":"DGS2",
	"inflation_expectation": "T5YIE",
	"economic_uncertainty": "USEPUINDXD",
}

"""FRED macro features for daily price data.

Fetch FRED series with pandas_datareader, lag them by ``shift`` observations
so a row never sees data published after it (to avoid look-ahead), and
align them to an existing DataFrame's DatetimeIndex with forward fill.

Example:
	from data_prep.fred_features import add_fred
	df = add_fred(df)                                  # default series
	df = add_fred(df, ["DGS10", "DGS2"])               # columns named by series id
	df = add_fred(df, {"DGS10": "ust_10y"}, shift=2)   # custom column names
"""



from typing import Iterable, Mapping

import pandas as pd
import pandas_datareader.data as web

# Series used in the original DataPreprocessApril19.py.
DEFAULT_SERIES: dict[str, str] = {
	"T5YIE": "T5YIE",            # 5-year breakeven inflation rate
	"DGS2": "DGS2",              # 2-year Treasury constant maturity
	"DGS10": "DGS10",            # 10-year Treasury constant maturity
	"USEPUINDXD": "USEPUINDXD",  # US economic policy uncertainty index (daily)
}


def _normalize_series(series: Iterable[str] | Mapping[str, str] | None) -> dict[str, str]:
	"""Return a {fred_id: column_name} mapping."""
	if series is None:
		return dict(DEFAULT_SERIES)
	if isinstance(series, Mapping):
		return dict(series)
	if isinstance(series, str):
		return {series: series}
	return {s: s for s in series}


def fetch_fred(
	series: Iterable[str] | Mapping[str, str] | None = None,
	start=None,
	end=None,
) -> pd.DataFrame:
	"""Download FRED series and return them with renamed columns, indexed by date."""
	mapping = _normalize_series(series)
	raw = web.DataReader(list(mapping), "fred", start, end)
	raw.index = pd.to_datetime(raw.index)
	return raw.rename(columns=mapping).sort_index()


def align_fred(
	fred: pd.DataFrame,
	index: pd.DatetimeIndex,
	shift: int = 1,
) -> pd.DataFrame:
	"""Lag FRED data by ``shift`` of its own observations, then forward fill onto ``index``.

	The shift happens on FRED's native dates (before reindexing), so "1 period"
	means one FRED observation, not one row of your DataFrame.
	"""
	lagged = fred.sort_index().shift(shift)
	target = pd.DatetimeIndex(index)
	tz = target.tz
	if tz is not None:
		target = target.tz_localize(None)
	combined = lagged.reindex(lagged.index.union(target)).ffill()
	out = combined.reindex(target)
	if tz is not None:
		out.index = out.index.tz_localize(tz)
	return out


def add_fred(
	df: pd.DataFrame,
	series: list[str] | dict[str, str] | None = None,
	shift: int = 1,
	lookback_days: int = 30,
) -> pd.DataFrame:
	"""Return a copy of ``df`` with lagged, forward-filled FRED columns added.

	Args:
		df: DataFrame with a DatetimeIndex (e.g. daily stock prices).
		series: FRED ids as a list, or a {fred_id: column_name} dict.
			Defaults to DEFAULT_SERIES.
		shift: Number of FRED observations to lag (1 avoids look-ahead).
		lookback_days: Extra history fetched before df's first date so the
			shift and forward fill have values to start from.
	"""
	if not isinstance(df.index, pd.DatetimeIndex):
		raise TypeError("add_fred expects df to have a DatetimeIndex; set it with df.set_index(...) first.")
	if df.empty:
		return df.copy()

	idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
	start = idx.min() - pd.Timedelta(days=lookback_days)
	end = idx.max()

	fred = fetch_fred(series, start, end)
	aligned = align_fred(fred, df.index, shift=shift)

	clashes = set(aligned.columns) & set(df.columns)
	if clashes:
		raise ValueError(f"FRED columns already exist in df: {sorted(clashes)}")
	return df.join(aligned)


if __name__ == "__main__":
	dates = pd.bdate_range("2024-01-01", "2024-03-31")
	demo = pd.DataFrame({"close": range(len(dates))}, index=dates)
	print(add_fred(demo).head(10))
