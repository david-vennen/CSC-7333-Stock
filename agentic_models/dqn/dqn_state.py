#dqn_state.py
#Continuous counterpart to tabular_q/tabularq_state.py.
#TabularQState bins the rolling features into a discrete tuple; DQNState keeps the same
#features continuous, z-scores them with statistics fit on the TRAINING period only, and clips.

import json
import numpy as np
import pandas as pd


#Price-scale columns (macd, atr) differ wildly between a $20 and a $500 stock, so they are
#divided by close before standardizing. Everything else is already scale-free.
PRICE_SCALED_COLS = {"macd", "macd_signal", "macd_hist", "atr"}

DEFAULT_FEATURES = [
	#the four tabularq uses (kept first so the two agents share a core)
	"return_10", "vol_10", "rsi_10", "ten_year_treasury",
	#extra continuous features from RollingFeatures that DQN can use without blowing up a table
	"return_5", "return_20",
	"vol_5", "vol_20",
	"rsi_5", "rsi_20",
	"price_ema_ratio_5", "price_ema_ratio_10", "price_ema_ratio_20",
	"macd_hist", "atr",
]

#Columns that are allowed to be missing (e.g. FRED not merged yet). Missing required columns raise.
OPTIONAL_FEATURES = {"ten_year_treasury"}


class DQNState:
	"""
	Builds the DQN observation from a ticker's rolling-feature DataFrame.

	observation at bar t = [lookback stack of standardized features ending at t, current position]

	Features at row t must only use data available by the close of bar t (RollingFeatures does).
	"""

	#read by experiment_harness.resolve_data_to_include / state_builder (same contract as TabularQState)
	required_features = list(DEFAULT_FEATURES)

	def __init__(self, feature_cols=None, lookback: int = 1, clip: float = 5.0, include_position: bool = True):
		self.requested_cols = list(feature_cols or DEFAULT_FEATURES)
		self.lookback = lookback
		self.clip = clip
		self.include_position = include_position

		self.feature_cols = None   #resolved on fit (drops missing optional columns)
		self.mean = None
		self.std = None

	# ---------- fitting ----------
	def _raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
		out = pd.DataFrame(index=df.index)
		for col in self.feature_cols:
			if col in PRICE_SCALED_COLS:
				out[col] = df[col] / df["close"]
			else:
				out[col] = df[col]
		return out.astype(np.float64)

	def fit(self, train_frames):
		"""
		train_frames: a DataFrame or a list/dict of DataFrames from the TRAINING period only.
		Pooled across tickers so one shared network sees one consistent scale.
		"""
		frames = self._as_list(train_frames)
		present = set.intersection(*(set(f.columns) for f in frames))

		missing_required = [c for c in self.requested_cols if c not in present and c not in OPTIONAL_FEATURES]
		if missing_required:
			raise KeyError(f"DQNState: required feature columns missing: {missing_required}")
		self.feature_cols = [c for c in self.requested_cols if c in present]
		dropped = [c for c in self.requested_cols if c not in present]
		if dropped:
			print(f"DQNState: optional features not found, skipping: {dropped}")

		pooled = pd.concat([self._raw_features(f) for f in frames]).replace([np.inf, -np.inf], np.nan)
		self.mean = pooled.mean().to_numpy()
		self.std = pooled.std().to_numpy()
		self.std[~np.isfinite(self.std) | (self.std < 1e-12)] = 1.0
		return self

	# ---------- transforming ----------
	def transform(self, df: pd.DataFrame) -> np.ndarray:
		"""Standardized feature matrix [n_bars, n_features] (no lookback, no position). NaN -> 0 (the mean)."""
		self._check_fitted()
		x = self._raw_features(df).to_numpy()
		x = (x - self.mean) / self.std
		x = np.nan_to_num(x, nan=0.0, posinf=self.clip, neginf=-self.clip)
		return np.clip(x, -self.clip, self.clip).astype(np.float32)

	def observation(self, features: np.ndarray, t: int, position: float) -> np.ndarray:
		"""
		features: output of transform() for one ticker. t: current bar index.
		Rows before the start of the series are zero-padded.
		"""
		start = t - self.lookback + 1
		if start >= 0:
			window = features[start:t + 1]
		else:
			pad = np.zeros((-start, features.shape[1]), dtype=np.float32)
			window = np.concatenate([pad, features[:t + 1]], axis=0)
		obs = window.reshape(-1)
		if self.include_position:
			obs = np.append(obs, np.float32(position))
		return obs.astype(np.float32)

	def observations_all(self, features: np.ndarray, position: float) -> np.ndarray:
		"""observation(features, t, position) for every t at once -> [n_bars, state_dim]. Used for fast backtests."""
		n, k = features.shape
		padded = np.concatenate([np.zeros((self.lookback - 1, k), dtype=np.float32), features], axis=0)
		stack = np.concatenate([padded[i:i + n] for i in range(self.lookback)], axis=1)
		if self.include_position:
			stack = np.concatenate([stack, np.full((n, 1), position, dtype=np.float32)], axis=1)
		return stack.astype(np.float32)

	def build_from_row(self, row: pd.Series, position: float = 0.0) -> np.ndarray:
		"""
		Mirror of TabularQState.build_from_row: one bar (a Series with the feature columns + close)
		-> standardized float32 vector. Fast path for the harness, which calls this per ticker per bar.
		"""
		self._check_fitted()
		if self.lookback != 1:
			raise ValueError("build_from_row only supports lookback=1; use transform()+observation()")
		x = row.reindex(self.feature_cols).to_numpy(dtype=np.float64)
		if self._price_scaled_mask.any():
			x[self._price_scaled_mask] /= float(row["close"])
		x = (x - self.mean) / self.std
		x = np.clip(np.nan_to_num(x, nan=0.0, posinf=self.clip, neginf=-self.clip), -self.clip, self.clip)
		if self.include_position:
			x = np.append(x, position)
		return x.astype(np.float32)

	@property
	def _price_scaled_mask(self) -> np.ndarray:
		return np.array([c in PRICE_SCALED_COLS for c in self.feature_cols])

	@property
	def state_dim(self) -> int:
		self._check_fitted()
		return len(self.feature_cols) * self.lookback + (1 if self.include_position else 0)

	# ---------- persistence (scaler must travel with the model) ----------
	def to_dict(self) -> dict:
		self._check_fitted()
		return {
			"feature_cols": self.feature_cols,
			"lookback": self.lookback,
			"clip": self.clip,
			"include_position": self.include_position,
			"mean": self.mean.tolist(),
			"std": self.std.tolist(),
		}

	@classmethod
	def from_dict(cls, d: dict) -> "DQNState":
		s = cls(d["feature_cols"], d["lookback"], d["clip"], d["include_position"])
		s.feature_cols = list(d["feature_cols"])
		s.mean = np.asarray(d["mean"])
		s.std = np.asarray(d["std"])
		return s

	def save(self, path: str):
		with open(path, "w") as f:
			json.dump(self.to_dict(), f, indent=2)

	@classmethod
	def load(cls, path: str) -> "DQNState":
		with open(path) as f:
			return cls.from_dict(json.load(f))

	# ---------- helpers ----------
	def _check_fitted(self):
		if self.mean is None:
			raise RuntimeError("DQNState must be fit() on training data before use")

	@staticmethod
	def _as_list(frames):
		if isinstance(frames, pd.DataFrame):
			return [frames]
		if isinstance(frames, dict):
			return list(frames.values())
		return list(frames)
