#sate_builder.py
from __future__ import annotations
import pandas as pd

#Producers
from data_prep.stock_data_loader import StockDataLoader
from data_prep.fred_features import add_fred, PROVIDES as FRED_PROVIDES, ALIASES
from data_prep.rolling_features import RollingFeatures, PROVIDES as ROLLING_PROVIDES
from data_prep.feature_engineering import MultiTimeFrameFeatures, PROVIDES as MULTI_TF_PROVIDES

DEFAULT_BAR_SIZE = "5min"

def _route_features(required: list[str])->dict[str, list[str]]:
	"""Return {producer_name: [features it should supply]}"""
	routing = {
		"fred":[],
		"rolling":[],
		"multi_tf":[],
	}

	for feat in required:
		if feat in FRED_PROVIDES:
			routing["fred"].append(feat)
		elif feat in ROLLING_PROVIDES:
			routing["rolling"].append(feat)
		elif feat in MULTI_TF_PROVIDES:
			routing["multi_tf"].append(feat)
		else: 
			raise ValueError(f"No producer registered for feature: {feat}")
	return routing

def build_state_frame(
	loader: StockDataLoader,
	state_cls,
	ticker: str, 
	start:str | None= None,
	end: str | None = None, 
	base_path: str | None = None, 
	bar_size: str = DEFAULT_BAR_SIZE,
)-> pd.DataFrame:
	"""
	state_builder will evaluate the required features from a provided class and construct the appropriate state for the model being run. 
	Looks at state_cls.required_features,
	asks the data pipeline only for those columns, 
	returns a DataFrame the state class can consume row-by-row
	"""

	required = getattr(state_cls, "required_features", None)
	if not required:
		raise AttributeError(f"{state_cls.__name__} must define required_features")
	loader = StockDataLoader(base_path)
	raw = loader.load1min(ticker, start, end)
	if raw.empty:
		return raw

	routing = _route_features(required)
	df = raw.resample(bar_size).agg({
		"open":"first",
		"high": "max",
		"low": "min",
		"close":"last",
		"volume": "sum",
	}).dropna()

	#Fred
	if routing["fred"]:
		#map internal names to FRED ids if necessary
		series_map = {
			ALIASES.get(feat,feat) : feat
			for feat in routing["fred"]
		}
		df = add_fred(df, series_map)
		
	#Rolling features (flat path) primary bar
	if routing["rolling"]:
		roller = RollingFeatures()
		#assumes single time frame for flat path. can be extended later if necessary
		df = roller.add_rolling_features(df,tf_key=bar_size)

	#Multi_tf 
	if routing["multi_tf"]:
		mt = MultiTimeFrameFeatures(loader)
		multi = mt.create_features(ticker, start, end)
		
		# flatten needed columns onto df when you want it
		if routing["rolling"]:
			multi = RollingFeatures().process(multi)

		#align higher-timeframe columsn to the primary bar
		for tf_key, tf_df in multi.items():
			if tf_key == bar_size:
				continue
			cols = [c for c in tf_df.columns if c in required]
			if not cols:
				continue
			aligned = tf_df[cols].reindex(df.index, method="ffill")
			aligned = aligned.add_prefix(f"{tf_key}_")
			df = df.join(aligned)
	keep = list(dict.fromkeys(["close", *required]))
	missing = [f for f in required if f not in df.columns]
	if missing:
		raise ValueError(f"Missing features after build: {missing}")
	return df[keep].dropna(subset=required)
