#experiment_test_config.py

from experiment_harness import ExperimentConfig

config = ExperimentConfig(
	train_start = "2012-01-01",
	train_end = "2020-12-31",
	test_start = "2021-01-01",
	test_end = "2025-12-31",
	tickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'CAT', 'CVX', 
		'CSCO', 'PG', 'HD', 'KO', 'UNH', 'MRK', 'GS', 'AXP', 'MCD', 'IBM', 
		'VZ', 'AMGN', 'DIS', 'BA', 'CRM', 'HON', 'SHW', 'MMM', 'NKE', 'TRV',],
	models_to_test=["tabularq"],
	data_to_include = [],
	starting_cash = 10000,
	payday_frequency = "weekly",
	payday_amount = 1000.00,
	payday_anchor = "2012-01-01",
)
