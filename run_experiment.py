# run_experiment.py
import importlib
from experiment_harness import prep_experiment, run

def main(config_module: str):
	cfg = importlib.import_module(config_module).config
	experiment = prep_experiment(cfg)
	result = run(experiment)
	return result

if __name__ == "__main__":
	main("experiments.configs.experiment_test_config")
