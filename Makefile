.PHONY: install install-gpu test lint smoke data
install:
	pip install -e ".[torch,bayes,dev]"
# On Windows/Linux pip's default torch wheel is CPU-only; pick the CUDA build your driver supports
# (nvidia-smi shows the max CUDA version). cu126 works with driver >= 560.
install-gpu:
	pip install -e ".[bayes,dev]"
	pip install torch --index-url https://download.pytorch.org/whl/cu126
test:
	pytest -q
lint:
	ruff check src tests
# End-to-end on synthetic data in a couple of minutes on a laptop CPU.
# Uses its own raw dir, database and artifacts dir so it never overwrites the real data or models.
SMOKE = --set paths.raw_dir=data/synthetic --set paths.db_path=artifacts/smoke/m5.duckdb --set paths.artifacts_dir=artifacts/smoke
smoke:
	m5 make-synthetic --out data/synthetic --items 12 --days 500
	m5 build-db $(SMOKE)
	m5 train-lgbm $(SMOKE) --set lgbm.num_boost_round=300 --set data.min_train_days=250
	m5 predict --model lgbm $(SMOKE)
	m5 train-patchtst --cv-only $(SMOKE) --set patchtst.epochs=3 --set patchtst.lookback=112 --set cv.n_folds=1
	m5 elasticity-dml $(SMOKE) --set elasticity.min_weeks_per_item=20
	m5 elasticity-bayes --method advi $(SMOKE) --set elasticity.max_items_per_category=10 --set elasticity.min_weeks_per_item=20
# Real data: put the Kaggle files in data/raw first (kaggle competitions download -c m5-forecasting-accuracy)
data:
	m5 build-db --set paths.sales_file=sales_train_evaluation.csv
