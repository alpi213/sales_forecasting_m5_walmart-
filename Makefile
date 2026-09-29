.PHONY: install test lint smoke data
install:
	pip install -e ".[torch,bayes,dev]"
test:
	pytest -q
lint:
	ruff check src tests
# End-to-end on synthetic data in a couple of minutes on a laptop CPU.
smoke:
	m5 make-synthetic --items 12 --days 500
	m5 build-db
	m5 train-lgbm --set lgbm.num_boost_round=300 --set data.min_train_days=250
	m5 predict --model lgbm
	m5 train-patchtst --cv-only --set patchtst.epochs=3 --set patchtst.lookback=112 --set cv.n_folds=1
	m5 elasticity-dml --set elasticity.min_weeks_per_item=20
	m5 elasticity-bayes --method advi --set elasticity.max_items_per_category=10 --set elasticity.min_weeks_per_item=20
# Real data: put the Kaggle files in data/raw first (kaggle competitions download -c m5-forecasting-accuracy)
data:
	m5 build-db --set paths.sales_file=sales_train_evaluation.csv
