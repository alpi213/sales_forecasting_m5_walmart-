.PHONY: install install-gpu test lint data
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
# Real data: put the Kaggle files in data/raw first (kaggle competitions download -c m5-forecasting-accuracy)
data:
	m5 build-db --set paths.sales_file=sales_train_evaluation.csv
