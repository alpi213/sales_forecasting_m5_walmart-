# M5 sales forecasting and price elasticity

Retail demand forecasting on the M5 (Walmart) dataset, built as a small production pipeline
rather than a notebook:

| Stage | What | Where |
|---|---|---|
| Data | validation, wide→long load into DuckDB, SQL feature engineering with window functions | `src/m5/data/` |
| Evaluation | rolling-origin CV, WRMSSE (the official M5 metric, all 12 aggregation levels) | `src/m5/evaluation/` |
| Baseline | global LightGBM with a Tweedie objective, direct 28-day horizon | `src/m5/models/lgbm.py` |
| Deep model | PatchTST written from scratch in PyTorch, same CV protocol and metric | `src/m5/models/patchtst.py`, `torch_train.py` |
| Causal | price elasticity by Double ML and by a hierarchical Bayesian model (PyMC) | `src/m5/models/elasticity_*.py` |
| Ops | config-driven CLI, structured logging, tests, CI, Docker | `src/m5/cli.py`, `tests/`, `.github/`, `Dockerfile` |

## Quick start

```bash
pip install -e ".[torch,bayes,dev]"
pytest -q                      # unit tests on a synthetic dataset (no download needed)
make smoke                     # every stage end-to-end on synthetic data, a few minutes on CPU
```

Real data: download the Kaggle files (`kaggle competitions download -c m5-forecasting-accuracy`)
into `data/raw/`, then

```bash
m5 build-db --set paths.sales_file=sales_train_evaluation.csv
m5 train-lgbm                  # 3-fold rolling-origin CV + final refit  (~30-60 min CPU, ~12 GB RAM)
m5 predict --model lgbm        # artifacts/submission_lgbm.csv
m5 train-patchtst --set patchtst.max_series=5000   # GPU recommended; subsample for CPU
m5 elasticity-dml
m5 elasticity-bayes --method advi --set elasticity.max_items_per_category=200
```

Every stage reads `configs/default.yaml`; any key can be overridden with `--set section.key=value`.
Stages communicate only through the DuckDB file and `artifacts/`, so they can be scheduled independently.

## Design decisions (and why)

**Direct multi-horizon instead of recursive.** Every sales-derived feature is shifted by at
least 28 days (lags ≥ 28, rolling windows ending 28 days back), so one model forecasts all 28
days from observed data only. Recursive forecasting (lag-1 features, feeding predictions back)
uses fresher information but compounds errors over the horizon and needs 28 sequential feature
recomputations at inference. The config validator rejects any lag smaller than the horizon: that
would be leakage. Price features are *not* shifted, because the price list for the forecast weeks
is known (in a retailer, next month's prices are a decision, not a forecast).

**Global model, Tweedie loss.** One LightGBM model over ~30k series learns cross-sectional
effects (price, calendar, department) that per-series models cannot. Daily item-store sales are
non-negative, intermittent and over-dispersed; Tweedie with variance power in (1, 2) is a
Poisson–Gamma compound that models the zero mass and the skew. MSE over-predicts on zeros, and
plain Poisson under-fits the dispersion. The PyTorch model optimises the same Tweedie deviance
so the comparison is like for like.

**WRMSSE, not RMSE.** Bottom-level RMSE is dominated by a few high-volume items and says nothing
about whether the store or category totals are right. WRMSSE scales every series by its own
in-sample naive error, weights it by dollar sales, and averages over 12 hierarchy levels. The
naive seasonal forecast scores ≈ 1; the test suite checks that.

**Rolling-origin CV, no random splits.** Validation windows are the last 28 days, then the 28
before, and so on. A shuffled K-fold would leak through the lag features and over-state accuracy.

**PatchTST.** Channel-independent transformer over patches of the series (`patch_len=16`,
`stride=8`, 224-day lookback → 27 tokens), instance normalisation in/out, pre-norm encoder,
flatten head, sharpened softplus for positivity. Written on top of `nn.TransformerEncoderLayer`
rather than imported from a library so every design choice can be defended. Expectation on M5:
it does **not** beat LightGBM. Gradient boosting wins on tabular, intermittent, heavily
covariate-driven data; transformers shine when the signal is in the shape of long histories
(energy, traffic) and there are few informative exogenous features. The comparison is the
result, not the leaderboard.

**Price elasticity via Double ML.** A plain log-log regression of sales on price is biased:
prices are cut when demand is expected to be weak and promotions coincide with high-traffic
weeks. DML fits E[log q | X] and E[log p | X] with gradient boosting, cross-fits them, and
regresses residual on residual; Neyman orthogonality makes the estimate insensitive to
first-order nuisance error, so it has a valid standard error. On the synthetic data the naive
slope is off by 0.6–1.3 while DML recovers the true category elasticities within the confidence
interval (`tests/test_elasticity.py`). Remaining attenuation at the low-volume end comes from
`log1p` on small counts.

**Hierarchical Bayesian elasticity.** Most products have too little price variation to estimate
anything on their own. The PyMC model gives every item-store its own elasticity shrunk toward its
category mean by a data-chosen amount (non-centred parameterisation to avoid the funnel). It is
the answer to "we need an elasticity for all 30k SKUs, not three categories".

## Taking it to production (system-design view)

* **Retraining cadence:** weekly, after the week's sales land; CV chooses the number of rounds,
  the final model refits on all data. `artifacts/lgbm_cv.json` is the model card.
* **Serving:** batch. The 28-day forecast is a nightly job (`m5 predict`) writing to a table
  consumed by replenishment; no online inference is needed for this use case.
* **Monitoring:** WRMSSE of last week's forecast vs actuals per level (drift shows first at the
  department level), feature null rates (a broken price feed shows up as `sell_price` nulls
  before it shows up in accuracy), and prediction volume vs history.
* **Data validation** runs before anything touches the database (`data/validate.py`) and the
  stage exits non-zero with a precise message, so a scheduler retries or pages.
* **Scaling:** the feature SQL is engine-agnostic; on Spark/BigQuery the same window functions
  apply. LightGBM training on the full data is a single 16 GB machine; PatchTST wants one GPU.

## Layout

```
configs/default.yaml        all hyper-parameters
src/m5/data/                validate.py, load.py (DuckDB), features.py (SQL), synthetic.py
src/m5/evaluation/          splits.py, wrmsse.py
src/m5/models/              lgbm.py, patchtst.py, torch_train.py, elasticity_{data,dml,bayes}.py
src/m5/cli.py               m5 <stage>
tests/                      pytest, runs on sqlite so duckdb/lightgbm/torch are optional
```
