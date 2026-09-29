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
pip install -e ".[torch,bayes,dev]"   # CPU torch; for a CUDA GPU use `make install-gpu` (see Makefile)
pytest -q                      # unit tests on a synthetic dataset (no download needed)
```

pip's default `torch` wheel on Windows and Linux is CPU-only even on a machine with an NVIDIA GPU;
`make install-gpu` installs the CUDA 12.6 build instead (one PatchTST epoch on 5000 series: 10 s
on an RTX 4080 vs 5 min on 32 CPU threads).

Real data: download the Kaggle files (`kaggle competitions download -c m5-forecasting-accuracy`)
into `data/raw/` (the `paths.raw_dir` key in `configs/default.yaml`), so that
`data/raw/calendar.csv`, `data/raw/sell_prices.csv` and `data/raw/sales_train_validation.csv`
exist. `sales_train_validation.csv` (days 1-1913) is the default; `sales_train_evaluation.csv`
(days 1-1941) is the same file with the final 28 days appended, so pass it with
`--set paths.sales_file=sales_train_evaluation.csv` when you have it. Then

```bash
m5 build-db                    # ~59M-row long table + SQL feature table in artifacts/m5.duckdb
m5 train-lgbm                  # 3-fold rolling-origin CV + final refit  (~30-60 min CPU, ~12 GB RAM)
m5 predict --model lgbm        # artifacts/submission_lgbm.csv
m5 train-patchtst --set patchtst.max_series=5000   # ~5 min for 3 folds on a GPU; subsample for CPU
m5 elasticity-dml                                  # ~2 min
m5 elasticity-bayes --method advi --set elasticity.max_items_per_category=200   # ~5 min
```

Outputs land in `artifacts/`: `lgbm_cv.json` / `patchtst_cv.json` (per-fold, per-level WRMSSE),
`submission_lgbm.csv` (Kaggle format), `elasticity_dml_{cat_id,dept_id,item_id}.csv` and
`elasticity_bayes_{category,series,hyper}.csv`.

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
result, not the leaderboard. Training samples are gathered as whole batches from a dense sales
tensor kept on the GPU (one indexing op per batch); a per-sample `DataLoader` would spend the
epoch in Python.

**Price elasticity via Double ML.** A plain log-log regression of sales on price is biased:
prices are cut when demand is expected to be weak and promotions coincide with high-traffic
weeks. DML fits E[log q | X] and E[log p | X] with gradient boosting, cross-fits them, and
regresses residual on residual; Neyman orthogonality makes the estimate insensitive to
first-order nuisance error, so it has a valid standard error. Item-store fixed effects are
removed by a within-transform *before* the nuisance fits, and the learner never sees the series
identity or a time index: in M5 the price is a deterministic function of (store, item, week), so
a learner given those inputs reproduces the price exactly, leaves no residual variation, and the
elasticity collapses to zero (R²(m) = 0.998 and θ ≈ 0 on the real data with that spec).

The synthetic test recovers the true category elasticities with this specification
(`tests/test_elasticity.py`); remaining attenuation at the low-volume end comes from `log1p` on
small counts. θ is the short-run elasticity (conditional on last week's demand), the same
quantity the Bayesian model estimates.

**Hierarchical Bayesian elasticity.** Most products have too little price variation to estimate
anything on their own. The PyMC model gives every item-store its own elasticity shrunk toward its
category mean by a data-chosen amount (non-centred parameterisation to avoid the funnel). It is
the answer to "we need an elasticity for all 30k SKUs, not three categories".

## Results on the real data

`sales_train_validation.csv` (days 1–1913), 3 rolling-origin folds, run on 2026-09-29.

| Model | fold d1886–1913 | fold d1858–1885 | fold d1830–1857 | mean WRMSSE |
|---|---|---|---|---|
| LightGBM, all 30 490 series | 0.534 | 0.652 | 0.734 | 0.640 |
| PatchTST, 5 000-series subsample | 0.671 | 0.730 | 0.723 | 0.708 |

For reference the seasonal-naive forecast scores about 1 and the M5 winning entry scored 0.52 on
the private leaderboard (a different 28-day window, so not directly comparable).

| Category | DML θ (± se) | Bayesian (ADVI, 200 items/cat) | naive pooled log-log |
|---|---|---|---|
| FOODS | −0.44 (0.009) | −0.77 | −0.57 |
| HOBBIES | −0.30 (0.014) | −0.45 | −0.20 |
| HOUSEHOLD | −0.24 (0.010) | −0.27 | −0.62 |

All seven departments come out negative (FOODS_1 −0.52 to HOUSEHOLD_2 −0.19). The item-level DML
estimates are noisy (median se 0.42), which is the case for Bayesian partial pooling: the hierarchical model shrinks each item toward its category by an amount the data chooses.

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
