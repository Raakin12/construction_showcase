# Construction Cycle Time Prediction

This project forecasts how many days a home will take to build using information available when construction starts. It compares forecasts with the incumbent floor-plan average, examines project and operational associations, and produces predictions for 578 homes.

The [research notebook](notebooks/cycle_time_research.ipynb) contains the data audit, model comparisons, statistical analysis, charts, and findings. The saved run selects **M4_time**, a CatBoost model combining project specifications, geography, and start-time information.

## Project files

| File | Purpose |
| --- | --- |
| `data/build_history.csv` | 4,684 historical rows before deduplication |
| `data/homes_to_predict.csv` | 578 homes requiring forecasts |
| `notebooks/cycle_time_research.ipynb` | Complete analysis and saved outputs |
| `src/data.py` | Cleaning, cohort audits, and output validation |
| `src/features.py` | Start-day features and training-fitted square-footage repair |
| `src/modeling.py` | Baselines, Ridge, and CatBoost models |
| `src/validation.py` | Temporal backtests, metrics, bootstrap intervals, and diagnostics |
| `predictions.csv` | `home_id` and `predicted_cycle_days`, in original input order |
| `requirements.txt` | Pinned Python dependencies |

## Mathematical and statistical approach

### Target and information boundary

The target is elapsed construction time:

$$y_i = \text{construction end date}_i - \text{construction start date}_i.$$

The analysis removes 25 exact duplicate rows and excludes seven non-positive durations from supervised modeling. It retains 575 unfinished homes in the start-event ledger, leaving 4,077 valid observed targets. Long positive durations are retained.

Forecast features exclude completion dates, inspection dates, final invoices, change orders, and construction status. Options value is excluded from the primary model because its start-day availability is unverified. This information boundary prevents the model from using future outcomes to predict the past.

Month is represented by sine and cosine terms, preserving the seasonal proximity of December and January. Permit and sale features use only events recorded on or before construction start. Pressure features count prior starts in the interval `[start − 90 days, start)`, excluding same-day and future starts; these measure incoming volume, not active workload.

Suspected factor-of-ten square-footage errors are repaired only when dividing by ten places the value within 20% of its training-sample plan median. Fitting this rule within each forecast training sample prevents holdout information from influencing the repair.

### Forecast models and their purpose

| Method | How it helps |
| --- | --- |
| Global mean | Establishes a simple reference forecast for every home. |
| Floor-plan mean | Reproduces the incumbent benchmark; unseen plans fall back to the global mean. |
| Ridge regression | Provides an additive reference with an L2 coefficient penalty, reducing instability from correlated predictors. Numeric imputation and scaling are fitted on training data; categories are one-hot encoded. |
| CatBoost regression | Captures nonlinear relationships and interactions in mixed numeric and categorical data. The fixed configuration uses 450 trees, depth 5, learning rate 0.045, L2 leaf regularization 8, and seed 42. |

Ridge minimizes squared prediction error plus a coefficient penalty:

$$\sum_i (y_i - \beta_0 - x_i^\top\beta)^2 + 20\lVert\beta\rVert_2^2.$$

CatBoost is trained with RMSE loss. Nested feature families add specifications, geography, time/process information, recent-start pressure, and manager identity. Comparing these families tests whether extra information improves forecasting. A separate plan-only CatBoost comparison distinguishes model-form gains from gains due to additional features.

### Temporal validation and error measures

At each forecast origin, training uses only valid outcomes completed **before** that date. Development windows cover August–October 2022, February–April 2023, and August–October 2023, with development labels restricted to outcomes known before August 2024. The later evaluation covers August–October 2024.

For error $e_i = \hat y_i-y_i$:

$$\mathrm{MAE}=\frac{1}{n}\sum_i |e_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i e_i^2},\qquad
\mathrm{bias}=\frac{1}{n}\sum_i e_i.$$

MAE measures typical error in days and drives model selection. RMSE emphasizes large misses; signed bias reveals systematic over- or underprediction. The notebook also reports the 90th percentile absolute error and the share within 30 days.

Selection uses pooled development MAE. Among models within one day of the best score, the declared preference order favors simpler models. The selected specification is then evaluated on the later period.

### Uncertainty, manager comparisons, and associations

- **Paired month-block bootstrap:** 2,000 resamples of start-month blocks estimate 95% intervals for mean error improvements and manager residuals. Keeping homes from each sampled month together accounts for within-month dependence. These intervals condition on fitted models and do not capture model-fitting uncertainty or dependence across months.
- **Empirical forecast bands:** the 90th percentile of development absolute errors supplies an allowance around each forecast, with the lower bound clipped to one day. This gives a planning range, without guaranteeing future coverage.
- **Adjusted manager comparisons:** actual-minus-predicted duration from models excluding manager identity is centered within each evaluation period, then averaged by manager. This helps separate observed project mix and common period bias from manager-associated differences. Alternative adjustments, fold checks, and a direct comparison of the leading two managers assess stability; the results do not establish causal personnel rankings.
- **Adjusted OLS regression:** plan, community, quarter, manager, and project/process controls estimate conditional associations with duration. Standard errors are clustered by construction-start month to allow correlated errors within a month. Alternative specifications and a later sample test stability; coefficients do not establish causation.
- **Sensitivity checks:** older-start training, removing square footage or start year, and a diagnostic options model expose dependence on cleaning, feature choices, and selectively completed recent cohorts.

## Saved results and limitations

On 287 scored homes in the later 2024 cohort, M4_time achieves **21.41-day MAE**, compared with **36.42 days** for the floor-plan baseline: a **41.2% reduction**. RMSE is **28.22 days**, and signed bias is **−9.39 days**. These are saved notebook results, not a guarantee of 2025 performance.

The empirical allowance is approximately **±76.71 days**, covering **97.56%** of the later scored cohort. Recent cohorts are selectively complete, and alternative final training specifications shift forecasts materially. The export timestamp is assumed rather than verified, and manager assignments are assumed to reflect start-day state.

## Reproduce the notebook

The saved run records **Python 3.13.2**. Use that version and the pinned dependencies for the closest reproduction. From the project root, on macOS/Linux:

```bash
python3.13 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name construction-repro --display-name "Construction research"
cd notebooks
python -m jupyterlab
```

Open `cycle_time_research.ipynb`, select **Construction research**, and choose **Restart Kernel and Run All Cells**. The kernel working directory must be `notebooks/`: the notebook uses `../data` and imports `src` from its parent directory. The first cell should print `True` for both input files. Save the notebook after execution to retain the new outputs.

For command-line execution, after the same setup and while still in `notebooks/`:

```bash
python -m jupyter nbconvert --to notebook --execute cycle_time_research.ipynb \
  --ExecutePreprocessor.kernel_name=construction-repro \
  --ExecutePreprocessor.timeout=1800 \
  --output cycle_time_research.reproduced.ipynb
```

Both execution paths **overwrite the root `predictions.csv`**. The command-line path saves notebook outputs in a separate reproduced notebook. No external API or credentials are needed for analysis; installing dependencies requires access to the package index. Fresh dependency installation has not been verified as part of this README update.

The final export cell checks 578 unique IDs in input order, positive finite predictions, and repeat-prediction consistency. It prints package versions and SHA-256 hashes of inputs, source files, and the CSV. Compare these with the supplied notebook's recorded values when checking reproducibility; environment differences can affect numerical results.
