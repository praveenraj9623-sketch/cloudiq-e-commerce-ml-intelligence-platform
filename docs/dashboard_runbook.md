# CloudIQ Dashboard Runbook

## Architecture

The Streamlit dashboard is local-only and reads compact dashboard marts from
`data/dashboard/`. It does not start Spark on page refresh. All data is
historical Olist marketplace data from 2016-2018 and is not live production
data.

Source flow:

1. Validated Delta outputs already exist under `data/bronze/`, `data/silver/`,
   and `data/gold/`.
2. `scripts/export_dashboard_data.py` starts one local Delta-enabled Spark
   session and reads:
   - `gold/bi_revenue`
   - `gold/rfm_segments`
   - `gold/demand_features`
   - `silver/seller_performance`
   - `silver/master_orders`
3. The export script also reads local report artifacts when present:
   - `reports/demand_forecast_metrics.json`
   - `reports/demand_backtest_predictions.csv`
   - `reports/demand_backtest_monthly_aggregate.csv`
   - `reports/demand_backtest_category_errors.csv`
   - `reports/olist_data_audit.json`
4. Streamlit reads only CSV/JSON marts from `data/dashboard/`.

## Dashboard Data Export

Run this after Bronze, Silver, Gold, and demand-model validation:

```powershell
python scripts/export_dashboard_data.py
```

The export writes:

- `data/dashboard/overview_kpis.json`
- `data/dashboard/monthly_revenue.csv`
- `data/dashboard/state_revenue.csv`
- `data/dashboard/payment_mix.csv`
- `data/dashboard/delivery_performance.csv`
- `data/dashboard/rfm_segment_distribution.csv`
- `data/dashboard/rfm_segment_profiles.csv`
- `data/dashboard/seller_performance.csv`
- `data/dashboard/demand_backtest_input.csv`
- `data/dashboard/demand_backtest_predictions.csv`
- `data/dashboard/demand_backtest_monthly_aggregate.csv`
- `data/dashboard/demand_backtest_category_errors.csv`
- `data/dashboard/demand_backtest_fold_metrics.csv`
- `data/dashboard/demand_model_metrics.json`
- `data/dashboard/data_quality_summary.json`

No customer IDs or customer-level records are exported. Seller IDs are shortened
for public-facing display.

## Local Launch

```powershell
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The dashboard shows setup
instructions if `data/dashboard/` has not been generated yet.

## Refresh Workflow

1. Re-run validated pipeline layers only when their source data or logic has
   intentionally changed.
2. Re-run demand backtesting only when `gold/demand_features` has intentionally
   changed.
3. Re-export dashboard marts:

```powershell
python scripts/export_dashboard_data.py
```

4. Refresh the Streamlit browser tab.

## Full Test Command

The local Spark tests can exhaust JVM native memory on this Windows machine
when Spark uses all CPU cores. Use the same JVM mitigation that previously
allowed the full suite to pass:

```powershell
$env:JAVA_TOOL_OPTIONS="-XX:ActiveProcessorCount=2 -XX:CICompilerCount=2 -XX:TieredStopAtLevel=1 -Xss512k"
python -m pytest tests -q
Remove-Item Env:\JAVA_TOOL_OPTIONS
```

## Model Selection Note

Demand forecasting uses `forecast_month` as the month being predicted and
`target_units` as the observed category unit demand in that month. The feature
availability cutoff is the end of the prior month: `lag_1` is
`forecast_month - 1`, `lag_2` is `forecast_month - 2`, `lag_4` is
`forecast_month - 4`, and `rolling_mean_3` averages `forecast_month - 3`
through `forecast_month - 1`. Validation is an expanding-window chronological
backtest that trains only on earlier forecast months. Partial or low-volume
target months are excluded from evaluation.

Headline model selection prioritizes pooled WAPE and MAE, with RMSE secondary.
MAPE is retained as a diagnostic only because low-volume categories can distort
percentage errors.
The selected champion is currently `naive_lag_1`, displayed as "Naive Lag-1
Benchmark", because XGBoost was evaluated and did not improve pooled
chronological MAE/WAPE.

## Metric Definitions

- Item Merchandise Value (excludes freight): sum of item price values from
  `silver/master_orders.order_revenue`.
- Average Item Merchandise Value per Order: item merchandise value excluding
  freight divided by distinct orders.
- Seller-Attributed Order Value (item + freight): seller-level item price plus
  seller-level freight value.
- Orders by Primary Payment Type: one deterministic payment type per order,
  selected by highest payment value, then lowest payment sequence, then payment
  type alphabetically.
- Late-delivery rate = delivered orders received after the estimated delivery
  date divided by all delivered orders with valid actual and estimated delivery
  dates. Months below 100 valid delivered orders are flagged and hidden from the
  default trend.
- RFM segments: relative historical behavior tiers. Repeat purchasing is
  limited in this dataset, so labels do not prove loyalty or churn risk.

## Churn Limitation

Churn classification is intentionally not presented as a flagship model. The
snapshot churn target has a 99.25% inactivity rate, which makes the supervised
classification framing unsuitable for this dataset without a different business
definition or sampling strategy.
