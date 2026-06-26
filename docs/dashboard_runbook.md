# CloudIQ Dashboard Runbook

## Architecture

The Streamlit dashboard is local-only and reads compact dashboard marts from
`data/dashboard/`. It does not start Spark on page refresh.

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
   - `reports/demand_forecast_validation_predictions.csv`
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
- `data/dashboard/demand_validation_predictions.csv`
- `data/dashboard/demand_model_metrics.json`
- `data/dashboard/data_quality_summary.json`

No customer IDs or customer-level records are exported.

## Local Launch

```powershell
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The dashboard shows setup
instructions if `data/dashboard/` has not been generated yet.

## Refresh Workflow

1. Re-run validated pipeline layers only when their source data or logic has
   intentionally changed.
2. Re-run demand training only when `gold/demand_features` has intentionally
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

Demand forecasting uses `gold/demand_features` with `target_next_month` as the
next calendar month's observed category unit demand. Validation is
chronological, using the final two available calendar months. The selected
champion is the `naive_lag_1` prior-month baseline because XGBoost was
evaluated and did not improve validation MAE.

## Churn Limitation

Churn classification is intentionally not presented as a flagship model. The
snapshot churn target has a 99.25% inactivity rate, which makes the supervised
classification framing unsuitable for this dataset without a different business
definition or sampling strategy.
