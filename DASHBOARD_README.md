# Tamil Nadu Temporal + Spatial Dashboard

This dashboard is configured for Tamil Nadu-focused fertilizer recommendation.

It uses:
- Temporal context (date, month, season, day-of-year)
- Spatial context (latitude and longitude)
- Auto-fetched climate context from clicked location (temperature, humidity, moisture)
- Manual agronomic choices for soil type, crop, and NPK values

## One-time setup

1. Install dependencies:

   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. Pretrain and save the model artifact:

   ```bash
   python pretrain_tn_model.py
   ```

## Run dashboard

```bash
streamlit run temporal_spatial_dashboard.py
```

## Behavior

- The dashboard loads a saved model artifact from `artifacts/tn_fertilizer_model.joblib`.
- No runtime retraining on dashboard load.
- Clicking a location on the Tamil Nadu map auto-updates latitude, longitude, temperature, humidity, and moisture.
- Crop options are restricted to major Tamil Nadu crops.
