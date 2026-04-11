# Tamil Nadu Fertilizer Recommendation System

A production-focused fertilizer recommendation project for Tamil Nadu, built with a Streamlit dashboard and a Spark Random Forest model exported as JSON metadata for fast, reproducible inference.

## 1. Project Summary

This project predicts the most suitable fertilizer based on:
- Soil profile (type, pH, moisture, organic carbon, electrical conductivity)
- Nutrient levels (N, P, K)
- Climate context (temperature, humidity, rainfall)
- Agronomic context (crop, growth stage, irrigation type)
- Spatial and temporal context (district/region + season inferred from date)

The current codebase is streamlined for the active dashboard pipeline and has removed legacy experimental folders not used by deployment.

## 2. Final Results

## 2.1 Current Production Model
- Model type: `spark_random_forest_json`
- Algorithm: Spark ML `RandomForestClassifier` (multiclass)
- Validation accuracy: `0.8667360749609578` (86.67%)

Source:
- `artifacts/tn_fertilizer_spark_model/metadata.json`

## 2.2 Historical Baseline Comparison
Historical metrics used for comparison:
- MLP Accuracy: `0.14239514403701678` (14.24%)
- Random Forest Accuracy: `0.13965868948703916` (13.97%)

## 2.3 Improvement
Compared to historical baseline values:
- Improvement vs historical RF: `+72.71` percentage points
- Improvement vs historical MLP: `+72.43` percentage points
- Relative uplift vs historical RF: about `6.21x`
- Relative uplift vs historical MLP: about `6.09x`

## 2.4 Accuracy Graph
- Static graph file: `artifacts/model_accuracy_graph.png`
- Dashboard graph section: rendered directly in the app

## 3. Key Findings

1. Richer agronomic + climate + spatial + temporal features significantly improve recommendation quality.
2. Spark Random Forest handles nonlinear interactions between soil, crop, and weather better than earlier baseline models.
3. Artifact-driven inference (saved metadata + parsed trees) gives stable and repeatable behavior in deployment.
4. Tamil Nadu-specific constraints (districts, region mapping, crop set, season logic) improve practical relevance.

## 4. Architecture

High-level flow:
1. Data is loaded from the production dataset.
2. Spark pipeline encodes categorical variables and assembles features.
3. Random Forest model is trained and exported as metadata + tree debug strings.
4. Streamlit dashboard loads artifact and performs tree-vote inference.
5. Dashboard returns:
   - Recommended fertilizer
   - Top confidence scores
   - Suitability explanation

Core files:
- `temporal_spatial_dashboard.py` - UI, map, input flow, prediction display
- `tn_model_utils.py` - training, artifact loading, tree parsing, inference, context helpers
- `pretrain_tn_model.py` - one-time model artifact generation
- `requirements_dashboard.txt` - dependencies

## 5. Datasets

Current dataset files in `datasets/` include:
- `fertilizer_recommendation.csv` (primary training source for current model)
- `Tamilnadu agriculture yield data.csv`
- `Tamilnadu Crop-Production.csv`
- `crop_production_history.csv`
- `rainfall_data.csv`
- `land_use.csv`
- `rice_production.csv`
- `ICRISAT-District Level Data.csv`
- `RS_Session_248_AU_648.csv`
- `fertilizers-recommendation.ipynb`

Note: The production training path currently uses `datasets/fertilizer_recommendation.csv` directly.

## 6. Feature Schema

### Categorical
- `Soil_Type`
- `Crop_Type`
- `Crop_Growth_Stage`
- `Season`
- `Irrigation_Type`
- `Region`

### Numeric
- `Soil_pH`
- `Soil_Moisture`
- `Organic_Carbon`
- `Electrical_Conductivity`
- `Nitrogen_Level`
- `Phosphorus_Level`
- `Potassium_Level`
- `Temperature`
- `Humidity`
- `Rainfall`

Target label:
- `Recommended_Fertilizer`

## 7. How to Run

## 7.1 Environment Setup
Use your project virtual environment, then install dependencies:

```bash
pip install -r requirements_dashboard.txt
```

## 7.2 Train and Save Artifact (one-time)

```bash
python pretrain_tn_model.py
```

## 7.3 Launch Dashboard

```bash
streamlit run temporal_spatial_dashboard.py --server.port 8501
```

## 8. Dashboard Capabilities

- District selection for Tamil Nadu
- Optional map click coordinate selection
- Live weather context fetch (Open-Meteo) with Tamil Nadu heuristics fallback
- Manual overrides for all major input variables
- Prediction with confidence ranking and suitability narrative
- Model accuracy graph in UI


