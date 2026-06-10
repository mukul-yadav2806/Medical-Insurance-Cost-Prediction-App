# Medical-Insurance-Cost-Prediction-App
Medical Insurance Cost Prediction App
# 🏥 MedCost AI — Medical Insurance Cost Predictor

A beautiful, dark-themed Streamlit app for predicting individual medical insurance charges using ML.

## 🚀 Run the App

```bash
# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

## 📊 Dataset

Uses the **Kaggle Medical Cost Personal Dataset** (1,338 records).

If you have the original CSV from Kaggle, replace `insurance.csv` with it.
Otherwise run `python generate_data.py` to create a synthetic equivalent.

**Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`  
**Target:** `charges` (annual medical insurance cost in USD)

## 🤖 Models

| Model | R² Score |
|-------|---------|
| Gradient Boosting | ~0.965 |
| Random Forest | ~0.940 |
| Ridge Regression | ~0.780 |

## 🎨 Features

- **Prediction Tab** — Real-time cost estimate with risk tier, percentile, cost-driver donut chart
- **Data Insights Tab** — Distribution plots, scatter analysis, correlation heatmap
- **Model Performance Tab** — Actual vs predicted, residuals, feature importance, model comparison
