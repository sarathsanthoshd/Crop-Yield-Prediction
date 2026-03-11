"""
=============================================================================
  CROP YIELD PREDICTOR — TRAINING SCRIPT  (train_model.py)

  Run this ONCE inside your project folder:
      python train_model.py

  Produces:
      crop_yield_model_v2.joblib    ← plain sklearn Pipeline (no custom class)
      unique_values_v2.joblib       ← all dropdown values + state climate defaults

  Requirements:
      pip install scikit-learn pandas numpy joblib matplotlib seaborn
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── STEP 1 : Load dataset ──────────────────────────────────────────────────────
DATA_PATH = "Crop_yield_dataset.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── STEP 2 : Log-transform target ─────────────────────────────────────────────
df['Yield_log'] = np.log1p(df['Yield'])
print(f"Yield skew raw={df['Yield'].skew():.2f}  log={df['Yield_log'].skew():.2f}")

# ── STEP 3 : Feature setup ────────────────────────────────────────────────────
DROP_COLS = ['Yield','Yield_log','Production','Fertilizer','Pesticide',
             'Season','Harvest_Month']
X = df.drop(columns=DROP_COLS)
y = df['Yield_log']

CATEGORICAL_COLS = ['Crop','Crop_Type','Season_clean','State',
                    'Region','Soil_Type','Irrigation_Type']
NUMERICAL_COLS   = [c for c in X.columns if c not in CATEGORICAL_COLS]
print(f"Features: {len(CATEGORICAL_COLS)} categorical, {len(NUMERICAL_COLS)} numerical")

# ── STEP 4 : Preprocessing ────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ), CATEGORICAL_COLS)
], remainder='passthrough')

# ── STEP 5 : Pipeline (plain sklearn — NO custom classes) ─────────────────────
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1,
        early_stopping=True, n_iter_no_change=30,
        validation_fraction=0.1, random_state=42,
    )),
])

# ── STEP 6 : Train ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {X_train.shape[0]:,} rows...")
pipeline.fit(X_train, y_train)
print(f"Done. Iterations used: {pipeline.named_steps['model'].n_iter_}")

# ── STEP 7 : Evaluate ─────────────────────────────────────────────────────────
# np.expm1 reverses the log1p transform applied to the target
y_pred = np.expm1(pipeline.predict(X_test))
y_true = np.expm1(y_test)
r2   = r2_score(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"R2={r2:.4f}  MAE={mae:.4f} t/ha  RMSE={rmse:.4f} t/ha")

# Optional evaluation plot
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cap  = np.percentile(y_true, 99)
    mask = y_true <= cap
    axes[0].scatter(y_true[mask], y_pred[mask], alpha=0.3, s=5, color='#2d6a4f')
    axes[0].plot([0,cap],[0,cap],'r--',lw=1.5)
    axes[0].set(xlabel='Actual (t/ha)', ylabel='Predicted (t/ha)',
                title=f'Actual vs Predicted  R²={r2:.3f}')
    axes[1].hist((y_pred-y_true)[mask], bins=60, color='#52b788', edgecolor='white', lw=0.3)
    axes[1].axvline(0, color='red', ls='--', lw=1.5)
    axes[1].set(xlabel='Residual', title='Residual Distribution')
    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150, bbox_inches='tight')
    print("Saved: model_evaluation.png")
    plt.close()
except Exception as e:
    print(f"Plot skipped: {e}")

# ── STEP 8 : Save model ───────────────────────────────────────────────────────
# Plain sklearn Pipeline — no custom wrapper class.
# app.py calls np.expm1(pipeline.predict(X)) to get t/ha values.
joblib.dump(pipeline, "crop_yield_model_v2.joblib")
print("Saved: crop_yield_model_v2.joblib")

# ── STEP 9 : Save metadata ────────────────────────────────────────────────────
metadata = {
    'unique_values': {
        'Crop':            sorted(df['Crop'].unique().tolist()),
        'Season':          sorted(df['Season_clean'].unique().tolist()),
        'State':           sorted(df['State'].unique().tolist()),
        'Soil_Type':       sorted(df['Soil_Type'].unique().tolist()),
        'Irrigation_Type': sorted(df['Irrigation_Type'].unique().tolist()),
        'Region':          sorted(df['Region'].unique().tolist()),
        'Crop_Type':       sorted(df['Crop_Type'].unique().tolist()),
    },
    'state_climate': (
        df.groupby('State')[['Annual_Rainfall','Humidity','Avg_Temperature',
                              'Max_Temperature','Min_Temperature']]
        .mean().round(1).to_dict('index')
    ),
    'crop_seasons': (
        df.groupby('Crop')['Season_clean']
        .apply(lambda x: sorted(x.unique().tolist())).to_dict()
    ),
    'crop_type_map': (
        df.drop_duplicates('Crop').set_index('Crop')['Crop_Type'].to_dict()
    ),
    'season_default_month': (
        df.groupby('Season_clean')['Harvest_Month']
        .agg(lambda x: int(x.mode()[0])).to_dict()
    ),
    'model_r2':   round(r2,   4),
    'model_mae':  round(mae,  4),
    'model_rmse': round(rmse, 4),
}
joblib.dump(metadata, "unique_values_v2.joblib")
print("Saved: unique_values_v2.joblib")
print(f"\nAll done! Run:  streamlit run app.py")