"""
XGBoost training pipeline for used car price prediction.

Technique summary:
- One-hot encoding for categorical features
- Standard scaling for numeric features
- HalvingRandomSearchCV for efficient hyperparameter search
- Permutation importance for interpretation
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform

from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
DATASET_PATH = PROJECT_DIR / "car_price_dataset_cleaned.csv"


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(MODEL_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace("Not_Available", np.nan)

    drop_cols = [c for c in ["Model", "Town"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    binary_cols = ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Millage(KM)" in df.columns and "Car_Age" in df.columns:
        df["Mileage_Per_Year"] = df["Millage(KM)"] / (df["Car_Age"] + 1)

    if "Engine (cc)" in df.columns:
        df["Engine_Category"] = pd.cut(
            df["Engine (cc)"],
            bins=[0, 1000, 1500, 2000, 3000, np.inf],
            labels=["Small", "Medium", "Large", "XLarge", "Luxury"],
        )

    if all(col in df.columns for col in binary_cols):
        df["Comfort_Score"] = df[binary_cols].sum(axis=1)

    return df


def use_log_target(price: pd.Series) -> bool:
    return abs(price.skew()) > 0.75


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def make_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=bins)


def plot_eda(df: pd.DataFrame) -> None:
    price = df["Price"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(price, bins=50, edgecolor="black", alpha=0.7)
    plt.title("Price Distribution")
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(price), bins=50, edgecolor="black", alpha=0.7)
    plt.title("Log Price Distribution")
    save_plot("price_distribution.png")

    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")

    plt.figure(figsize=(14, 8))
    axes = plt.gca()
    axes.boxplot([df[c].dropna() for c in [c for c in ["Price", "Engine (cc)", "Millage(KM)", "Car_Age"] if c in df.columns]])
    axes.set_xticklabels([c for c in ["Price", "Engine (cc)", "Millage(KM)", "Car_Age"] if c in df.columns])
    axes.set_title("Outlier Boxplots")
    save_plot("outlier_boxplots.png")


def make_learning_curves(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    bins = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        cv=cv.split(X_train, bins),
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse.mean(axis=1), "o-", label="Training RMSE")
    plt.plot(train_sizes, val_rmse.mean(axis=1), "o-", label="Validation RMSE")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title("XGBoost Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("learning_curves.png")


def tune_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict, float]:
    bins = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_train, bins))
    search = HalvingRandomSearchCV(
        pipeline,
        {
            "model__n_estimators": randint(300, 900),
            "model__max_depth": randint(3, 10),
            "model__learning_rate": uniform(0.02, 0.18),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 8),
            "model__gamma": uniform(0.0, 0.5),
            "model__reg_alpha": uniform(0.0, 1.0),
            "model__reg_lambda": uniform(0.5, 2.5),
        },
        n_candidates=40,
        factor=3,
        cv=splits,
        scoring="neg_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(np.sqrt(-search.best_score_))


def train_baseline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, float, float]:
    bins = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv.split(X_train, bins),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    r2_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv.split(X_train, bins),
        scoring="r2",
        n_jobs=-1,
    )
    pipeline.fit(X_train, y_train)
    return pipeline, float(np.sqrt(-rmse_scores).mean()), float(r2_scores.mean())


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_target: bool,
) -> dict:
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    if log_target:
        y_train = np.expm1(y_train)
        y_test = np.expm1(y_test)
        y_train_pred = np.expm1(y_train_pred)
        y_test_pred = np.expm1(y_test_pred)

    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "test_mape": float(mean_absolute_percentage_error(y_test, y_test_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
    }

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.5, edgecolor="k", linewidth=0.4)
    max_val = max(float(np.max(y_test)), float(np.max(y_test_pred)))
    plt.plot([0, max_val], [0, max_val], "r--")
    plt.title("Predicted vs Actual")
    plt.subplot(1, 2, 2)
    plt.hist(y_test - y_test_pred, bins=50, edgecolor="black", alpha=0.7)
    plt.title("Residuals")
    save_plot("model_evaluation.png")

    return metrics


def plot_feature_importance(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X_train)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = list(X_train.columns)

    importance = permutation_importance(
        model,
        X_transformed,
        y_train,
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance.importances_mean,
            "Std": importance.importances_std,
        }
    ).sort_values("Importance", ascending=False)

    top = df.head(min(15, len(df)))
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top)), top["Importance"], xerr=top["Std"], alpha=0.8)
    plt.yticks(range(len(top)), top["Feature"])
    plt.gca().invert_yaxis()
    plt.title("XGBoost Feature Importance")
    save_plot("feature_importance.png")
    df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    return df


def save_bundle(pipeline: Pipeline, log_target: bool, best_params: dict, metrics: dict, feature_names: list[str]) -> None:
    joblib.dump(pipeline, MODEL_DIR / "xgboost_model.joblib")
    metadata = {
        "model_type": "XGBRegressor",
        "training_technique": "One-hot encoding + StandardScaler + HalvingRandomSearchCV",
        "feature_names": feature_names,
        "use_log_transform": bool(log_target),
        "best_params": {k.replace("model__", ""): v for k, v in best_params.items()},
        "metrics": metrics,
        "created_at": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    predict_code = '''"""
Prediction helper for the trained XGBoost model.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)


def predict(input_dict: dict) -> dict:
    try:
        df = pd.DataFrame([input_dict])
        df["Mileage_Per_Year"] = df["Millage(KM)"] / (df["Car_Age"] + 1)
        df["Engine_Category"] = pd.cut(
            df["Engine (cc)"],
            bins=[0, 1000, 1500, 2000, 3000, np.inf],
            labels=["Small", "Medium", "Large", "XLarge", "Luxury"],
        )
        df["Comfort_Score"] = (
            pd.to_numeric(df["AIR CONDITION"], errors="coerce").fillna(0)
            + pd.to_numeric(df["POWER STEERING"], errors="coerce").fillna(0)
            + pd.to_numeric(df["POWER MIRROR"], errors="coerce").fillna(0)
            + pd.to_numeric(df["POWER WINDOW"], errors="coerce").fillna(0)
        )
        prediction = model.predict(df)[0]
        if metadata.get("use_log_transform", False):
            prediction = np.expm1(prediction)
        return {"status": "success", "predicted_price": round(float(prediction), 2)}
    except Exception as exc:
        return {"status": "error", "errors": [str(exc)]}
'''
    (MODEL_DIR / "predict.py").write_text(predict_code, encoding="utf-8")


def main() -> None:
    df = engineer_features(load_data())
    log_target = use_log_target(df["Price"])

    X = df.drop(columns=["Price"])
    y = np.log1p(df["Price"]) if log_target else df["Price"]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    plot_eda(df)

    X_train, X_test, y_train, y_test = make_split(X, y)

    base = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")),
        ]
    )
    base, baseline_rmse, baseline_r2 = train_baseline(base, X_train, y_train)
    tuned, best_params, tuned_rmse = tune_model(
        Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")),
            ]
        ),
        X_train,
        y_train,
    )
    make_learning_curves(tuned, X_train, y_train)
    metrics = evaluate_model(tuned, X_train, y_train, X_test, y_test, log_target)
    feature_df = plot_feature_importance(tuned, X_train, y_train)
    save_bundle(tuned, log_target, best_params, metrics, list(feature_df["Feature"]))

    print("\nXGBoost training complete")
    print(f"Baseline CV RMSE: {baseline_rmse:.4f}, R2: {baseline_r2:.4f}")
    print(f"Tuned CV RMSE: {tuned_rmse:.4f}")
    print(f"Test R2: {metrics['test_r2']:.4f}")


if __name__ == "__main__":
    main()
