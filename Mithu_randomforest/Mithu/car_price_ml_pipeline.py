"""
Production-Ready Machine Learning Pipeline for Used Car Price Prediction
=========================================================================
Author: ML Engineer
Date: 2026-03-30
Description: End-to-end pipeline using Random Forest with anti-overfitting controls

This script covers:
- Phase 1: Exploratory Data Analysis
- Phase 2: Preprocessing Pipeline
- Phase 3: Model Training with Anti-Overfitting Controls
- Phase 4: Model Evaluation
- Phase 5: Production Readiness
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint, uniform

from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, 
    StratifiedKFold, learning_curve
)
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance

import joblib
import json
from datetime import datetime
import os

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_status(message: str) -> None:
    """Print a status message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ✓ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  WARNING: {message}")


def print_diagnostic(condition: bool, metric_name: str, value: float, 
                     threshold: float, suggestion: str) -> None:
    """Print diagnostic warning if condition is met."""
    if condition:
        print_warning(f"{metric_name} = {value:.4f} (threshold: {threshold})")
        print(f"   💡 SUGGESTION: {suggestion}")


# =============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS
# =============================================================================

def load_and_explore_data(filepath: str) -> pd.DataFrame:
    """Load CSV and perform exploratory data analysis."""
    print_section("PHASE 1: EXPLORATORY DATA ANALYSIS")
    
    # Load data
    df = pd.read_csv(filepath)
    print_status(f"Data loaded successfully")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display column info
    print("\n📊 Column Data Types:")
    print("-" * 40)
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        print(f"   {col}: {dtype} ({nunique} unique)")
    
    # Statistical summary for numerical columns
    print("\n📈 Statistical Summary (Numerical Columns):")
    print("-" * 40)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe().round(2).to_string())
    
    return df


def analyze_target_distribution(df: pd.DataFrame, target_col: str = 'Price') -> tuple:
    """Analyze target variable distribution and check for skewness."""
    print("\n📊 Target Variable Analysis (Price):")
    print("-" * 40)
    
    price = df[target_col]
    
    # Basic statistics
    print(f"   Mean:     {price.mean():.2f}")
    print(f"   Median:   {price.median():.2f}")
    print(f"   Std Dev:  {price.std():.2f}")
    print(f"   Min:      {price.min():.2f}")
    print(f"   Max:      {price.max():.2f}")
    
    # Skewness analysis
    skewness = price.skew()
    print(f"\n   Skewness: {skewness:.3f}")
    
    # Determine if log transform is needed
    use_log_transform = abs(skewness) > 0.75
    
    if use_log_transform:
        log_price = np.log1p(price)
        log_skewness = log_price.skew()
        print(f"   ➡️  High skewness detected. Log-transform recommended.")
        print(f"   Log-transformed skewness: {log_skewness:.3f}")
    else:
        print(f"   ➡️  Skewness within acceptable range. No transform needed.")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original distribution
    axes[0].hist(price, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Price Distribution (Original)')
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(price.mean(), color='red', linestyle='--', label=f'Mean: {price.mean():.1f}')
    axes[0].axvline(price.median(), color='green', linestyle='--', label=f'Median: {price.median():.1f}')
    axes[0].legend()
    
    # Log-transformed distribution
    axes[1].hist(np.log1p(price), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title('Price Distribution (Log-transformed)')
    axes[1].set_xlabel('Log(Price + 1)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'price_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_status("Saved: price_distribution.png")
    
    return use_log_transform, skewness


def analyze_features(df: pd.DataFrame) -> dict:
    """Analyze features and identify candidates for removal."""
    print("\n📋 Feature Analysis:")
    print("-" * 40)
    
    removal_candidates = {}
    
    # Check for ID-like columns (usually have unique values close to row count)
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9 and df[col].dtype == 'object':
            removal_candidates[col] = f"ID-like column (unique ratio: {unique_ratio:.2%})"
    
    # Check for near-zero variance
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'Price':
            continue
        var = df[col].var()
        unique_count = df[col].nunique()
        if unique_count <= 1:
            removal_candidates[col] = f"Zero/near-zero variance (unique values: {unique_count})"
    
    # Check for constant categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() == 1:
            removal_candidates[col] = "Constant value (1 unique value)"
    
    # Check for high missingness
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct > 0.4:
            removal_candidates[col] = f"High missingness ({missing_pct:.1%})"
    
    # Print findings
    if removal_candidates:
        print("   ❌ Columns flagged for removal:")
        for col, reason in removal_candidates.items():
            print(f"      - {col}: {reason}")
    else:
        print("   ✓ No columns flagged for automatic removal")
    
    # Analyze categorical cardinality
    print("\n   📊 Categorical Column Cardinality:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        nunique = df[col].nunique()
        print(f"      - {col}: {nunique} unique values")
    
    return removal_candidates


def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze outliers using IQR method and create boxplots."""
    print("\n📊 Outlier Analysis (IQR Method):")
    print("-" * 40)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = []
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_pct = len(outliers) / len(df) * 100
        
        outlier_summary.append({
            'Column': col,
            'Outliers': len(outliers),
            'Outlier %': f"{outlier_pct:.1f}%",
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        })
        
        if len(outliers) > 0:
            print(f"   {col}: {len(outliers)} outliers ({outlier_pct:.1f}%)")
    
    # Create boxplots for key numerical features
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    plot_cols = ['Price', 'Engine (cc)', 'Millage(KM)', 'Car_Age']
    plot_cols = [c for c in plot_cols if c in df.columns]
    
    for i, col in enumerate(plot_cols):
        if i < len(axes):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_ylabel(col)
    
    # Hide unused subplots
    for j in range(len(plot_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_status("Saved: outlier_boxplots.png")
    
    return pd.DataFrame(outlier_summary)


def analyze_correlations(df: pd.DataFrame, target_col: str = 'Price') -> pd.DataFrame:
    """Analyze correlation between features and target."""
    print("\n📊 Feature Correlation with Price:")
    print("-" * 40)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numerical_cols].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    
    print("\n   Feature Correlations (sorted by absolute value):")
    for feat, corr in correlations.items():
        bar = '█' * int(abs(corr) * 20)
        sign = '+' if corr > 0 else '-'
        print(f"   {feat:25s} {sign}{abs(corr):.3f} {bar}")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_status("Saved: correlation_heatmap.png")
    
    return correlations


# =============================================================================
# PHASE 2: PREPROCESSING PIPELINE
# =============================================================================

def preprocess_data(df: pd.DataFrame, use_log_transform: bool = False) -> tuple:
    """Preprocess data and engineer features."""
    print_section("PHASE 2: PREPROCESSING PIPELINE")
    
    df_processed = df.copy()
    
    # 1. Handle special string values that represent missing data
    print_status("Handling special missing value indicators...")
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = df_processed[col].replace('Not_Available', np.nan)
    
    # 2. Identify columns to drop (based on EDA findings)
    print("\n   Columns to drop with justification:")
    cols_to_drop = []
    
    # Model column - too high cardinality (likely 1000+ unique values)
    # and overlaps with Brand info; would cause dimensionality explosion
    if 'Model' in df_processed.columns:
        model_cardinality = df_processed['Model'].nunique()
        print(f"   - Model: High cardinality ({model_cardinality} unique values), "
              f"will cause dimensionality explosion with OneHot encoding")
        cols_to_drop.append('Model')
    
    # Town - high cardinality location data with limited predictive value
    if 'Town' in df_processed.columns:
        town_cardinality = df_processed['Town'].nunique()
        print(f"   - Town: High cardinality location ({town_cardinality} unique values), "
              f"limited predictive value for price")
        cols_to_drop.append('Town')
    
    # Condition - if all values are same (e.g., all "USED")
    if 'Condition' in df_processed.columns:
        if df_processed['Condition'].nunique() == 1:
            print(f"   - Condition: Single constant value '{df_processed['Condition'].iloc[0]}', "
                  f"no variance to learn from")
            cols_to_drop.append('Condition')
    
    df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')
    print_status(f"Dropped {len(cols_to_drop)} columns")
    
    # 3. Convert binary feature columns that contain 1/0/Not_Available
    print_status("Converting binary feature columns...")
    binary_cols = ['AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW']
    for col in binary_cols:
        if col in df_processed.columns:
            # Convert to numeric (1, 0, or NaN for Not_Available)
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # 4. Feature Engineering
    print_status("Engineering new features...")
    
    # Mileage per year (usage intensity indicator)
    if 'Millage(KM)' in df_processed.columns and 'Car_Age' in df_processed.columns:
        df_processed['Mileage_Per_Year'] = df_processed['Millage(KM)'] / (df_processed['Car_Age'] + 1)
        print("   + Created: Mileage_Per_Year (usage intensity)")
    
    # Engine size category
    if 'Engine (cc)' in df_processed.columns:
        df_processed['Engine_Category'] = pd.cut(
            df_processed['Engine (cc)'],
            bins=[0, 1000, 1500, 2000, 3000, np.inf],
            labels=['Small', 'Medium', 'Large', 'XLarge', 'Luxury']
        )
        print("   + Created: Engine_Category (binned engine size)")
    
    # Features comfort score (sum of available features)
    if all(col in df_processed.columns for col in binary_cols):
        df_processed['Comfort_Score'] = df_processed[binary_cols].sum(axis=1)
        print("   + Created: Comfort_Score (sum of comfort features)")
    
    # 5. Separate target and features
    target_col = 'Price'
    if use_log_transform:
        y = np.log1p(df_processed[target_col])
        print_status("Applied log-transform to target variable (Price)")
    else:
        y = df_processed[target_col]
    
    X = df_processed.drop(columns=[target_col])
    
    # 6. Identify column types for preprocessing
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n   Feature Summary:")
    print(f"   - Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"   - Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"   - Total features: {len(numerical_cols) + len(categorical_cols)}")
    
    # 7. Analyze missingness
    print("\n   Missingness Analysis:")
    for col in X.columns:
        missing_count = X[col].isnull().sum()
        missing_pct = missing_count / len(X) * 100
        if missing_count > 0:
            print(f"   - {col}: {missing_count} missing ({missing_pct:.1f}%)")
    
    return X, y, numerical_cols, categorical_cols, use_log_transform


def build_preprocessing_pipeline(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    print_status("Building preprocessing pipeline...")
    
    # Numerical pipeline: median imputation (no scaling needed for Random Forest)
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Categorical pipeline: impute with 'Unknown', then ordinal encode
    # Using OrdinalEncoder for efficiency (OneHot would explode dimensions for high-cardinality)
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    
    print_status("Preprocessing pipeline built successfully")
    print(f"   - Numerical: Median imputation for {len(numerical_cols)} features")
    print(f"   - Categorical: Unknown imputation + Ordinal encoding for {len(categorical_cols)} features")
    
    return preprocessor


# =============================================================================
# PHASE 3: MODEL TRAINING
# =============================================================================

def create_stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
    """Create stratified train/test split based on price quantiles."""
    print_section("PHASE 3: MODEL TRAINING")
    print_status("Creating stratified train/test split...")
    
    # Create price quantile bins for stratification
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y_binned
    )
    
    print(f"   Training set: {len(X_train)} samples ({100-test_size*100:.0f}%)")
    print(f"   Test set: {len(X_test)} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(pipeline: Pipeline, X_train: pd.DataFrame, 
                         y_train: pd.Series) -> tuple:
    """Train baseline Random Forest and evaluate with cross-validation."""
    print_status("Training baseline Random Forest model...")
    
    # 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Create price bins for stratified CV
    y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
    
    # Cross-validation scores (negative MSE, we'll convert to RMSE)
    cv_scores_neg_mse = cross_val_score(
        pipeline, X_train, y_train, 
        cv=cv.split(X_train, y_binned), 
        scoring='neg_mean_squared_error'
    )
    cv_scores_r2 = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv.split(X_train, y_binned),
        scoring='r2'
    )
    
    # Convert negative MSE to RMSE
    cv_rmse = np.sqrt(-cv_scores_neg_mse)
    
    print("\n   📊 5-Fold Cross-Validation Results (Baseline):")
    print("-" * 50)
    print(f"   RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"   R²:   {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
    print(f"   Fold RMSE scores: {[f'{s:.4f}' for s in cv_rmse]}")
    
    # Fit on full training data
    pipeline.fit(X_train, y_train)
    print_status("Baseline model fitted on training data")
    
    return pipeline, cv_rmse.mean(), cv_scores_r2.mean()


def perform_hyperparameter_tuning(pipeline: Pipeline, X_train: pd.DataFrame,
                                  y_train: pd.Series) -> tuple:
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    print_status("Performing hyperparameter tuning (RandomizedSearchCV)...")
    
    # Define parameter grid with regularization focus
    param_distributions = {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': [5, 10, 15, 20, 25, 30, None],
        'model__min_samples_split': randint(2, 20),
        'model__min_samples_leaf': randint(1, 15),
        'model__max_features': ['sqrt', 'log2', 0.5, 0.7, None]
    }
    
    # Stratified CV for tuning
    y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        cv=cv.split(X_train, y_binned),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    print("   Searching 50 parameter combinations across 5 folds...")
    search.fit(X_train, y_train)
    
    # Best parameters
    best_params = search.best_params_
    best_rmse = np.sqrt(-search.best_score_)
    
    print("\n   📊 Hyperparameter Tuning Results:")
    print("-" * 50)
    print(f"   Best CV RMSE: {best_rmse:.4f}")
    print("\n   Best Parameters:")
    for param, value in best_params.items():
        clean_param = param.replace('model__', '')
        print(f"      - {clean_param}: {value}")
    
    return search.best_estimator_, best_params, best_rmse


def plot_learning_curves(estimator: Pipeline, X_train: pd.DataFrame, 
                         y_train: pd.Series) -> None:
    """Plot learning curves to detect overfitting."""
    print_status("Generating learning curves...")
    
    y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train,
        cv=cv.split(X_train, y_binned),
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Convert to RMSE
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    
    # Calculate mean and std
    train_mean = train_rmse.mean(axis=1)
    train_std = train_rmse.std(axis=1)
    val_mean = val_rmse.mean(axis=1)
    val_std = val_rmse.std(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training RMSE')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation RMSE')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curves (Training vs Validation Error)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_status("Saved: learning_curves.png")
    
    # Check for overfitting
    final_gap = train_mean[-1] - val_mean[-1]
    print(f"   Final train-val RMSE gap: {abs(final_gap):.4f}")
    
    if abs(final_gap) > 0.5 * val_mean[-1]:
        print_warning("Large gap between training and validation error detected!")
        print("   💡 SUGGESTION: Increase regularization (min_samples_leaf, max_depth)")


# =============================================================================
# PHASE 4: MODEL EVALUATION
# =============================================================================

def evaluate_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series, 
                   use_log_transform: bool) -> dict:
    """Comprehensive model evaluation on test set."""
    print_section("PHASE 4: MODEL EVALUATION")
    
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # If log-transformed, convert back for interpretable metrics
    if use_log_transform:
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
        y_train_pred_actual = np.expm1(y_train_pred)
        y_test_pred_actual = np.expm1(y_test_pred)
    else:
        y_train_actual = y_train
        y_test_actual = y_test
        y_train_pred_actual = y_train_pred
        y_test_pred_actual = y_test_pred
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual)),
        'train_mae': mean_absolute_error(y_train_actual, y_train_pred_actual),
        'train_r2': r2_score(y_train_actual, y_train_pred_actual),
        'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual)),
        'test_mae': mean_absolute_error(y_test_actual, y_test_pred_actual),
        'test_mape': mean_absolute_percentage_error(y_test_actual, y_test_pred_actual),
        'test_r2': r2_score(y_test_actual, y_test_pred_actual)
    }
    
    print("\n   📊 Model Performance Metrics:")
    print("-" * 50)
    print(f"\n   Training Set:")
    print(f"      RMSE: {metrics['train_rmse']:.4f}")
    print(f"      MAE:  {metrics['train_mae']:.4f}")
    print(f"      R²:   {metrics['train_r2']:.4f}")
    
    print(f"\n   Test Set (Held-out):")
    print(f"      RMSE: {metrics['test_rmse']:.4f}")
    print(f"      MAE:  {metrics['test_mae']:.4f}")
    print(f"      MAPE: {metrics['test_mape']:.2%}")
    print(f"      R²:   {metrics['test_r2']:.4f}")
    
    # Overfitting detection
    r2_gap = metrics['train_r2'] - metrics['test_r2']
    rmse_ratio = metrics['train_rmse'] / metrics['test_rmse'] if metrics['test_rmse'] > 0 else 0
    
    print(f"\n   📈 Overfitting Analysis:")
    print(f"      Train-Test R² Gap: {r2_gap:.4f}")
    print(f"      Train/Test RMSE Ratio: {rmse_ratio:.4f}")
    
    # Diagnostic warnings
    print_diagnostic(
        metrics['test_r2'] < 0.7, 
        "Test R²", metrics['test_r2'], 0.7,
        "Model may lack predictive power. Consider additional features or feature engineering."
    )
    
    print_diagnostic(
        r2_gap > 0.1,
        "Train-Test R² Gap", r2_gap, 0.1,
        "Significant overfitting detected. Increase regularization or reduce model complexity."
    )
    
    # Plot predictions vs actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_test_actual, y_test_pred_actual, alpha=0.5, edgecolor='k', linewidth=0.5)
    max_val = max(y_test_actual.max(), y_test_pred_actual.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price')
    axes[0].set_ylabel('Predicted Price')
    axes[0].set_title('Predicted vs Actual Prices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    residuals = y_test_actual - y_test_pred_actual
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residual Distribution (Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_status("Saved: model_evaluation.png")
    
    return metrics


def plot_feature_importance(pipeline: Pipeline, X_train: pd.DataFrame, 
                            y_train: pd.Series, feature_names: list) -> pd.DataFrame:
    """Generate feature importance plot using permutation importance."""
    print_status("Computing permutation feature importance...")
    
    # Get preprocessor and model
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    # Transform features
    X_train_transformed = preprocessor.transform(X_train)
    
    # Permutation importance (more reliable than impurity-based)
    perm_importance = permutation_importance(
        model, X_train_transformed, y_train,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("\n   📊 Top 10 Feature Importances (Permutation):")
    print("-" * 50)
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['Importance'] / importance_df['Importance'].max() * 20)
        print(f"   {row['Feature']:25s} {row['Importance']:.4f} {bar}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)
    
    plt.barh(range(top_n), top_features['Importance'], 
             xerr=top_features['Std'], align='center', alpha=0.8)
    plt.yticks(range(top_n), top_features['Feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Top Feature Importances (Permutation Method)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_status("Saved: feature_importance.png")
    
    return importance_df


# =============================================================================
# PHASE 5: PRODUCTION READINESS
# =============================================================================

def save_model(pipeline: Pipeline, feature_names: list, use_log_transform: bool,
               best_params: dict, metrics: dict) -> None:
    """Save the trained pipeline and metadata."""
    print_section("PHASE 5: PRODUCTION READINESS")
    
    # Save pipeline
    model_path = os.path.join(OUTPUT_DIR, 'car_price_model.joblib')
    joblib.dump(pipeline, model_path)
    print_status(f"Saved model: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'feature_names': feature_names,
        'use_log_transform': use_log_transform,
        'best_params': {k.replace('model__', ''): v for k, v in best_params.items()},
        'metrics': metrics,
        'created_at': datetime.now().isoformat(),
        'random_state': RANDOM_STATE
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print_status(f"Saved metadata: {metadata_path}")


def create_prediction_function() -> None:
    """Create a predict function for production use."""
    
    predict_code = '''
"""
Production Prediction Function for Car Price Model
==================================================
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

# Load model and metadata
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_DIR, 'car_price_model.joblib'))

with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

REQUIRED_FEATURES = [
    'Brand', 'Engine (cc)', 'Gear', 'Fuel Type', 'Millage(KM)',
    'Leasing', 'AIR CONDITION', 'POWER STEERING', 'POWER MIRROR',
    'POWER WINDOW', 'Car_Age'
]


def validate_input(input_dict: dict) -> tuple:
    """Validate input dictionary and return errors if any."""
    errors = []
    
    # Check for required features
    for feature in REQUIRED_FEATURES:
        if feature not in input_dict:
            errors.append(f"Missing required feature: '{feature}'")
    
    # Type validation
    numeric_features = ['Engine (cc)', 'Millage(KM)', 'Car_Age']
    for feat in numeric_features:
        if feat in input_dict:
            try:
                float(input_dict[feat])
            except (ValueError, TypeError):
                errors.append(f"'{feat}' must be numeric, got: {type(input_dict[feat]).__name__}")
    
    # Value range validation
    if 'Car_Age' in input_dict:
        try:
            age = float(input_dict['Car_Age'])
            if age < 0 or age > 100:
                errors.append(f"'Car_Age' should be between 0-100, got: {age}")
        except:
            pass
    
    if 'Millage(KM)' in input_dict:
        try:
            mileage = float(input_dict['Millage(KM)'])
            if mileage < 0:
                errors.append(f"'Millage(KM)' cannot be negative, got: {mileage}")
        except:
            pass
    
    return len(errors) == 0, errors


def predict(input_dict: dict) -> dict:
    """
    Predict car price from raw input dictionary.
    
    Parameters:
    -----------
    input_dict : dict
        Dictionary containing car features. Required keys:
        - Brand: str (e.g., 'TOYOTA', 'BMW')
        - Engine (cc): float (e.g., 1500.0)
        - Gear: str ('Automatic' or 'Manual')
        - Fuel Type: str ('Petrol', 'Diesel', 'Hybrid', 'Electric')
        - Millage(KM): float (e.g., 50000.0)
        - Leasing: str or int ('0', '1', or 'Ongoing Lease')
        - AIR CONDITION: int (0 or 1)
        - POWER STEERING: int (0 or 1)
        - POWER MIRROR: int (0 or 1)
        - POWER WINDOW: int (0 or 1)
        - Car_Age: int (e.g., 5)
    
    Returns:
    --------
    dict: {'predicted_price': float, 'status': 'success'} or
          {'error': str, 'status': 'error'}
    
    Example:
    --------
    >>> result = predict({
    ...     'Brand': 'TOYOTA',
    ...     'Engine (cc)': 1500.0,
    ...     'Gear': 'Automatic',
    ...     'Fuel Type': 'Petrol',
    ...     'Millage(KM)': 50000.0,
    ...     'Leasing': 0,
    ...     'AIR CONDITION': 1,
    ...     'POWER STEERING': 1,
    ...     'POWER MIRROR': 1,
    ...     'POWER WINDOW': 1,
    ...     'Car_Age': 5
    ... })
    >>> print(result)
    {'predicted_price': 85.5, 'status': 'success'}
    """
    
    # Validate input
    is_valid, errors = validate_input(input_dict)
    if not is_valid:
        return {
            'status': 'error',
            'errors': errors
        }
    
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_dict])
        
        # Engineer features (same as training)
        df['Mileage_Per_Year'] = df['Millage(KM)'] / (df['Car_Age'] + 1)
        df['Engine_Category'] = pd.cut(
            df['Engine (cc)'],
            bins=[0, 1000, 1500, 2000, 3000, np.inf],
            labels=['Small', 'Medium', 'Large', 'XLarge', 'Luxury']
        )
        df['Comfort_Score'] = (
            df['AIR CONDITION'].astype(float) + 
            df['POWER STEERING'].astype(float) + 
            df['POWER MIRROR'].astype(float) + 
            df['POWER WINDOW'].astype(float)
        )
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # If log-transformed during training, convert back
        if metadata.get('use_log_transform', False):
            prediction = np.expm1(prediction)
        
        return {
            'status': 'success',
            'predicted_price': round(float(prediction), 2)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'errors': [str(e)]
        }


# Example usage
if __name__ == '__main__':
    # Test prediction
    test_input = {
        'Brand': 'TOYOTA',
        'Engine (cc)': 1500.0,
        'Gear': 'Automatic',
        'Fuel Type': 'Petrol',
        'Millage(KM)': 50000.0,
        'Leasing': 0,
        'AIR CONDITION': 1,
        'POWER STEERING': 1,
        'POWER MIRROR': 1,
        'POWER WINDOW': 1,
        'Car_Age': 5
    }
    
    result = predict(test_input)
    print(f"Prediction Result: {result}")
'''
    
    predict_path = os.path.join(OUTPUT_DIR, 'predict.py')
    with open(predict_path, 'w') as f:
        f.write(predict_code)
    print_status(f"Created prediction function: {predict_path}")


def print_model_summary(best_params: dict, cv_rmse: float, metrics: dict, 
                        feature_count: int) -> None:
    """Print final model summary card."""
    print("\n" + "═" * 70)
    print("                    📋 MODEL SUMMARY CARD")
    print("═" * 70)
    
    print("\n   🔧 Best Hyperparameters:")
    for param, value in best_params.items():
        clean_param = param.replace('model__', '')
        print(f"      • {clean_param}: {value}")
    
    print(f"\n   📊 Performance Scores:")
    print(f"      • Cross-Validation RMSE: {cv_rmse:.4f}")
    print(f"      • Test Set RMSE: {metrics['test_rmse']:.4f}")
    print(f"      • Test Set MAE: {metrics['test_mae']:.4f}")
    print(f"      • Test Set MAPE: {metrics['test_mape']:.2%}")
    print(f"      • Test Set R²: {metrics['test_r2']:.4f}")
    
    print(f"\n   📈 Model Info:")
    print(f"      • Algorithm: Random Forest Regressor")
    print(f"      • Feature Count: {feature_count}")
    print(f"      • Random State: {RANDOM_STATE}")
    
    print(f"\n   📁 Saved Files:")
    print(f"      • car_price_model.joblib (trained pipeline)")
    print(f"      • model_metadata.json (configuration)")
    print(f"      • predict.py (production prediction function)")
    print(f"      • *.png (visualization plots)")
    
    print("\n" + "═" * 70)
    
    # Final assessment
    if metrics['test_r2'] >= 0.85:
        print("   ✅ Model demonstrates EXCELLENT predictive performance!")
    elif metrics['test_r2'] >= 0.7:
        print("   ✅ Model demonstrates GOOD predictive performance.")
    else:
        print("   ⚠️  Model performance below target. Consider:")
        print("      - Additional feature engineering")
        print("      - More training data")
        print("      - Alternative algorithms (GradientBoosting, XGBoost)")
    
    print("═" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "█" * 70)
    print("    USED CAR PRICE PREDICTION - ML PIPELINE")
    print("    Random Forest with Anti-Overfitting Controls")
    print("█" * 70)
    
    # File path
    csv_path = os.path.join(OUTPUT_DIR, 'car_price_dataset_cleaned.csv')
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: CSV file not found at: {csv_path}")
        print("   Please ensure 'car_price_dataset_cleaned.csv' is in the same directory.")
        return
    
    # Phase 1: EDA
    df = load_and_explore_data(csv_path)
    use_log_transform, skewness = analyze_target_distribution(df)
    removal_candidates = analyze_features(df)
    outlier_summary = analyze_outliers(df)
    correlations = analyze_correlations(df)
    
    # Phase 2: Preprocessing
    X, y, numerical_cols, categorical_cols, use_log_transform = preprocess_data(
        df, use_log_transform
    )
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    # Create feature names list (after preprocessing)
    feature_names = numerical_cols + categorical_cols
    
    # Phase 3: Model Training
    X_train, X_test, y_train, y_test = create_stratified_split(X, y)
    
    # Build full pipeline
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    
    # Train baseline and get CV scores
    baseline_pipeline, baseline_cv_rmse, baseline_cv_r2 = train_baseline_model(
        baseline_pipeline, X_train, y_train
    )
    
    # Hyperparameter tuning
    tuned_pipeline, best_params, cv_rmse = perform_hyperparameter_tuning(
        Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        X_train, y_train
    )
    
    # Learning curves
    plot_learning_curves(tuned_pipeline, X_train, y_train)
    
    # Phase 4: Evaluation
    metrics = evaluate_model(
        tuned_pipeline, X_train, y_train, X_test, y_test, use_log_transform
    )
    importance_df = plot_feature_importance(
        tuned_pipeline, X_train, y_train, feature_names
    )
    
    # Phase 5: Production Readiness
    save_model(tuned_pipeline, feature_names, use_log_transform, best_params, metrics)
    create_prediction_function()
    print_model_summary(best_params, cv_rmse, metrics, len(feature_names))
    
    print_status("Pipeline execution completed successfully!")
    
    return tuned_pipeline, metrics


if __name__ == '__main__':
    pipeline, metrics = main()
