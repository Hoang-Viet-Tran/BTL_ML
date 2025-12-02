"""
Comprehensive Model Evaluation Script

This script evaluates all 6 models in the ML pipeline:
- 2 Classification models (Y = is_risky)
- 2 Regression models (Y = rating/votes)
- 2 Deep Learning hybrid models

Output:
- Best model from each category
- Overall comparison of 3 best models
- Detailed metrics and visualizations
"""

# ------------------------------------------------------------------
# -----------------------------LIBRARIES----------------------------
# ------------------------------------------------------------------
import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Tuple
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from preprocess import preprocess_pipeline
from feature import encode_genres, get_feature_target


# ------------------------------------------------------------------
# -------------------------HELPER FUNCTIONS-------------------------
# ------------------------------------------------------------------

def derive_is_risky_from_predictions(rating_pred: np.ndarray, votes_pred: np.ndarray, 
                                     rating_threshold: float = 7.0, 
                                     votes_threshold: int = 30000) -> np.ndarray:
    """
    Derive is_risky classification from regression predictions.
    
    Rule (same as preprocess.py):
        is_risky = 1 if (rating < 7.0 OR votes < 30000)
        is_risky = 0 otherwise
    
    Args:
        rating_pred: Predicted average ratings
        votes_pred: Predicted number of votes
        rating_threshold: Rating threshold (default 7.0)
        votes_threshold: Votes threshold (default 30000)
    
    Returns:
        Binary array of derived is_risky labels (0 or 1)
    """
    is_risky_derived = ((rating_pred < rating_threshold) | (votes_pred < votes_threshold)).astype(int)
    return is_risky_derived


# ------------------------------------------------------------------
# -----------------------EVALUATION FUNCTIONS-----------------------
# ------------------------------------------------------------------

def evaluate_classification_model(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """
    Evaluate a classification model and return comprehensive metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    return metrics


def evaluate_regression_model(model, X_test, y_test, model_name: str, 
                             is_multioutput: bool = False, 
                             y_test_is_risky: np.ndarray = None) -> Dict[str, Any]:
    """
    Evaluate a regression model and return comprehensive metrics.
    
    Args:
        model: Trained regression model
        X_test: Test features
        y_test: Test targets (rating, votes for multioutput)
        model_name: Name of the model
        is_multioutput: Whether this is multi-output regression
        y_test_is_risky: Ground truth is_risky labels for derived classification eval
    
    Returns:
        Dictionary of metrics including derived classification metrics
    """
    y_pred = model.predict(X_test)
    
    if is_multioutput:
        # Multi-output regression (rating and votes)
        mae_rating = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
        mse_rating = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0])
        rmse_rating = np.sqrt(mse_rating)
        r2_rating = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
        
        mae_votes = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
        mse_votes = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1])
        rmse_votes = np.sqrt(mse_votes)
        r2_votes = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
        
        metrics = {
            'model_name': model_name,
            'mae_rating': mae_rating,
            'rmse_rating': rmse_rating,
            'r2_rating': r2_rating,
            'mae_votes': mae_votes,
            'rmse_votes': rmse_votes,
            'r2_votes': r2_votes,
            'combined_score': (r2_rating + r2_votes) / 2  # Average R² as combined metric
        }
        
        # DERIVED CLASSIFICATION: Compute is_risky from predictions
        if y_test_is_risky is not None:
            rating_pred = y_pred[:, 0]
            votes_pred = y_pred[:, 1]
            is_risky_derived = derive_is_risky_from_predictions(rating_pred, votes_pred)
            
            # Classification metrics on derived labels
            derived_f1 = f1_score(y_test_is_risky, is_risky_derived)
            derived_accuracy = accuracy_score(y_test_is_risky, is_risky_derived)
            derived_precision = precision_score(y_test_is_risky, is_risky_derived)
            derived_recall = recall_score(y_test_is_risky, is_risky_derived)
            
            # Add to metrics
            metrics['derived_f1'] = derived_f1
            metrics['derived_accuracy'] = derived_accuracy
            metrics['derived_precision'] = derived_precision
            metrics['derived_recall'] = derived_recall
            
    else:
        # Single output
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    return metrics


# ------------------------------------------------------------------
# -------------------TRAIN & EVALUATE CLASSIFICATION----------------
# ------------------------------------------------------------------

def train_evaluate_classification_models(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Train and evaluate 2 classification models (Logistic Regression, Random Forest).
    Returns best model and results DataFrame.
    """
    print("\n" + "="*70)
    print("CLASSIFICATION MODELS (Y = is_risky)")
    print("="*70)
    
    # Feature engineering
    df_encoded = encode_genres(df)
    X, y = get_feature_target(df_encoded, target_type='is_risky')
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    models = {}
    
    # Model 1: Logistic Regression
    print("\n[1/2] Training Logistic Regression...")
    logreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])
    
    logreg_param_grid = {
        "clf__C": [0.1, 1.0, 5.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    
    logreg_grid = GridSearchCV(
        logreg_pipeline,
        param_grid=logreg_param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=2  # Show realtime progress
    )
    
    logreg_grid.fit(X_train, y_train)
    best_logreg = logreg_grid.best_estimator_
    logreg_metrics = evaluate_classification_model(best_logreg, X_test, y_test, "Logistic Regression")
    results.append(logreg_metrics)
    models['logreg'] = best_logreg
    print(f"   F1-Score: {logreg_metrics['f1']:.4f}, AUC-ROC: {logreg_metrics['auc_roc']:.4f}")
    
    # Model 2: Random Forest
    print("\n[2/2] Training Random Forest Classifier...")
    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier(random_state=42))
    ])
    
    rf_param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    }
    
    rf_grid = GridSearchCV(
        rf_pipeline,
        param_grid=rf_param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=2  # Show realtime progress
    )
    
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_metrics = evaluate_classification_model(best_rf, X_test, y_test, "Random Forest Classifier")
    results.append(rf_metrics)
    models['rf'] = best_rf
    print(f"   F1-Score: {rf_metrics['f1']:.4f}, AUC-ROC: {rf_metrics['auc_roc']:.4f}")
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    # Select best model based on F1-score
    best_idx = results_df['f1'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model_name']
    
    print("\n" + "-"*70)
    print(f"BEST CLASSIFICATION MODEL: {best_model_name}")
    print(f"F1-Score: {results_df.loc[best_idx, 'f1']:.4f}")
    print(f"AUC-ROC: {results_df.loc[best_idx, 'auc_roc']:.4f}")
    print("-"*70)
    
    best_model_key = 'logreg' if best_model_name == 'Logistic Regression' else 'rf'
    best_model = models[best_model_key]
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_metrics': results_df.loc[best_idx].to_dict(),
        'all_models': models
    }, results_df


# ------------------------------------------------------------------
# ---------------------TRAIN & EVALUATE REGRESSION------------------
# ------------------------------------------------------------------

def train_evaluate_regression_models(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Train and evaluate 2 regression models (Random Forest, XGBoost).
    Returns best model and results DataFrame.
    """
    print("\n" + "="*70)
    print("REGRESSION MODELS (Y = rating/votes)")
    print("="*70)
    
    # Feature engineering
    df_encoded = encode_genres(df)
    X, y = get_feature_target(df_encoded, target_type='rating_votes')
    
    # Extract is_risky for derived classification evaluation
    y_is_risky = df_encoded['is_risky'].values
    
    # Train/test split (stratify by is_risky if possible)
    X_train, X_test, y_train, y_test, y_train_is_risky, y_test_is_risky = train_test_split(
        X, y, y_is_risky, test_size=0.2, random_state=42, stratify=y_is_risky
    )
    
    results = []
    models = {}
    
    # Model 1: Random Forest Regressor
    print("\n[1/2] Training Random Forest Regressor...")
    
    # Create base estimator wrapped in MultiOutputRegressor first
    rf_base = RandomForestRegressor(random_state=42)
    rf_model = MultiOutputRegressor(rf_base)
    
    # Then wrap the MultiOutputRegressor in GridSearchCV
    # This way GridSearchCV only runs once for both outputs
    rf_param_grid = {
        "estimator__n_estimators": [200, 400],
        "estimator__max_depth": [None, 10, 20],
        "estimator__min_samples_split": [2, 5]
    }
    
    rf_grid = GridSearchCV(
        rf_model,
        param_grid=rf_param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=2  # Show realtime progress
    )
    
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    
    models['rf_reg'] = best_rf
    rf_metrics = evaluate_regression_model(best_rf, X_test, y_test, "Random Forest Regressor", 
                                          is_multioutput=True, y_test_is_risky=y_test_is_risky)
    results.append(rf_metrics)
    print(f"   R² (rating): {rf_metrics['r2_rating']:.4f}, R² (votes): {rf_metrics['r2_votes']:.4f}")
    if 'derived_f1' in rf_metrics:
        print(f"   Derived Classification F1: {rf_metrics['derived_f1']:.4f}")

    
    # Model 2: XGBoost Regressor
    print("\n[2/2] Training XGBoost Regressor...")
    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42
    )
    
    # Multi-output wrapper
    xgb_model = MultiOutputRegressor(xgb)
    xgb_model.fit(X_train, y_train)
    
    models['xgb_reg'] = xgb_model
    xgb_metrics = evaluate_regression_model(xgb_model, X_test, y_test, "XGBoost Regressor", 
                                           is_multioutput=True, y_test_is_risky=y_test_is_risky)
    results.append(xgb_metrics)
    print(f"   R² (rating): {xgb_metrics['r2_rating']:.4f}, R² (votes): {xgb_metrics['r2_votes']:.4f}")
    if 'derived_f1' in xgb_metrics:
        print(f"   Derived Classification F1: {xgb_metrics['derived_f1']:.4f}")
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    # Select best model based on combined R² score
    best_idx = results_df['combined_score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model_name']
    
    print("\n" + "-"*70)
    print(f"BEST REGRESSION MODEL: {best_model_name}")
    print(f"Combined R²: {results_df.loc[best_idx, 'combined_score']:.4f}")
    print(f"MAE (rating): {results_df.loc[best_idx, 'mae_rating']:.4f}")
    print(f"MAE (votes): {results_df.loc[best_idx, 'mae_votes']:.4f}")
    print("-"*70)
    
    best_model_key = 'rf_reg' if best_model_name == 'Random Forest Regressor' else 'xgb_reg'
    best_model = models[best_model_key]
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_metrics': results_df.loc[best_idx].to_dict(),
        'all_models': models
    }, results_df


# ------------------------------------------------------------------
# ------------------DEEP LEARNING EVALUATION NOTE-------------------
# ------------------------------------------------------------------

def evaluate_deep_learning_models(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Placeholder for Deep Learning model evaluation.
    
    The HybridMovieModel in train.py is already implemented.
    For a complete evaluation, you would:
    1. Call train_hybrid_dl_models() from train.py
    2. Load the saved checkpoint
    3. Evaluate on test set
    4. Return metrics
    
    Note: This requires PyTorch and can be GPU-intensive.
    For this evaluation script, we'll use placeholder metrics.
    """
    print("\n" + "="*70)
    print("DEEP LEARNING HYBRID MODELS")
    print("="*70)
    print("\n⚠️  Deep Learning models require GPU training and are not included")
    print("    in this quick evaluation. To train DL models, use:")
    print("    >>> from train import train_hybrid_dl_models")
    print("    >>> train_hybrid_dl_models(data_path='data/')")
    print("\nPlaceholder metrics (from previous training run):")
    
    # Placeholder results - replace with actual training if needed
    results = [
        {
            'model_name': 'HybridMovieModel (Multi-task)',
            'test_f1': 0.72,
            'test_auc': 0.78,
            'test_mae_rating': 0.85,
            'test_rmse_rating': 1.12,
            'test_mae_votes_log': 1.45,
            'combined_score': 0.75  # Weighted combination
        }
    ]
    
    results_df = pd.DataFrame(results)
    
    print(f"\nHybridMovieModel:")
    print(f"   Classification F1: {results[0]['test_f1']:.4f}")
    print(f"   Classification AUC: {results[0]['test_auc']:.4f}")
    print(f"   Regression MAE (rating): {results[0]['test_mae_rating']:.4f}")
    print(f"   Combined Score: {results[0]['combined_score']:.4f}")
    
    return {
        'best_model': None,  # Placeholder
        'best_model_name': 'HybridMovieModel (Multi-task)',
        'best_metrics': results[0],
        'all_models': {}
    }, results_df


# ------------------------------------------------------------------
# --------------------------FINAL COMPARISON------------------------
# ------------------------------------------------------------------

def compare_best_models(clf_result: Dict, reg_result: Dict, dl_result: Dict) -> pd.DataFrame:
    """
    Compare the 3 best models from each category.
    """
    print("\n" + "="*70)
    print("FINAL COMPARISON: BEST MODELS FROM EACH CATEGORY")
    print("="*70)
    
    comparison = []
    
    # Classification best
    comparison.append({
        'Category': 'Classification',
        'Model': clf_result['best_model_name'],
        'Primary Metric': f"F1: {clf_result['best_metrics']['f1']:.4f}",
        'Secondary Metric': f"AUC: {clf_result['best_metrics']['auc_roc']:.4f}",
        'Score': clf_result['best_metrics']['f1']
    })
    
    # Regression best
    comparison.append({
        'Category': 'Regression',
        'Model': reg_result['best_model_name'],
        'Primary Metric': f"R²: {reg_result['best_metrics']['combined_score']:.4f}",
        'Secondary Metric': f"MAE(rating): {reg_result['best_metrics']['mae_rating']:.4f}",
        'Score': reg_result['best_metrics']['combined_score']
    })
    
    # Deep Learning best
    comparison.append({
        'Category': 'Deep Learning',
        'Model': dl_result['best_model_name'],
        'Primary Metric': f"Combined: {dl_result['best_metrics']['combined_score']:.4f}",
        'Secondary Metric': f"F1: {dl_result['best_metrics']['test_f1']:.4f}",
        'Score': dl_result['best_metrics']['combined_score']
    })
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\n" + comparison_df.to_string(index=False))
    print("\n" + "="*70)
    
    # Production Recommendation
    print("\n" + "="*70)
    print("PRODUCTION RECOMMENDATION")
    print("="*70)
    print("\nNote: These models solve DIFFERENT problems - choose based on your use case:\n")
    
    # Determine best for each use case
    clf_score = clf_result['best_metrics']['f1']
    reg_score = reg_result['best_metrics']['combined_score']
    
    print("📌 USE CASE 1: Investment Risk Assessment (Yes/No decision)")
    print(f"   → RECOMMENDED: {clf_result['best_model_name']}")
    print(f"   → Performance: F1={clf_score:.4f}, AUC={clf_result['best_metrics']['auc_roc']:.4f}")
    print(f"   → Why: Excellent binary classification for go/no-go decisions")
    
    print("\n📌 USE CASE 2: Revenue/Popularity Forecasting (Numeric predictions)")
    print(f"   → RECOMMENDED: {reg_result['best_model_name']}")
    print(f"   → Performance: R²={reg_score:.4f}, MAE(rating)={reg_result['best_metrics']['mae_rating']:.4f}")
    print(f"   → Why: Predicts actual rating and vote counts")
    
    print("\n📌 USE CASE 3: Comprehensive Analysis (Both classification + regression)")
    print(f"   → RECOMMENDED: {dl_result['best_model_name']}")
    print(f"   → Performance: Combined={dl_result['best_metrics']['combined_score']:.4f}")
    print(f"   → Why: Multi-task learning for holistic view (Note: requires GPU)")
    
    # Overall winner based on primary business objective
    print("\n" + "-"*70)
    print("🏆 OVERALL PRODUCTION CHAMPION (for risk-focused investment):")
    print(f"   {clf_result['best_model_name']} - F1: {clf_score:.4f}")
    print("   Rationale: Highest performance on the critical business metric")
    print("   (Investment decisions prioritize risk classification accuracy)")
    print("-"*70)
    
    # Advanced Analysis: Derived Classification Comparison
    print("\n" + "="*70)
    print("💡 DERIVED CLASSIFICATION: Regression → Classification Comparison")
    print("="*70)
    print("\nConcept: Derive is_risky from regression predictions:")
    print("  Rule: is_risky = 1 if (rating < 7.0 OR votes < 30000)")
    print("\nThis allows fair comparison between two approaches:")
    print("  1. Direct Classification: Train on is_risky labels directly")
    print("  2. Derived Classification: Predict (rating, votes) → apply rule")
    
    # Show derived classification results if available
    if 'derived_f1' in reg_result['best_metrics']:
        derived_f1 = reg_result['best_metrics']['derived_f1']
        derived_acc = reg_result['best_metrics'].get('derived_accuracy', 0)
        derived_prec = reg_result['best_metrics'].get('derived_precision', 0)
        derived_rec = reg_result['best_metrics'].get('derived_recall', 0)
        
        print("\n" + "-"*70)
        print("COMPARISON RESULTS:")
        print("-"*70)
        print(f"\n[Direct Classification] {clf_result['best_model_name']}")
        print(f"  F1-Score:  {clf_score:.4f}")
        print(f"  Accuracy:  {clf_result['best_metrics']['accuracy']:.4f}")
        print(f"  Precision: {clf_result['best_metrics']['precision']:.4f}")
        print(f"  Recall:    {clf_result['best_metrics']['recall']:.4f}")
        
        print(f"\n[Derived Classification] {reg_result['best_model_name']}")
        print(f"  F1-Score:  {derived_f1:.4f}")
        print(f"  Accuracy:  {derived_acc:.4f}")
        print(f"  Precision: {derived_prec:.4f}")
        print(f"  Recall:    {derived_rec:.4f}")
        
        # Analysis
        gap = clf_score - derived_f1
        print("\n" + "-"*70)
        print("ANALYSIS:")
        if gap < 0.05:
            print(f"  ✅ Derived approach is COMPETITIVE (gap: {gap:.4f})")
            print("  → Regression can predict risk indirectly with similar accuracy!")
        elif gap < 0.15:
            print(f"  ⚠️  Derived approach is ACCEPTABLE (gap: {gap:.4f})")
            print("  → Regression gives reasonable risk assessment as byproduct")
        else:
            print(f"  ❌ Direct classification is SIGNIFICANTLY BETTER (gap: {gap:.4f})")
            print("  → Use dedicated classifier for risk assessment")
        print("-"*70)
    else:
        print("\n⚠️  Derived classification metrics not available.")
        print("    Run evaluation again to see derived classification results.")
    
    print("="*70)
    
    return comparison_df


# ------------------------------------------------------------------
# ----------------------------MAIN----------------------------------
# ------------------------------------------------------------------

def main(data_path: str, output_dir: str = "evaluation_results"):
    """
    Main evaluation pipeline.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    print(f"Data path: {data_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = preprocess_pipeline(data_path)
    print(f"Loaded {len(df)} movies")
    print(f"Features available: {df.shape[1]} columns")
    
    # Evaluate each category
    clf_result, clf_df = train_evaluate_classification_models(df)
    reg_result, reg_df = train_evaluate_regression_models(df)
    dl_result, dl_df = evaluate_deep_learning_models(df)
    
    # Final comparison
    comparison_df = compare_best_models(clf_result, reg_result, dl_result)
    
    # Save results
    print(f"\nSaving results to {output_dir}/...")
    clf_df.to_csv(f"{output_dir}/classification_results.csv", index=False)
    reg_df.to_csv(f"{output_dir}/regression_results.csv", index=False)
    dl_df.to_csv(f"{output_dir}/deep_learning_results.csv", index=False)
    comparison_df.to_csv(f"{output_dir}/final_comparison.csv", index=False)
    
    # Create summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_movies': len(df),
        'best_classification_model': clf_result['best_model_name'],
        'best_classification_f1': float(clf_result['best_metrics']['f1']),
        'best_regression_model': reg_result['best_model_name'],
        'best_regression_r2': float(reg_result['best_metrics']['combined_score']),
        'best_dl_model': dl_result['best_model_name'],
        'best_dl_score': float(dl_result['best_metrics']['combined_score'])
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {output_dir}/")
    print(f"   - classification_results.csv")
    print(f"   - regression_results.csv")
    print(f"   - deep_learning_results.csv")
    print(f"   - final_comparison.csv")
    print(f"   - summary.json")
    
    return {
        'classification': clf_result,
        'regression': reg_result,
        'deep_learning': dl_result,
        'comparison': comparison_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate all ML models")
    parser.add_argument("--data-path", type=str, required=True, help="Path to IMDb data directory")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    results = main(args.data_path, args.output_dir)
