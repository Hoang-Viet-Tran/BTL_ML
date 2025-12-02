"""
Main Script for Movie Success Prediction

This script provides a unified interface to:
1. Evaluate all models and select the best one
2. Train the best model
3. Make predictions using the best model
4. Serve predictions via command-line interface

Usage:
    # Run full evaluation and train best model
    python main.py --mode evaluate --data-path ../data/

    # Train specific model
    python main.py --mode train --model-type classification --data-path ../data/

    # Make predictions
    python main.py --mode predict --model-path models/best_model.pkl --input-file test_movies.csv

    # Interactive prediction
    python main.py --mode interactive --model-path models/best_model.pkl
"""

# ------------------------------------------------------------------
# -----------------------------LIBRARIES----------------------------
# ------------------------------------------------------------------
import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from preprocess import preprocess_pipeline
from feature import encode_genres, get_feature_target
from evaluate_models import (
    train_evaluate_classification_models,
    train_evaluate_regression_models,
    evaluate_deep_learning_models,
    compare_best_models
)


# ------------------------------------------------------------------
# --------------------------CONFIGURATION---------------------------
# ------------------------------------------------------------------

DEFAULT_DATA_PATH = '../data/'
DEFAULT_MODEL_DIR = 'models/'
DEFAULT_OUTPUT_DIR = 'evaluation_results/'

# Model selection configuration
MODEL_SELECTION_CRITERIA = {
    'classification': 'f1',  # Select based on F1-score
    'regression': 'combined_score',  # Select based on combined R²
    'deep_learning': 'combined_score'  # Select based on combined score
}


# ------------------------------------------------------------------
# -------------------------EVALUATION MODE--------------------------
# ------------------------------------------------------------------

def run_full_evaluation(data_path: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    """
    Run full model evaluation pipeline and return best models from each category.
    
    Args:
        data_path: Path to data directory
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing best models and their metrics
    """
    print("\n" + "="*70)
    print("RUNNING FULL MODEL EVALUATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("\n[1/4 - 25%] Loading and preprocessing data...")
    df = preprocess_pipeline(data_path)
    print(f"   ✓ Loaded {len(df)} movies with {df.shape[1]} features")
    
    # Evaluate each category
    print("\n[2/4 - 50%] Evaluating classification models...")
    clf_result, clf_df = train_evaluate_classification_models(df)
    
    print("\n[3/4 - 75%] Evaluating regression models...")
    reg_result, reg_df = train_evaluate_regression_models(df)
    
    print("\n[4/4 - 100%] Evaluating deep learning models...")
    dl_result, dl_df = evaluate_deep_learning_models(df)
    
    # Final comparison
    comparison_df = compare_best_models(clf_result, reg_result, dl_result)
    
    # Save results
    print(f"\n💾 Saving results to {output_dir}...")
    clf_df.to_csv(f"{output_dir}/classification_results.csv", index=False)
    reg_df.to_csv(f"{output_dir}/regression_results.csv", index=False)
    dl_df.to_csv(f"{output_dir}/deep_learning_results.csv", index=False)
    comparison_df.to_csv(f"{output_dir}/final_comparison.csv", index=False)
    
    # Save best models
    model_save_dir = os.path.join(output_dir, 'best_models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    if clf_result['best_model'] is not None:
        joblib.dump(clf_result['best_model'], f"{model_save_dir}/best_classification.pkl")
        print(f"   ✓ Saved {clf_result['best_model_name']} to best_classification.pkl")
    
    if reg_result['best_model'] is not None:
        joblib.dump(reg_result['best_model'], f"{model_save_dir}/best_regression.pkl")
        print(f"   ✓ Saved {reg_result['best_model_name']} to best_regression.pkl")
    
    # Save metadata
    metadata = {
        'classification': {
            'model_name': clf_result['best_model_name'],
            'metrics': clf_result['best_metrics']
        },
        'regression': {
            'model_name': reg_result['best_model_name'],
            'metrics': reg_result['best_metrics']
        },
        'deep_learning': {
            'model_name': dl_result['best_model_name'],
            'metrics': dl_result['best_metrics']
        }
    }
    
    with open(f"{output_dir}/best_models_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    
    return {
        'classification': clf_result,
        'regression': reg_result,
        'deep_learning': dl_result,
        'comparison': comparison_df
    }


# ------------------------------------------------------------------
# ---------------------------TRAINING MODE--------------------------
# ------------------------------------------------------------------

def train_best_model(
    data_path: str,
    model_type: str,
    output_dir: str = DEFAULT_MODEL_DIR
) -> Dict[str, Any]:
    """
    Train the best model for a specific task type.
    
    Args:
        data_path: Path to data directory
        model_type: 'classification', 'regression', or 'deep_learning'
        output_dir: Directory to save trained model
        
    Returns:
        Dictionary with model and metrics
    """
    print(f"\n{'='*70}")
    print(f"TRAINING BEST {model_type.upper()} MODEL")
    print(f"{'='*70}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = preprocess_pipeline(data_path)
    
    # Train based on model type
    if model_type == 'classification':
        result, metrics_df = train_evaluate_classification_models(df)
        save_path = os.path.join(output_dir, 'best_classification.pkl')
    elif model_type == 'regression':
        result, metrics_df = train_evaluate_regression_models(df)
        save_path = os.path.join(output_dir, 'best_regression.pkl')
    elif model_type == 'deep_learning':
        result, metrics_df = evaluate_deep_learning_models(df)
        save_path = os.path.join(output_dir, 'best_deep_learning.pkl')
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    if result['best_model'] is not None:
        joblib.dump(result['best_model'], save_path)
        print(f"\n💾 Saved {result['best_model_name']} to {save_path}")
    else:
        print(f"\n⚠️  {model_type} model not available for saving (DL models require separate training)")
    
    # Save metrics
    metrics_df.to_csv(os.path.join(output_dir, f'{model_type}_metrics.csv'), index=False)
    
    return result


# ------------------------------------------------------------------
# -------------------------PREDICTION MODE--------------------------
# ------------------------------------------------------------------

def load_model(model_path: str) -> Any:
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"📂 Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("   ✓ Model loaded successfully")
    return model


def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for prediction (same as training preprocessing)."""
    # If already preprocessed, just encode genres
    if 'is_risky' not in df.columns:
        # Need to run full preprocessing (for raw data)
        # Note: This assumes df is raw IMDB data
        print("   ⚠️  Input appears to be raw data - running full preprocessing")
        df = preprocess_pipeline(df)
    
    # Encode genres
    df = encode_genres(df)
    return df


def predict_batch(
    model: Any,
    input_data: pd.DataFrame,
    target_type: str = 'is_risky'
) -> pd.DataFrame:
    """
    Make batch predictions on input data.
    
    Args:
        model: Trained model
        input_data: DataFrame with movie features
        target_type: Type of prediction ('is_risky', 'rating', 'rating_votes')
        
    Returns:
        DataFrame with predictions
    """
    print(f"\n🔮 Making predictions on {len(input_data)} movies...")
    
    # Preprocess
    df_processed = preprocess_for_prediction(input_data.copy())
    
    # Extract features
    X, _ = get_feature_target(df_processed, target_type=target_type)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Format results
    if target_type == 'is_risky':
        # Classification
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[:, 1]
            results = pd.DataFrame({
                'prediction': predictions,
                'is_risky': predictions,
                'risk_probability': proba,
                'recommendation': ['High Risk' if p > 0.5 else 'Low Risk' for p in proba]
            })
        else:
            results = pd.DataFrame({
                'prediction': predictions,
                'is_risky': predictions,
                'recommendation': ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
            })
    elif target_type == 'rating_votes':
        # Multi-output regression
        results = pd.DataFrame({
            'predicted_rating': predictions[:, 0],
            'predicted_votes': predictions[:, 1]
        })
    else:
        # Single regression
        results = pd.DataFrame({
            'prediction': predictions
        })
    
    # Add original titles if available
    if 'primaryTitle' in input_data.columns:
        results.insert(0, 'title', input_data['primaryTitle'].values)
    
    print(f"   ✓ Predictions complete")
    return results


def predict_single_interactive(model: Any, target_type: str = 'is_risky'):
    """
    Interactive prediction mode for single movies.
    """
    print("\n" + "="*70)
    print("INTERACTIVE PREDICTION MODE")
    print("="*70)
    print("\nEnter movie features (or 'quit' to exit):\n")
    
    while True:
        try:
            # Get input
            print("-" * 50)
            title = input("Movie Title: ").strip()
            if title.lower() == 'quit':
                break
            
            director_rating = float(input("Director Mean Rating (0-10): "))
            director_films = int(input("Director Total Films: "))
            cast_rating = float(input("Cast Mean Rating (0-10): "))
            cast_films = int(input("Cast Total Films: "))
            writer_rating = float(input("Writer Mean Rating (0-10): "))
            writer_films = int(input("Writer Total Films: "))
            runtime = int(input("Runtime (minutes): "))
            start_year = int(input("Release Year: "))
            
            # Genres
            print("\nGenres (comma-separated, e.g., Action,Drama,Thriller):")
            genres_input = input("Genres: ").strip()
            genres = [g.strip() for g in genres_input.split(',')]
            
            # Create feature dictionary
            features = {
                'director_mean_rating': director_rating,
                'director_total_films': director_films,
                'cast_mean_rating': cast_rating,
                'cast_total_films': cast_films,
                'writer_mean_rating': writer_rating,
                'writer_total_films': writer_films,
                'runtimeMinutes': runtime,
                'startYear': start_year
            }
            
            # Add genre features (assumes all genres are 0 by default)
            # In real scenario, you'd need the full list of possible genres
            for genre in genres:
                features[f'genre_{genre}'] = 1
            
            # Create DataFrame
            df_input = pd.DataFrame([features])
            
            # Make prediction
            prediction = model.predict(df_input)[0]
            
            # Display result
            print("\n" + "="*50)
            print(f"🎬 Movie: {title}")
            if target_type == 'is_risky':
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(df_input)[0][1]
                    print(f"📊 Risk Probability: {proba:.2%}")
                    print(f"🎯 Prediction: {'HIGH RISK ⚠️' if prediction == 1 else 'LOW RISK ✅'}")
                else:
                    print(f"🎯 Prediction: {'HIGH RISK ⚠️' if prediction == 1 else 'LOW RISK ✅'}")
            else:
                print(f"🎯 Prediction: {prediction:.2f}")
            print("="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")


# ------------------------------------------------------------------
# ------------------------------MAIN--------------------------------
# ------------------------------------------------------------------

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Movie Success Prediction - Best Model Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation
  python main.py --mode evaluate --data-path ../data/

  # Train best classification model
  python main.py --mode train --model-type classification --data-path ../data/

  # Batch predictions
  python main.py --mode predict --model-path models/best_classification.pkl --input-file test.csv --output-file predictions.csv

  # Interactive predictions
  python main.py --mode interactive --model-path models/best_classification.pkl
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['evaluate', 'train', 'predict', 'interactive'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='Path to data directory (for evaluate/train modes)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['classification', 'regression', 'deep_learning'],
        help='Type of model to train (for train mode)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model file (for predict/interactive modes)'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        help='Input CSV file for batch predictions (for predict mode)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output CSV file for predictions (for predict mode)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--target-type',
        type=str,
        default='is_risky',
        choices=['is_risky', 'rating', 'votes', 'rating_votes'],
        help='Target type for predictions'
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'evaluate':
        print("\n🚀 Starting full evaluation pipeline...")
        results = run_full_evaluation(args.data_path, args.output_dir)
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\n📊 Best Models:")
        print(f"  • Classification: {results['classification']['best_model_name']}")
        print(f"  • Regression: {results['regression']['best_model_name']}")
        print(f"  • Deep Learning: {results['deep_learning']['best_model_name']}")
        print(f"\n📁 Results saved to: {args.output_dir}")
        
    elif args.mode == 'train':
        if not args.model_type:
            parser.error("--model-type is required for train mode")
        print(f"\n🚀 Training best {args.model_type} model...")
        result = train_best_model(args.data_path, args.model_type, args.output_dir)
        print(f"\n✅ Training complete!")
        print(f"   Best Model: {result['best_model_name']}")
        
    elif args.mode == 'predict':
        if not args.model_path:
            parser.error("--model-path is required for predict mode")
        if not args.input_file:
            parser.error("--input-file is required for predict mode")
        
        print(f"\n🚀 Running batch predictions...")
        model = load_model(args.model_path)
        input_data = pd.read_csv(args.input_file)
        
        predictions = predict_batch(model, input_data, args.target_type)
        
        # Save predictions
        output_file = args.output_file or 'predictions.csv'
        predictions.to_csv(output_file, index=False)
        print(f"\n✅ Predictions saved to: {output_file}")
        print(f"\nPreview:")
        print(predictions.head())
        
    elif args.mode == 'interactive':
        if not args.model_path:
            parser.error("--model-path is required for interactive mode")
        
        model = load_model(args.model_path)
        predict_single_interactive(model, args.target_type)


if __name__ == "__main__":
    main()