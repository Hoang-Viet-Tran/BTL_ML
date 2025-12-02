"""
Production Inference Module

Lightweight, optimized prediction module for production deployment.
Designed for: API serving, batch processing, microservices.

Features:
- Fast model loading with caching
- Batch prediction optimization
- Comprehensive error handling
- Production logging
- Memory efficient

Usage:
    # Single prediction
    from predict import ModelPredictor
    
    predictor = ModelPredictor('models/best_model.pkl')
    result = predictor.predict_single(features)
    
    # Batch prediction
    results = predictor.predict_batch(features_list)
    
    # As module functions
    from predict import predict_single, predict_batch
    
    result = predict_single('models/best_model.pkl', features)
"""

# ------------------------------------------------------------------
# -----------------------------LIBRARIES----------------------------
# ------------------------------------------------------------------
import os
import logging
import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# -------------------------MODEL PREDICTOR--------------------------
# ------------------------------------------------------------------

class ModelPredictor:
    """
    Production-ready model predictor with optimization and caching.
    
    Features:
    - Model caching (load once, reuse)
    - Batch optimization
    - Input validation
    - Error handling
    - Performance logging
    
    Example:
        predictor = ModelPredictor('models/best_model.pkl')
        
        # Single prediction
        result = predictor.predict_single({
            'director_mean_rating': 8.5,
            'cast_mean_rating': 7.8,
            ...
        })
        
        # Batch prediction
        results = predictor.predict_batch(features_list)
    """
    
    _model_cache: Dict[str, Any] = {}  # Class-level cache for models
    
    def __init__(self, model_path: str, enable_cache: bool = True):
        """
        Initialize predictor with model.
        
        Args:
            model_path: Path to trained model (.pkl file)
            enable_cache: Whether to cache loaded models
        """
        self.model_path = model_path
        self.enable_cache = enable_cache
        self.model = self._load_model()
        
        # Detect model type and capabilities
        self.has_proba = hasattr(self.model, 'predict_proba')
        self.is_pipeline = hasattr(self.model, 'named_steps')
        
        logger.info(f"Predictor initialized with model: {model_path}")
        logger.info(f"  - Probability support: {self.has_proba}")
        logger.info(f"  - Pipeline model: {self.is_pipeline}")
    
    def _load_model(self) -> Any:
        """
        Load model from disk with optional caching.
        
        Returns:
            Loaded model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Check cache first
        if self.enable_cache and self.model_path in self._model_cache:
            logger.info(f"Loading model from cache: {self.model_path}")
            return self._model_cache[self.model_path]
        
        # Load from disk
        logger.info(f"Loading model from disk: {self.model_path}")
        try:
            model = joblib.load(self.model_path)
            
            # Cache if enabled
            if self.enable_cache:
                self._model_cache[self.model_path] = model
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _validate_features(self, features: Dict[str, Any]) -> None:
        """
        Validate input features.
        
        Args:
            features: Feature dictionary
            
        Raises:
            ValueError: If features are invalid
        """
        if not features:
            raise ValueError("Features dictionary is empty")
        
        # Check for NaN/None values
        for key, value in features.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                logger.warning(f"Feature '{key}' has NaN/None value, will be filled with 0")
    
    def _prepare_input(self, features: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """
        Prepare input features as DataFrame.
        
        Args:
            features: Single dict or list of dicts
            
        Returns:
            DataFrame ready for prediction
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        else:
            raise TypeError(f"Features must be dict or list of dicts, got {type(features)}")
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    
    def predict_single(
        self,
        features: Dict[str, Any],
        return_proba: bool = True
    ) -> Dict[str, Any]:
        """
        Predict for a single instance.
        
        Args:
            features: Dictionary of features
            return_proba: Whether to return probabilities (if available)
            
        Returns:
            Dictionary with prediction results:
            {
                'prediction': int/float,
                'probability': float (if classification),
                'confidence': float (if classification)
            }
        """
        try:
            # Validate
            self._validate_features(features)
            
            # Prepare input
            X = self._prepare_input(features)
            
            # Predict
            prediction = self.model.predict(X)[0]
            
            result = {
                'prediction': float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction
            }
            
            # Add probability if available
            if self.has_proba and return_proba:
                proba = self.model.predict_proba(X)[0]
                result['probability'] = float(proba[1])  # Probability of positive class
                result['confidence'] = float(max(proba))  # Max probability
                result['probabilities'] = {
                    'class_0': float(proba[0]),
                    'class_1': float(proba[1])
                }
            
            logger.debug(f"Single prediction: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        return_proba: bool = True,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict for multiple instances (optimized batch processing).
        
        Args:
            features_list: List of feature dictionaries
            return_proba: Whether to return probabilities
            batch_size: Process in batches of this size (None = all at once)
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if not features_list:
                return []
            
            logger.info(f"Batch prediction for {len(features_list)} instances")
            
            # Process all at once if no batch_size specified
            if batch_size is None:
                return self._predict_batch_internal(features_list, return_proba)
            
            # Process in batches
            results = []
            for i in range(0, len(features_list), batch_size):
                batch = features_list[i:i + batch_size]
                batch_results = self._predict_batch_internal(batch, return_proba)
                results.extend(batch_results)
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(features_list)-1)//batch_size + 1}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _predict_batch_internal(
        self,
        features_list: List[Dict],
        return_proba: bool
    ) -> List[Dict]:
        """Internal batch prediction without batching."""
        # Prepare input
        X = self._prepare_input(features_list)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': float(pred) if isinstance(pred, (np.integer, np.floating)) else pred
            }
            
            # Add probabilities if available
            if self.has_proba and return_proba:
                proba = self.model.predict_proba(X.iloc[[i]])[0]
                result['probability'] = float(proba[1])
                result['confidence'] = float(max(proba))
                result['probabilities'] = {
                    'class_0': float(proba[0]),
                    'class_1': float(proba[1])
                }
            
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        info = {
            'model_path': self.model_path,
            'model_type': type(self.model).__name__,
            'has_probability': self.has_proba,
            'is_pipeline': self.is_pipeline,
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024)
        }
        
        # Get feature info if possible
        if hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
        
        if self.is_pipeline and hasattr(self.model.named_steps.get('clf'), 'feature_importances_'):
            info['has_feature_importance'] = True
        
        return info


# ------------------------------------------------------------------
# ------------------------CONVENIENCE FUNCTIONS---------------------
# ------------------------------------------------------------------

def predict_single(
    model_path: str,
    features: Dict[str, Any],
    return_proba: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for one-off single prediction.
    
    Args:
        model_path: Path to model file
        features: Feature dictionary
        return_proba: Return probabilities if available
        
    Returns:
        Prediction result dictionary
        
    Example:
        result = predict_single(
            'models/best_model.pkl',
            {
                'director_mean_rating': 8.5,
                'cast_mean_rating': 7.8,
                'writer_mean_rating': 7.5,
                'runtimeMinutes': 120,
                'startYear': 2023,
                'genre_Action': 1,
                'genre_Drama': 0,
                ...
            }
        )
        print(result)
        # {'prediction': 0, 'probability': 0.15, 'confidence': 0.85}
    """
    predictor = ModelPredictor(model_path)
    return predictor.predict_single(features, return_proba)


def predict_batch(
    model_path: str,
    features_list: List[Dict[str, Any]],
    return_proba: bool = True,
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for one-off batch prediction.
    
    Args:
        model_path: Path to model file
        features_list: List of feature dictionaries
        return_proba: Return probabilities if available
        batch_size: Process in batches of this size
        
    Returns:
        List of prediction result dictionaries
        
    Example:
        results = predict_batch(
            'models/best_model.pkl',
            [
                {'director_mean_rating': 8.5, ...},
                {'director_mean_rating': 7.2, ...},
                {'director_mean_rating': 9.0, ...}
            ]
        )
        for r in results:
            print(r)
    """
    predictor = ModelPredictor(model_path)
    return predictor.predict_batch(features_list, return_proba, batch_size)


def predict_from_csv(
    model_path: str,
    input_csv: str,
    output_csv: Optional[str] = None,
    return_proba: bool = True
) -> pd.DataFrame:
    """
    Predict from CSV file and optionally save results.
    
    Args:
        model_path: Path to model file
        input_csv: Input CSV file with features
        output_csv: Output CSV file (optional)
        return_proba: Include probabilities
        
    Returns:
        DataFrame with predictions
        
    Example:
        df_results = predict_from_csv(
            'models/best_model.pkl',
            'input_movies.csv',
            'predictions.csv'
        )
    """
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Convert to list of dicts
    features_list = df.to_dict('records')
    
    # Predict
    predictor = ModelPredictor(model_path)
    results = predictor.predict_batch(features_list, return_proba)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Combine with original data
    df_combined = pd.concat([df, df_results], axis=1)
    
    # Save if output path provided
    if output_csv:
        df_combined.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to {output_csv}")
    
    return df_combined


# ------------------------------------------------------------------
# ----------------------------CLI INTERFACE-------------------------
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Inference Module")
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, help='Output CSV file (optional)')
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    parser.add_argument('--no-proba', action='store_true', help='Disable probability output')
    
    args = parser.parse_args()
    
    # Run prediction
    df_results = predict_from_csv(
        model_path=args.model,
        input_csv=args.input,
        output_csv=args.output,
        return_proba=not args.no_proba
    )
    
    print(f"\n✅ Processed {len(df_results)} predictions")
    print(f"\nPreview:")
    print(df_results.head())
