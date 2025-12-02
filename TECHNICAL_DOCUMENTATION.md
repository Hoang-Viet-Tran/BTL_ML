# TECHNICAL DOCUMENTATION
# Movie Success Prediction System

Version: 1.0.0  
Last Updated: 2025-11-20  
Authors: Data Science Team

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Module Reference](#module-reference)
   - 3.1 [preprocess.py](#31-preprocesspy)
   - 3.2 [feature.py](#32-featurepy)
   - 3.3 [train.py](#33-trainpy)
   - 3.4 [evaluate_models.py](#34-evaluate_modelspy)
   - 3.5 [main.py](#35-mainpy)
   - 3.6 [predict.py](#36-predictpy)
   - 3.7 [api_server.py](#37-api_serverpy)
4. [Data Structures](#data-structures)
5. [Local Deployment Guide](#local-deployment-guide)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)

---

## 1. SYSTEM OVERVIEW

### 1.1 Purpose

The Movie Success Prediction System is a machine learning platform designed to assess investment risk for movie production. The system evaluates multiple factors including director track record, cast reputation, writer history, and genre combinations to predict whether a movie is likely to be successful or risky.

### 1.2 Key Features

- Comprehensive data preprocessing pipeline
- Feature engineering with writer, director, and cast statistics
- 6 machine learning models across 3 categories
- Production-ready inference module
- REST API server with auto-generated documentation
- Docker containerization support

### 1.3 Technology Stack

- Python 3.10+
- scikit-learn 1.3.0
- XGBoost 1.7.6
- PyTorch 2.0.1 (optional for deep learning)
- PyTorch Lightning 2.0.6 (optional for deep learning)
- FastAPI 0.101.0
- Pandas 2.0.3
- NumPy 1.24.3

---

## 2. ARCHITECTURE

### 2.1 System Architecture

```
Data Pipeline:
    Raw IMDb Data
        |
        v
    preprocess.py (Data Cleaning & Person Stats)
        |
        v
    feature.py (Feature Engineering & Encoding)
        |
        v
    train.py (Model Training)
        |
        v
    evaluate_models.py (Model Selection)
        |
        v
    Best Model
        |
        +-- Development --> main.py
        |
        +-- Production --> predict.py --> api_server.py
```

### 2.2 Module Dependencies

```
preprocess.py
    |
    +-- utilities/helper.py
    |
feature.py (uses output from preprocess.py)
    |
train.py (uses feature.py + preprocess.py)
    |
evaluate_models.py (uses train, feature, preprocess)
    |
main.py (uses evaluate_models, feature, preprocess)
    |
predict.py (standalone, uses saved models)
    |
api_server.py (uses predict.py)
```

---

## 3. MODULE REFERENCE

### 3.1 preprocess.py

**Purpose**: Data preprocessing pipeline for IMDb data. Cleans raw data, computes person statistics, and creates target labels.

**Location**: `src/preprocess.py`

**Dependencies**: `pandas`, `utilities/helper.py`

#### 3.1.1 Functions

##### preprocess_pipeline

```python
def preprocess_pipeline(data_path: str) -> pd.DataFrame
```

Main preprocessing function that orchestrates the entire data cleaning pipeline.

**Parameters**:
- `data_path` (str): Path to directory containing IMDb TSV files. Must contain:
  - title.basics.tsv
  - title.ratings.tsv
  - title.crew.tsv
  - title.principals.tsv
  - name.basics.tsv
  - title.akas.tsv

**Returns**:
- `pd.DataFrame`: Preprocessed DataFrame with columns:
  - Movie metadata: tconst, primaryTitle, startYear, runtimeMinutes, genres
  - Person identifiers: directors_nconst, writers_nconst, cast_nconst
  - Person statistics: director_mean_rating, director_total_films, writer_mean_rating, writer_total_films, cast_mean_rating, cast_total_films
  - Ratings: averageRating, numVotes
  - Target labels: is_success, is_risky

**Processing Steps**:
1. Load all IMDb TSV files
2. Filter movies only, remove adult content
3. Merge rating information
4. Process crew (directors, writers) and cast
5. Compute person-based statistics
6. Fill missing values
7. Create binary labels (is_success, is_risky)

**Example**:
```python
from preprocess import preprocess_pipeline

df = preprocess_pipeline('../data/')
print(df.shape)  # (45000, 28)
print(df.columns)
```

**Raises**:
- `FileNotFoundError`: If data files are missing
- `ValueError`: If data format is invalid

---

### 3.2 feature.py

**Purpose**: Feature engineering module. Handles genre encoding and feature extraction for machine learning models.

**Location**: `src/feature.py`

**Dependencies**: `pandas`, `numpy`, `typing`

#### 3.2.1 Functions

##### encode_genres

```python
def encode_genres(df: pd.DataFrame, genre_col: str = 'genre_list') -> pd.DataFrame
```

Converts genre lists into multi-hot encoded binary columns.

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame with genre information
- `genre_col` (str, optional): Column name containing genre lists. Default: 'genre_list'

**Returns**:
- `pd.DataFrame`: DataFrame with additional genre columns (genre_Action, genre_Drama, etc.)

**Behavior**:
- Creates one binary column per unique genre in dataset
- Each column has value 1 if movie belongs to that genre, 0 otherwise
- Handles missing/invalid genre data gracefully

**Example**:
```python
from feature import encode_genres

df['genre_list'] = [['Action', 'Drama'], ['Comedy'], ['Action', 'Thriller']]
df_encoded = encode_genres(df)
# New columns: genre_Action, genre_Drama, genre_Comedy, genre_Thriller
```

##### get_feature_target

```python
def get_feature_target(
    df: pd.DataFrame, 
    target_type: str = 'rating'
) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]
```

Splits dataset into feature matrix X and target vector(s) y.

**Parameters**:
- `df` (pd.DataFrame): Preprocessed DataFrame with features and targets
- `target_type` (str): Type of target variable. Options:
  - `'rating'`: Single-output regression (averageRating)
  - `'votes'`: Single-output regression (numVotes)
  - `'rating_votes'`: Multi-output regression (both)
  - `'is_risky'`: Binary classification
  - Any column name (backwards compatibility)

**Returns**:
- `tuple`: (X, y) where:
  - `X` (pd.DataFrame): Feature matrix with columns:
    - director_mean_rating, director_total_films
    - writer_mean_rating, writer_total_films
    - cast_mean_rating, cast_total_films
    - genre_* (all encoded genres)
    - runtimeMinutes, startYear, director_year_gap (if available)
  - `y` (pd.Series or pd.DataFrame): Target variable(s)

**Example**:
```python
from feature import get_feature_target

# Classification
X, y = get_feature_target(df, target_type='is_risky')
print(X.shape)  # (45000, 25)
print(y.shape)  # (45000,)

# Multi-output regression
X, y = get_feature_target(df, target_type='rating_votes')
print(y.shape)  # (45000, 2)
```

---

### 3.3 train.py

**Purpose**: Model training module. Contains 6 machine learning models and training pipelines.

**Location**: `src/train.py`

**Dependencies**: `sklearn`, `xgboost`, `pytorch`, `pytorch_lightning`

#### 3.3.1 Functions

##### train_is_risky_models

```python
def train_is_risky_models(
    data_path: str,
    save_dir: str = "models/is_risky"
) -> Dict[str, Any]
```

Trains classification models for predicting movie risk (is_risky target).

**Parameters**:
- `data_path` (str): Path to raw data directory or preprocessed CSV
- `save_dir` (str, optional): Directory to save trained models. Default: "models/is_risky"

**Returns**:
- `dict`: Dictionary with keys:
  - `'logreg'`: Trained Logistic Regression model
  - `'rf'`: Trained Random Forest Classifier

**Models Trained**:
1. Logistic Regression with GridSearchCV
   - Hyperparameters: C=[0.1, 1.0, 5.0]
   - Scoring: F1-score
   - Cross-validation: 3-fold

2. Random Forest Classifier with GridSearchCV
   - Hyperparameters: n_estimators=[100, 200], max_depth=[None, 10, 20]
   - Scoring: F1-score
   - Cross-validation: 3-fold

**Side Effects**:
- Saves models to `{save_dir}/logreg_is_risky.pkl` and `{save_dir}/rf_is_risky.pkl`
- Prints training progress to console

**Example**:
```python
from train import train_is_risky_models

models = train_is_risky_models('../data/')
best_logreg = models['logreg']
best_rf = models['rf']
```

##### train_rating_votes_models

```python
def train_rating_votes_models(
    data_path: str,
    save_dir: str = "models/rating_votes"
) -> Dict[str, Any]
```

Trains regression models for predicting movie ratings and vote counts.

**Parameters**:
- `data_path` (str): Path to data directory or CSV
- `save_dir` (str, optional): Directory to save models. Default: "models/rating_votes"

**Returns**:
- `dict`: Dictionary with keys:
  - `'rf_reg'`: Trained Random Forest Regressor (MultiOutput)
  - `'xgb_reg'`: Trained XGBoost Regressor (MultiOutput)

**Models Trained**:
1. Random Forest Regressor
   - Multi-output: Predicts both averageRating and numVotes
   - Hyperparameters: n_estimators=[200, 400], max_depth=[None, 10, 20]
   - Scoring: Negative MAE

2. XGBoost Regressor
   - Multi-output wrapped
   - Fixed hyperparameters: n_estimators=300, learning_rate=0.05, max_depth=8
   - Objective: reg:squarederror

**Example**:
```python
from train import train_rating_votes_models

models = train_rating_votes_models('../data/')
rf_model = models['rf_reg']
xgb_model = models['xgb_reg']
```

##### train_hybrid_dl_models

```python
def train_hybrid_dl_models(
    data_path: str,
    save_dir: str = "models/hybrid_dl",
    topk_dir: int = 5000,
    topk_cast: int = 20000,
    topk_writer: int = 5000,
    batch_size: int = 256,
    max_epochs: int = 20,
    lr: float = 1e-3,
    gpu: Optional[int] = 0
) -> Dict[str, Any]
```

Trains deep learning hybrid model with embeddings for multi-task learning.

**Parameters**:
- `data_path` (str): Path to data
- `save_dir` (str): Save directory for checkpoints and vocabs
- `topk_dir` (int): Number of top directors to include in vocabulary
- `topk_cast` (int): Number of top cast members to include
- `topk_writer` (int): Number of top writers to include
- `batch_size` (int): Training batch size
- `max_epochs` (int): Maximum training epochs
- `lr` (float): Learning rate
- `gpu` (Optional[int]): GPU device ID, None for CPU

**Returns**:
- `dict`: Dictionary with keys:
  - `'best_ckpt'`: Path to best checkpoint
  - `'metrics'`: Test set metrics
  - `'vocab_paths'`: Paths to vocabulary files

**Model Architecture**:
- Embeddings for directors, cast, writers
- Batch normalization on numeric/genre features
- Shared MLP backbone
- Three task heads: classification + 2 regression

**Example**:
```python
from train import train_hybrid_dl_models

result = train_hybrid_dl_models(
    data_path='../data/',
    max_epochs=20,
    gpu=0
)
print(result['metrics'])
```

#### 3.3.2 Classes

##### SimplerDNN

```python
class SimplerDNN(pl.LightningModule)
```

Simpler deep neural network for multi-task learning without person embeddings.

**Constructor Parameters**:
- `numeric_dim` (int): Number of numeric features
- `genre_dim` (int): Number of genre features
- `hidden_dims` (List[int]): Hidden layer dimensions. Default: [256, 128, 64]
- `dropout` (float): Dropout rate. Default: 0.3
- `lr` (float): Learning rate. Default: 1e-3
- `weight_decay` (float): L2 regularization. Default: 1e-4
- `cls_weight` (float): Classification loss weight. Default: 1.0
- `rating_weight` (float): Rating loss weight. Default: 1.0
- `votes_weight` (float): Votes loss weight. Default: 1.0

**Methods**:

###### forward

```python
def forward(self, numeric: torch.Tensor, genre: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Forward pass through network.

**Parameters**:
- `numeric` (torch.Tensor): Numeric features, shape (batch_size, numeric_dim)
- `genre` (torch.Tensor): Genre features, shape (batch_size, genre_dim)

**Returns**:
- `tuple`: (cls_logits, rating_out, votes_out)
  - cls_logits (torch.Tensor): Classification logits, shape (batch_size,)
  - rating_out (torch.Tensor): Rating predictions, shape (batch_size,)
  - votes_out (torch.Tensor): Votes predictions (log-scaled), shape (batch_size,)

##### HybridMovieModel

```python
class HybridMovieModel(pl.LightningModule)
```

Advanced hybrid model with person embeddings for multi-task learning.

**Constructor Parameters**:
- `numeric_dim` (int): Number of numeric features
- `genre_dim` (int): Number of genre features
- `dir_vocab_size` (int): Director vocabulary size
- `cast_vocab_size` (int): Cast vocabulary size
- `writer_vocab_size` (int): Writer vocabulary size
- `dir_emb_dim` (int): Director embedding dimension. Default: 32
- `cast_emb_dim` (int): Cast embedding dimension. Default: 64
- `writer_emb_dim` (int): Writer embedding dimension. Default: 32
- `shared_hidden_dims` (List[int]): Shared MLP dimensions. Default: [256, 128]
- `dropout` (float): Dropout rate. Default: 0.2
- `lr` (float): Learning rate. Default: 1e-3
- `weight_decay` (float): Weight decay. Default: 1e-5
- `cls_weight` (float): Classification loss weight. Default: 1.0
- `rating_weight` (float): Rating loss weight. Default: 1.0
- `votes_weight` (float): Votes loss weight. Default: 1.0

**Methods**:

###### forward

```python
def forward(
    self, 
    numeric: torch.Tensor, 
    genre: torch.Tensor,
    dirs: torch.Tensor,
    cast: torch.Tensor,
    writers: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Forward pass with person embeddings.

**Parameters**:
- `numeric` (torch.Tensor): Numeric features
- `genre` (torch.Tensor): Genre features
- `dirs` (torch.Tensor): Director IDs, shape (batch_size, max_directors)
- `cast` (torch.Tensor): Cast IDs, shape (batch_size, max_cast)
- `writers` (torch.Tensor): Writer IDs, shape (batch_size, max_writers)

**Returns**:
- `tuple`: (cls_logits, rating_out, votes_out)

---

### 3.4 evaluate_models.py

**Purpose**: Model evaluation and comparison framework.

**Location**: `src/evaluate_models.py`

#### 3.4.1 Functions

##### evaluate_classification_model

```python
def evaluate_classification_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str
) -> Dict[str, Any]
```

Evaluates a classification model on test data.

**Parameters**:
- `model`: Trained classification model (must have predict and optionally predict_proba methods)
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test labels
- `model_name` (str): Name for logging

**Returns**:
- `dict`: Metrics dictionary with keys:
  - `'model_name'` (str): Model name
  - `'accuracy'` (float): Accuracy score (0-1)
  - `'precision'` (float): Precision score (0-1)
  - `'recall'` (float): Recall score (0-1)
  - `'f1'` (float): F1-score (0-1)
  - `'auc_roc'` (float): AUC-ROC score (0-1)

##### evaluate_regression_model

```python
def evaluate_regression_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, pd.DataFrame],
    model_name: str,
    is_multioutput: bool = False
) -> Dict[str, Any]
```

Evaluates a regression model on test data.

**Parameters**:
- `model`: Trained regression model
- `X_test` (pd.DataFrame): Test features
- `y_test` (Union[pd.Series, pd.DataFrame]): Test targets
- `model_name` (str): Model name
- `is_multioutput` (bool): Whether model predicts multiple outputs

**Returns**:
- `dict`: Metrics dictionary. For single-output:
  - `'mae'` (float): Mean Absolute Error
  - `'rmse'` (float): Root Mean Squared Error
  - `'r2'` (float): R-squared score
  
  For multi-output:
  - `'mae_rating'`, `'rmse_rating'`, `'r2_rating'`: Rating metrics
  - `'mae_votes'`, `'rmse_votes'`, `'r2_votes'`: Votes metrics
  - `'combined_score'` (float): Average R-squared

##### train_evaluate_classification_models

```python
def train_evaluate_classification_models(
    df: pd.DataFrame
) -> Tuple[Dict, pd.DataFrame]
```

Trains and evaluates all classification models.

**Parameters**:
- `df` (pd.DataFrame): Preprocessed DataFrame

**Returns**:
- `tuple`: (result_dict, results_df)
  - result_dict (Dict): Best model info
  - results_df (pd.DataFrame): Metrics for all models

##### train_evaluate_regression_models

```python
def train_evaluate_regression_models(
    df: pd.DataFrame
) -> Tuple[Dict, pd.DataFrame]
```

Trains and evaluates all regression models.

**Parameters**:
- `df` (pd.DataFrame): Preprocessed DataFrame

**Returns**:
- `tuple`: (result_dict, results_df)

##### main

```python
def main(data_path: str, output_dir: str = "evaluation_results") -> Dict
```

Main evaluation pipeline orchestrator.

**Parameters**:
- `data_path` (str): Path to data
- `output_dir` (str): Output directory for results

**Returns**:
- `dict`: Complete evaluation results

**Side Effects**:
- Saves CSV files: classification_results.csv, regression_results.csv, etc.
- Saves JSON summary

---

### 3.5 main.py

**Purpose**: Unified CLI interface for all operations (evaluate, train, predict, interactive).

**Location**: `src/main.py`

#### 3.5.1 Functions

##### run_full_evaluation

```python
def run_full_evaluation(
    data_path: str, 
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]
```

Executes complete model evaluation pipeline.

**Parameters**:
- `data_path` (str): Path to data directory
- `output_dir` (str): Output directory. Default: "evaluation_results"

**Returns**:
- `dict`: Evaluation results with keys:
  - `'classification'`: Classification results
  - `'regression'`: Regression results
  - `'deep_learning'`: DL results
  - `'comparison'`: Final comparison DataFrame

**Side Effects**:
- Creates output_dir
- Saves best models to output_dir/best_models/
- Saves metrics CSVs
- Saves metadata JSON

##### train_best_model

```python
def train_best_model(
    data_path: str,
    model_type: str,
    output_dir: str = "models"
) -> Dict[str, Any]
```

Trains best model for specified type.

**Parameters**:
- `data_path` (str): Path to data
- `model_type` (str): One of 'classification', 'regression', 'deep_learning'
- `output_dir` (str): Save directory

**Returns**:
- `dict`: Training results

**Raises**:
- `ValueError`: If model_type is invalid

##### predict_batch

```python
def predict_batch(
    model: Any,
    input_data: pd.DataFrame,
    target_type: str = 'is_risky'
) -> pd.DataFrame
```

Makes batch predictions on input DataFrame.

**Parameters**:
- `model`: Trained model
- `input_data` (pd.DataFrame): Input features
- `target_type` (str): Type of prediction

**Returns**:
- `pd.DataFrame`: Predictions with original data

##### predict_single_interactive

```python
def predict_single_interactive(model: Any, target_type: str = 'is_risky') -> None
```

Interactive prediction mode (CLI).

**Parameters**:
- `model`: Trained model
- `target_type` (str): Prediction type

**Side Effects**:
- Prompts user for input
- Prints predictions
- Loops until user quits

##### main

```python
def main() -> None
```

CLI entry point with argument parsing.

**Command Line Arguments**:
- `--mode`: Operation mode (evaluate, train, predict, interactive)
- `--data-path`: Data directory path
- `--model-type`: Model type for training
- `--model-path`: Path to saved model
- `--input-file`: Input CSV for predictions
- `--output-file`: Output CSV for predictions
- `--output-dir`: Output directory
- `--target-type`: Target type for predictions

---

### 3.6 predict.py

**Purpose**: Production inference module with optimization and caching.

**Location**: `src/predict.py`

#### 3.6.1 Classes

##### ModelPredictor

```python
class ModelPredictor
```

Production-ready model predictor with caching and optimization.

**Class Attributes**:
- `_model_cache` (Dict[str, Any]): Class-level model cache

**Constructor**:

```python
def __init__(self, model_path: str, enable_cache: bool = True)
```

**Parameters**:
- `model_path` (str): Path to .pkl model file
- `enable_cache` (bool): Enable model caching. Default: True

**Raises**:
- `FileNotFoundError`: If model file doesn't exist

**Instance Attributes**:
- `model_path` (str): Path to model
- `enable_cache` (bool): Cache enabled flag
- `model` (Any): Loaded model
- `has_proba` (bool): Whether model supports predict_proba
- `is_pipeline` (bool): Whether model is sklearn Pipeline

**Methods**:

###### predict_single

```python
def predict_single(
    self,
    features: Dict[str, Any],
    return_proba: bool = True
) -> Dict[str, Any]
```

Predicts for single instance.

**Parameters**:
- `features` (Dict[str, Any]): Feature dictionary with keys matching model input
- `return_proba` (bool): Return probabilities (if available). Default: True

**Returns**:
- `dict`: Prediction result with keys:
  - `'prediction'` (float): Predicted value/class
  - `'probability'` (float, optional): Probability of positive class
  - `'confidence'` (float, optional): Confidence score
  - `'probabilities'` (dict, optional): Per-class probabilities

**Raises**:
- `ValueError`: If features are invalid
- `TypeError`: If features type is wrong

**Example**:
```python
predictor = ModelPredictor('models/best_model.pkl')
result = predictor.predict_single({
    'director_mean_rating': 8.5,
    'cast_mean_rating': 7.8,
    'writer_mean_rating': 7.5,
    'runtimeMinutes': 148,
    'startYear': 2023,
    'genre_Action': 1,
    'genre_Drama': 1
})
# Returns: {'prediction': 0, 'probability': 0.12, 'confidence': 0.88}
```

###### predict_batch

```python
def predict_batch(
    self,
    features_list: List[Dict[str, Any]],
    return_proba: bool = True,
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]
```

Predicts for multiple instances with batch optimization.

**Parameters**:
- `features_list` (List[Dict[str, Any]]): List of feature dictionaries
- `return_proba` (bool): Return probabilities. Default: True
- `batch_size` (Optional[int]): Process in batches of this size. None = all at once

**Returns**:
- `List[Dict[str, Any]]`: List of prediction dictionaries

**Example**:
```python
results = predictor.predict_batch([
    {'director_mean_rating': 8.5, ...},
    {'director_mean_rating': 7.2, ...}
], batch_size=100)
```

###### get_model_info

```python
def get_model_info(self) -> Dict[str, Any]
```

Returns model metadata.

**Returns**:
- `dict`: Model information with keys:
  - `'model_path'` (str): Path to model file
  - `'model_type'` (str): Class name of model
  - `'has_probability'` (bool): Probability support
  - `'is_pipeline'` (bool): Is sklearn Pipeline
  - `'model_size_mb'` (float): File size in MB
  - `'n_features'` (int, optional): Number of input features

#### 3.6.2 Functions

##### predict_single

```python
def predict_single(
    model_path: str,
    features: Dict[str, Any],
    return_proba: bool = True
) -> Dict[str, Any]
```

Convenience function for one-off single prediction.

**Parameters**:
- `model_path` (str): Path to model file
- `features` (Dict[str, Any]): Feature dictionary
- `return_proba` (bool): Return probabilities

**Returns**:
- `dict`: Prediction result

##### predict_batch

```python
def predict_batch(
    model_path: str,
    features_list: List[Dict[str, Any]],
    return_proba: bool = True,
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]
```

Convenience function for one-off batch prediction.

**Parameters**:
- `model_path` (str): Path to model
- `features_list` (List[Dict]): Feature list
- `return_proba` (bool): Return probabilities
- `batch_size` (Optional[int]): Batch size

**Returns**:
- `List[dict]`: Prediction results

##### predict_from_csv

```python
def predict_from_csv(
    model_path: str,
    input_csv: str,
    output_csv: Optional[str] = None,
    return_proba: bool = True
) -> pd.DataFrame
```

Predicts from CSV file.

**Parameters**:
- `model_path` (str): Path to model
- `input_csv` (str): Input CSV path
- `output_csv` (Optional[str]): Output CSV path (if None, doesn't save)
- `return_proba` (bool): Include probabilities

**Returns**:
- `pd.DataFrame`: Results DataFrame

**Side Effects**:
- Saves to output_csv if provided

**Example**:
```python
df_results = predict_from_csv(
    'models/best_model.pkl',
    'input.csv',
    'output.csv'
)
```

---

### 3.7 api_server.py

**Purpose**: FastAPI REST API server for model serving.

**Location**: `src/api_server.py`

**Dependencies**: `fastapi`, `pydantic`, `uvicorn`, `predict.py`

#### 3.7.1 Configuration

**Module-Level Variables**:
- `MODEL_PATH` (str): Path to model file. Default: "models/best_classification.pkl"
- `MODEL_VERSION` (str): API version. Default: "1.0.0"
- `app` (FastAPI): FastAPI application instance
- `predictor` (Optional[ModelPredictor]): Global predictor instance

#### 3.7.2 Data Models

##### MovieFeatures

```python
class MovieFeatures(BaseModel)
```

Pydantic model for movie feature validation.

**Fields**:
- `director_mean_rating` (float): Range 0-10, required
- `director_total_films` (int): >= 0, required
- `cast_mean_rating` (float): Range 0-10, required
- `cast_total_films` (int): >= 0, required
- `writer_mean_rating` (float): Range 0-10, required
- `writer_total_films` (int): >= 0, required
- `runtimeMinutes` (int): Range 1-500, required
- `startYear` (int): Range 1900-2050, required
- `genre_Action` (int): 0 or 1, optional
- `genre_Adventure` (int): 0 or 1, optional
- (additional genre fields...)

**Validation**:
- Automatic type checking
- Range validation
- Required field enforcement

##### BatchPredictionRequest

```python
class BatchPredictionRequest(BaseModel)
```

Request model for batch predictions.

**Fields**:
- `movies` (List[MovieFeatures]): List of movies, 1-1000 items
- `return_proba` (bool): Include probabilities. Default: True

##### PredictionResponse

```python
class PredictionResponse(BaseModel)
```

Response model for predictions.

**Fields**:
- `prediction` (int): 0 (Low Risk) or 1 (High Risk)
- `probability` (Optional[float]): Risk probability (0-1)
- `confidence` (Optional[float]): Confidence score (0-1)
- `risk_level` (str): Risk level label ("LOW", "MEDIUM", "HIGH")
- `recommendation` (str): Investment recommendation

##### BatchPredictionResponse

```python
class BatchPredictionResponse(BaseModel)
```

Response model for batch predictions.

**Fields**:
- `predictions` (List[PredictionResponse]): List of predictions
- `total_predictions` (int): Total count
- `processing_time_ms` (float): Processing time in milliseconds

#### 3.7.3 API Endpoints

##### GET /

```python
@app.get("/")
async def root() -> dict
```

Root endpoint returning API information.

**Returns**:
- `dict`: Service metadata

##### GET /health

```python
@app.get("/health")
async def health_check() -> dict
```

Health check endpoint.

**Returns**:
- `dict`: Health status with keys:
  - `'status'` (str): "healthy"
  - `'model_loaded'` (bool): Model loaded flag
  - `'timestamp'` (str): ISO timestamp
  - `'version'` (str): API version

**Raises**:
- `HTTPException 503`: If model not loaded

##### GET /model/info

```python
@app.get("/model/info")
async def get_model_info() -> dict
```

Returns model metadata.

**Returns**:
- `dict`: Model information

##### POST /predict/single

```python
@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single_movie(features: MovieFeatures) -> PredictionResponse
```

Predicts risk for single movie.

**Request Body**:
- `features` (MovieFeatures): Movie features

**Returns**:
- `PredictionResponse`: Prediction result

**Raises**:
- `HTTPException 503`: Service unavailable
- `HTTPException 500`: Prediction failed

**Example Request**:
```json
{
  "director_mean_rating": 8.5,
  "director_total_films": 15,
  "cast_mean_rating": 7.8,
  "cast_total_films": 35,
  "writer_mean_rating": 7.5,
  "writer_total_films": 10,
  "runtimeMinutes": 148,
  "startYear": 2023,
  "genre_Action": 1,
  "genre_Drama": 1
}
```

**Example Response**:
```json
{
  "prediction": 0,
  "probability": 0.12,
  "confidence": 0.88,
  "risk_level": "LOW",
  "recommendation": "Low risk - safe to invest"
}
```

##### POST /predict/batch

```python
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_movies(request: BatchPredictionRequest) -> BatchPredictionResponse
```

Predicts risk for multiple movies.

**Request Body**:
- `request` (BatchPredictionRequest): Batch request

**Returns**:
- `BatchPredictionResponse`: Batch results

**Example Request**:
```json
{
  "movies": [
    {"director_mean_rating": 8.5, ...},
    {"director_mean_rating": 7.2, ...}
  ],
  "return_proba": true
}
```

---

## 4. DATA STRUCTURES

### 4.1 Core DataFrames

#### Preprocessed DataFrame Schema

**Columns**:
- Movie Identifiers:
  - `tconst` (str): IMDb title ID
  - `primaryTitle` (str): Movie title

- Metadata:
  - `startYear` (int): Release year
  - `runtimeMinutes` (int): Duration in minutes
  - `genres` (str): Comma-separated genre list
  - `genre_list` (list): Parsed genre list

- Person Data:
  - `directors_nconst` (list): Director IMDb IDs
  - `writers_nconst` (list): Writer IMDb IDs
  - `cast_nconst` (list): Cast IMDb IDs
  - `director_names` (list): Director names
  - `cast_names` (list): Cast names

- Person Statistics:
  - `director_mean_rating` (float): Director average rating (0-10)
  - `director_total_films` (int): Director filmography count
  - `writer_mean_rating` (float): Writer average rating (0-10)
  - `writer_total_films` (int): Writer filmography count
  - `cast_mean_rating` (float): Cast average rating (0-10)
  - `cast_total_films` (int): Cast filmography count

- Ratings:
  - `averageRating` (float): IMDb rating (0-10)
  - `numVotes` (int): Vote count

- Target Labels:
  - `is_success` (int): 1 if rating >= 7.0 and votes >= 30000, else 0
  - `is_risky` (int): Inverse of is_success

### 4.2 Feature Matrix (X)

**Shape**: (n_samples, n_features)

**Features** (typically 25-30 columns):
- Person stats: director_mean_rating, director_total_films, writer_mean_rating, writer_total_films, cast_mean_rating, cast_total_films
- Movie metadata: runtimeMinutes, startYear
- Genre indicators: genre_Action, genre_Drama, genre_Comedy, etc. (one-hot encoded)

### 4.3 API Request/Response Formats

See section 3.7.2 for detailed field specifications.

---

## 5. LOCAL DEPLOYMENT GUIDE

### 5.1 System Requirements

**Hardware**:
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Disk: 5GB free space
- GPU: Optional (for deep learning models)

**Software**:
- OS: macOS, Linux, or Windows
- Python: 3.10 or higher
- pip: Latest version

### 5.2 Installation Steps

#### Step 1: Clone or Download Project

```bash
cd /path/to/project
```

#### Step 2: Create Virtual Environment

**Option A: venv**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

**Option B: conda**
```bash
conda create -n movie-predict python=3.10
conda activate movie-predict
```

#### Step 3: Install Dependencies

**Core Dependencies**:
```bash
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install xgboost==1.7.6
pip install joblib==1.3.1
```

**For API Server**:
```bash
pip install fastapi==0.101.0
pip install uvicorn[standard]==0.23.2
pip install pydantic==2.1.1
```

**For Deep Learning (Optional)**:
```bash
pip install torch==2.0.1
pip install pytorch-lightning==2.0.6
```

**All at Once**:
```bash
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
python -c "import pandas; import sklearn; import xgboost; print('OK')"
```

### 5.3 Data Preparation

#### Required Data Files

Place IMDb data files in `data/` directory:
- title.basics.tsv
- title.ratings.tsv
- title.crew.tsv
- title.principals.tsv
- name.basics.tsv
- title.akas.tsv

**Data Source**: IMDb Non-Commercial Datasets (https://datasets.imdbws.com/)

#### Data Preprocessing

```bash
cd src/
python -c "from preprocess import preprocess_pipeline; df = preprocess_pipeline('../data/'); df.to_csv('../data/preprocessed.csv', index=False)"
```

This creates `data/preprocessed.csv` for subsequent use.

### 5.4 Model Training

#### Option 1: Train All Models (Recommended)

```bash
cd src/
python main.py --mode evaluate --data-path ../data/
```

This will:
1. Preprocess data
2. Train all 6 models
3. Evaluate and compare
4. Save best models to `evaluation_results/best_models/`

**Output**:
```
evaluation_results/
├── best_models/
│   ├── best_classification.pkl
│   └── best_regression.pkl
├── classification_results.csv
├── regression_results.csv
├── deep_learning_results.csv
├── final_comparison.csv
└── best_models_metadata.json
```

#### Option 2: Train Specific Model Type

**Classification**:
```bash
python main.py --mode train --model-type classification --data-path ../data/
```

**Regression**:
```bash
python main.py --mode train --model-type regression --data-path ../data/
```

Models saved to `models/best_classification.pkl` and `models/best_regression.pkl`.

### 5.5 Running Predictions Locally

#### Using main.py (Development)

**Interactive Mode**:
```bash
python main.py --mode interactive --model-path evaluation_results/best_models/best_classification.pkl
```

Follow prompts to enter movie features.

**Batch Prediction**:
```bash
python main.py --mode predict \
  --model-path evaluation_results/best_models/best_classification.pkl \
  --input-file ../test_movies.csv \
  --output-file ../predictions.csv
```

#### Using predict.py (Production)

**CLI Mode**:
```bash
python predict.py --model evaluation_results/best_models/best_classification.pkl \
  --input ../test_movies.csv \
  --output ../predictions.csv
```

**Programmatic Use**:
```python
from predict import ModelPredictor

predictor = ModelPredictor('evaluation_results/best_models/best_classification.pkl')

# Single prediction
result = predictor.predict_single({
    'director_mean_rating': 8.5,
    'director_total_films': 15,
    'cast_mean_rating': 7.8,
    'cast_total_films': 35,
    'writer_mean_rating': 7.5,
    'writer_total_films': 10,
    'runtimeMinutes': 148,
    'startYear': 2023,
    'genre_Action': 1,
    'genre_Drama': 1,
    'genre_Thriller': 0,
    # ... other genre fields (set to 0 if not applicable)
})

print(result)
# Output: {'prediction': 0, 'probability': 0.12, 'confidence': 0.88, 'probabilities': {...}}
```

### 5.6 Running API Server Locally

#### Start Server

**Development Mode** (auto-reload on code changes):
```bash
cd src/
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

**Production Mode** (4 workers):
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Access API Documentation

Open browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Root: http://localhost:8000

#### Test API Endpoints

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Single Prediction**:
```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "director_mean_rating": 8.5,
    "director_total_films": 15,
    "cast_mean_rating": 7.8,
    "cast_total_films": 35,
    "writer_mean_rating": 7.5,
    "writer_total_films": 10,
    "runtimeMinutes": 148,
    "startYear": 2023,
    "genre_Action": 1,
    "genre_Drama": 1,
    "genre_Comedy": 0,
    "genre_Thriller": 0
  }'
```

**Expected Response**:
```json
{
  "prediction": 0,
  "probability": 0.12,
  "confidence": 0.88,
  "risk_level": "LOW",
  "recommendation": "Low risk - safe to invest"
}
```

**Batch Prediction**:
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "movies": [
      {
        "director_mean_rating": 8.5,
        "director_total_films": 15,
        "cast_mean_rating": 7.8,
        "cast_total_films": 35,
        "writer_mean_rating": 7.5,
        "writer_total_films": 10,
        "runtimeMinutes": 148,
        "startYear": 2023,
        "genre_Action": 1,
        "genre_Drama": 1
      },
      {
        "director_mean_rating": 6.5,
        "director_total_films": 3,
        "cast_mean_rating": 6.2,
        "cast_total_films": 10,
        "writer_mean_rating": 6.0,
        "writer_total_films": 2,
        "runtimeMinutes": 95,
        "startYear": 2024,
        "genre_Comedy": 1,
        "genre_Romance": 1
      }
    ],
    "return_proba": true
  }'
```

### 5.7 Troubleshooting

#### Issue: ModuleNotFoundError

**Solution**:
```bash
# Ensure you're in virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue: Model file not found

**Solution**:
```bash
# Check model path
ls -la evaluation_results/best_models/

# Ensure you've run evaluation first
python main.py --mode evaluate --data-path ../data/
```

#### Issue: API server fails to start

**Solution**:
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
uvicorn api_server:app --port 8001
```

#### Issue: Out of memory during training

**Solution**:
```python
# Reduce data size
df_sample = df.sample(n=10000, random_state=42)

# Or reduce batch size for deep learning
train_hybrid_dl_models(batch_size=128)  # Instead of 256
```

---

## 6. USAGE EXAMPLES

### 6.1 Complete Workflow Example

```python
# Step 1: Preprocess data
from preprocess import preprocess_pipeline

df = preprocess_pipeline('../data/')
print(f"Loaded {len(df)} movies")

# Step 2: Feature engineering
from feature import encode_genres, get_feature_target

df_encoded = encode_genres(df)
X, y = get_feature_target(df_encoded, target_type='is_risky')
print(f"Features shape: {X.shape}")

# Step 3: Train models
from train import train_is_risky_models

models = train_is_risky_models(df, save_dir='models/classification')
best_model = models['rf']  # Use Random Forest

# Step 4: Make predictions
from predict import ModelPredictor

predictor = ModelPredictor('models/classification/rf_is_risky.pkl')

new_movie = {
    'director_mean_rating': 8.5,
    'director_total_films': 15,
    'cast_mean_rating': 7.8,
    'cast_total_films': 35,
    'writer_mean_rating': 7.5,
    'writer_total_films': 10,
    'runtimeMinutes': 148,
    'startYear': 2023,
    'genre_Action': 1,
    'genre_Drama': 1,
    'genre_Thriller': 0,
    # ... set all other genres to 0
}

result = predictor.predict_single(new_movie)
print(f"Risk prediction: {result}")
```

### 6.2 Batch Processing Example

```python
import pandas as pd
from predict import predict_from_csv

# Prepare input CSV with required features
input_df = pd.DataFrame([
    {
        'director_mean_rating': 8.5,
        'director_total_films': 15,
        # ... all features
    },
    {
        'director_mean_rating': 7.2,
        'director_total_films': 8,
        # ... all features
    }
])

input_df.to_csv('batch_input.csv', index=False)

# Process batch
results_df = predict_from_csv(
    model_path='models/best_classification.pkl',
    input_csv='batch_input.csv',
    output_csv='batch_output.csv'
)

print(results_df[['prediction', 'probability', 'recommendation']])
```

### 6.3 API Client Example

```python
import requests
import json

API_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{API_URL}/predict/single",
    json={
        "director_mean_rating": 8.5,
        "director_total_films": 15,
        "cast_mean_rating": 7.8,
        "cast_total_films": 35,
        "writer_mean_rating": 7.5,
        "writer_total_films": 10,
        "runtimeMinutes": 148,
        "startYear": 2023,
        "genre_Action": 1,
        "genre_Drama": 1
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

---

## 7. API REFERENCE

### 7.1 Endpoint Summary

| Method | Endpoint | Purpose | Auth Required |
|--------|----------|---------|---------------|
| GET | / | API information | No |
| GET | /health | Health check | No |
| GET | /docs | Swagger UI | No |
| GET | /redoc | ReDoc documentation | No |
| GET | /model/info | Model metadata | No |
| POST | /predict/single | Single prediction | No |
| POST | /predict/batch | Batch prediction | No |

### 7.2 Status Codes

- 200: Success
- 400: Bad Request (validation error)
- 422: Unprocessable Entity (invalid input format)
- 500: Internal Server Error (prediction failed)
- 503: Service Unavailable (model not loaded)

### 7.3 Rate Limits

No rate limiting is configured by default. For production deployment, implement rate limiting using SlowAPI or similar middleware.

### 7.4 Authentication

No authentication is required by default. For production use, implement API key authentication or OAuth2.

---

## APPENDIX A: File Structure

```
project/
├── data/
│   ├── title.basics.tsv
│   ├── title.ratings.tsv
│   ├── title.crew.tsv
│   ├── title.principals.tsv
│   ├── name.basics.tsv
│   └── title.akas.tsv
├── src/
│   ├── preprocess.py
│   ├── feature.py
│   ├── train.py
│   ├── evaluate_models.py
│   ├── main.py
│   ├── predict.py
│   ├── api_server.py
│   └── utilities/
│       └── helper.py
├── models/
│   ├── best_classification.pkl
│   └── best_regression.pkl
├── evaluation_results/
│   ├── best_models/
│   ├── classification_results.csv
│   ├── regression_results.csv
│   └── final_comparison.csv
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## APPENDIX B: Model Performance Benchmarks

Based on typical IMDb dataset:

| Model | F1-Score | AUC-ROC | R-squared | MAE |
|-------|----------|---------|-----------|-----|
| Logistic Regression | 0.76-0.78 | 0.82-0.84 | N/A | N/A |
| Random Forest Classifier | 0.78-0.82 | 0.85-0.87 | N/A | N/A |
| Random Forest Regressor | N/A | N/A | 0.60-0.65 | 0.9-1.1 |
| XGBoost Regressor | N/A | N/A | 0.65-0.70 | 0.85-1.0 |
| SimplerDNN | 0.72-0.75 | 0.78-0.82 | 0.62-0.67 | 0.95-1.05 |
| HybridMovieModel | 0.75-0.78 | 0.80-0.85 | 0.68-0.72 | 0.88-0.98 |

---

## APPENDIX C: Glossary

- **IMDb**: Internet Movie Database
- **nconst**: IMDb person identifier (e.g., nm0000123)
- **tconst**: IMDb title identifier (e.g., tt0111161)
- **is_risky**: Binary target: 1 if movie is high-risk investment, 0 if low-risk
- **Multi-output regression**: Model predicting multiple targets simultaneously
- **Person statistics**: Historical performance metrics for directors/cast/writers
- **Genre encoding**: Converting categorical genre data to binary features
- **Pipeline**: scikit-learn Pipeline object combining preprocessing and model

---

END OF TECHNICAL DOCUMENTATION
