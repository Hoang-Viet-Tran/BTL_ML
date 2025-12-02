"""
FastAPI Production Server for Movie Risk Prediction

High-performance REST API for serving ML predictions.

Features:
- Fast async processing
- Auto-generated documentation (Swagger UI)
- Input validation
- Health checks
- Model versioning support
- Request logging

Quick Start:
    # Install dependencies
    pip install fastapi uvicorn pydantic

    # Run server
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    
    # View docs: http://localhost:8000/docs

Endpoints:
    POST /predict/single    - Single movie prediction
    POST /predict/batch     - Batch predictions
    GET  /health            - Health check
    GET  /model/info        - Model information
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from predict import ModelPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# -------------------------APP CONFIGURATION------------------------
# ------------------------------------------------------------------

# Model configuration
MODEL_PATH = "models/best_classification.pkl"  # Change to your model path
MODEL_VERSION = "1.0.0"

# Initialize FastAPI app
app = FastAPI(
    title="Movie Risk Prediction API",
    description="Production-grade API for predicting movie investment risk",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model predictor (loaded once at startup)
predictor: Optional[ModelPredictor] = None


# ------------------------------------------------------------------
# --------------------------REQUEST MODELS--------------------------
# ------------------------------------------------------------------

class MovieFeatures(BaseModel):
    """
    Movie features for prediction.
    
    All numeric features between 0-10 rating scale or positive integers.
    Genre fields are binary (0 or 1).
    """
    # Person features
    director_mean_rating: float = Field(..., ge=0, le=10, description="Director's average rating (0-10)")
    director_total_films: int = Field(..., ge=0, description="Director's total films")
    cast_mean_rating: float = Field(..., ge=0, le=10, description="Cast average rating (0-10)")
    cast_total_films: int = Field(..., ge=0, description="Cast total films")
    writer_mean_rating: float = Field(..., ge=0, le=10, description="Writer average rating (0-10)")
    writer_total_films: int = Field(..., ge=0, description="Writer total films")
    
    # Movie metadata
    runtimeMinutes: int = Field(..., ge=1, le=500, description="Movie runtime in minutes")
    startYear: int = Field(..., ge=1900, le=2050, description="Release year")
    
    # Genres (example - add all your genres)
    genre_Action: Optional[int] = Field(0, ge=0, le=1)
    genre_Adventure: Optional[int] = Field(0, ge=0, le=1)
    genre_Animation: Optional[int] = Field(0, ge=0, le=1)
    genre_Biography: Optional[int] = Field(0, ge=0, le=1)
    genre_Comedy: Optional[int] = Field(0, ge=0, le=1)
    genre_Crime: Optional[int] = Field(0, ge=0, le=1)
    genre_Documentary: Optional[int] = Field(0, ge=0, le=1)
    genre_Drama: Optional[int] = Field(0, ge=0, le=1)
    genre_Family: Optional[int] = Field(0, ge=0, le=1)
    genre_Fantasy: Optional[int] = Field(0, ge=0, le=1)
    genre_History: Optional[int] = Field(0, ge=0, le=1)
    genre_Horror: Optional[int] = Field(0, ge=0, le=1)
    genre_Music: Optional[int] = Field(0, ge=0, le=1)
    genre_Mystery: Optional[int] = Field(0, ge=0, le=1)
    genre_Romance: Optional[int] = Field(0, ge=0, le=1)
    genre_SciFi: Optional[int] = Field(0, ge=0, le=1, alias="genre_Sci-Fi")
    genre_Sport: Optional[int] = Field(0, ge=0, le=1)
    genre_Thriller: Optional[int] = Field(0, ge=0, le=1)
    genre_War: Optional[int] = Field(0, ge=0, le=1)
    genre_Western: Optional[int] = Field(0, ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
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
                "genre_Thriller": 1
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request with multiple movies."""
    movies: List[MovieFeatures] = Field(..., min_items=1, max_items=1000, description="List of movies to predict")
    return_proba: bool = Field(True, description="Include prediction probabilities")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    prediction: int = Field(..., description="Prediction: 0 (Low Risk) or 1 (High Risk)")
    probability: Optional[float] = Field(None, description="Probability of high risk (0-1)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    risk_level: str = Field(..., description="Risk level label")
    recommendation: str = Field(..., description="Investment recommendation")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.12,
                "confidence": 0.88,
                "risk_level": "LOW",
                "recommendation": "Safe to invest"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_predictions: int
    processing_time_ms: float


# ------------------------------------------------------------------
# --------------------------STARTUP/SHUTDOWN------------------------
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    global predictor
    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        predictor = ModelPredictor(MODEL_PATH, enable_cache=True)
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model info: {predictor.get_model_info()}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")


# ------------------------------------------------------------------
# ----------------------------ENDPOINTS-----------------------------
# ------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "Movie Risk Prediction API",
        "version": MODEL_VERSION,
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.utcnow().isoformat(),
        "version": MODEL_VERSION
    }


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model metadata
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return predictor.get_model_info()


@app.post("/predict/single", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single_movie(features: MovieFeatures):
    """
    Predict risk for a single movie.
    
    Args:
        features: Movie features
        
    Returns:
        Prediction result with risk assessment
        
    Example Request:
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
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert to dict
        features_dict = features.dict()
        
        # Predict
        result = predictor.predict_single(features_dict, return_proba=True)
        
        # Format response
        prediction = int(result['prediction'])
        probability = result.get('probability', None)
        confidence = result.get('confidence', None)
        
        # Determine risk level and recommendation
        if prediction == 1:
            risk_level = "HIGH"
            recommendation = "High risk - thorough review recommended"
        else:
            if probability and probability < 0.2:
                risk_level = "LOW"
                recommendation = "Low risk - safe to invest"
            elif probability and probability < 0.4:
                risk_level = "MEDIUM-LOW"
                recommendation = "Moderate-low risk - generally safe"
            else:
                risk_level = "MEDIUM"
                recommendation = "Medium risk - careful consideration advised"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            risk_level=risk_level,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_movies(request: BatchPredictionRequest):
    """
    Predict risk for multiple movies (batch processing).
    
    Args:
        request: Batch prediction request with list of movies
        
    Returns:
        Batch prediction results
        
    Example Request:
        ```json
        {
          "movies": [
            {
              "director_mean_rating": 8.5,
              "cast_mean_rating": 7.8,
              ...
            },
            {
              "director_mean_rating": 7.2,
              "cast_mean_rating": 6.9,
              ...
            }
          ],
          "return_proba": true
        }
        ```
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = datetime.now()
        
        # Convert to list of dicts
        features_list = [movie.dict() for movie in request.movies]
        
        # Batch predict
        results = predictor.predict_batch(
            features_list,
            return_proba=request.return_proba
        )
        
        # Format responses
        predictions = []
        for result in results:
            prediction = int(result['prediction'])
            probability = result.get('probability', None)
            confidence = result.get('confidence', None)
            
            if prediction == 1:
                risk_level = "HIGH"
                recommendation = "High risk - thorough review recommended"
            else:
                if probability and probability < 0.2:
                    risk_level = "LOW"
                    recommendation = "Low risk - safe to invest"
                else:
                    risk_level = "MEDIUM"
                    recommendation = "Medium risk - careful consideration advised"
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                risk_level=risk_level,
                recommendation=recommendation
            ))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ------------------------------------------------------------------
# -----------------------------RUN SERVER---------------------------
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development)
        log_level="info"
    )
