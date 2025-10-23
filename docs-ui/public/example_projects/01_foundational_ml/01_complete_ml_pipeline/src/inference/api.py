"""
FastAPI application for serving customer churn predictions.
Provides RESTful API endpoints for single and batch predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging
import asyncio
import time
from datetime import datetime
import json
from pathlib import Path
import uvicorn

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector
from .predict import ChurnPredictor
from .model_loader import ModelLoader
from ..utils.database import DatabaseManager

logger = get_logger(__name__)

# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    """Model for customer features."""
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., ge=18, le=120, description="Customer age")
    tenure: int = Field(..., ge=0, description="Months with company")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    gender: str = Field(..., description="Customer gender")
    contract_type: str = Field(..., description="Contract type")
    payment_method: str = Field(..., description="Payment method")
    phone_service: bool = Field(..., description="Has phone service")
    internet_service: str = Field(..., description="Internet service type")
    online_security: bool = Field(..., description="Has online security")
    tech_support: bool = Field(..., description="Has tech support")
    senior_citizen: bool = Field(..., description="Is senior citizen")
    partner: bool = Field(..., description="Has partner")
    dependents: bool = Field(..., description="Has dependents")

class BatchPredictionRequest(BaseModel):
    """Model for batch prediction requests."""
    customers: List[CustomerFeatures] = Field(..., description="List of customers to predict")
    return_probabilities: bool = Field(True, description="Return prediction probabilities")

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    customer_id: str
    prediction: int
    probability: float
    confidence: str
    prediction_time: datetime
    model_version: str
    features_used: List[str]

class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""
    predictions: List[PredictionResponse]
    total_predictions: int
    avg_probability: float
    churn_count: int
    processing_time: float
    model_version: str

class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    database_connected: bool
    version: str

class MetricsResponse(BaseModel):
    """Model for metrics response."""
    predictions_count: int
    avg_latency: float
    error_rate: float
    model_accuracy: Optional[float]
    last_updated: datetime

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using ensemble ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor: Optional[ChurnPredictor] = None
metrics_collector: Optional[MetricsCollector] = None
db_manager: Optional[DatabaseManager] = None
model_loader: Optional[ModelLoader] = None

# Dependency for getting predictor
def get_predictor() -> ChurnPredictor:
    """Get the current predictor instance."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor

# Dependency for getting metrics collector
def get_metrics_collector() -> MetricsCollector:
    """Get the metrics collector instance."""
    if metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics not available")
    return metrics_collector

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global predictor, metrics_collector, db_manager, model_loader

    try:
        logger.info("Starting up Churn Prediction API...")

        # Initialize metrics collector
        metrics_collector = MetricsCollector()
        logger.info("Metrics collector initialized")

        # Initialize database manager
        db_manager = DatabaseManager()
        if db_manager.test_connection():
            logger.info("Database connection established")
        else:
            logger.warning("Database connection failed")

        # Initialize model loader
        model_loader = ModelLoader()
        model_path = "models/ensemble_model.joblib"
        preprocessor_path = "models/preprocessor.joblib"

        if Path(model_path).exists() and Path(preprocessor_path).exists():
            predictor = model_loader.load_predictor(model_path, preprocessor_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model files not found, predictor not initialized")

        logger.info("API startup completed")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down API...")
    if metrics_collector:
        metrics_collector.close()
    if db_manager:
        db_manager.close()
    logger.info("API shutdown completed")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model_loaded = predictor is not None
        db_connected = db_manager is not None and db_manager.test_connection()

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            timestamp=datetime.now(),
            model_loaded=model_loaded,
            database_connected=db_connected,
            version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            model_loaded=False,
            database_connected=False,
            version="1.0.0"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(
    customer: CustomerFeatures,
    predictor: ChurnPredictor = Depends(get_predictor),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Predict churn for a single customer.

    Args:
        customer: Customer features
        predictor: Churn predictor instance
        metrics: Metrics collector

    Returns:
        Prediction response with probability and confidence
    """
    start_time = time.time()

    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])

        # Make prediction
        result = predictor.predict(df)

        # Calculate confidence
        probability = result['probability']
        if probability < 0.3:
            confidence = "Low"
        elif probability < 0.7:
            confidence = "Medium"
        else:
            confidence = "High"

        # Record metrics
        processing_time = time.time() - start_time
        metrics.record_prediction(processing_time, result['prediction'])

        # Log prediction to database if available
        if db_manager:
            await log_prediction_to_db(customer.customer_id, result, processing_time)

        return PredictionResponse(
            customer_id=customer.customer_id,
            prediction=result['prediction'],
            probability=probability,
            confidence=confidence,
            prediction_time=datetime.now(),
            model_version=getattr(predictor, 'model_version', 'unknown'),
            features_used=getattr(predictor, 'feature_names', [])
        )

    except Exception as e:
        logger.error(f"Prediction failed for customer {customer.customer_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_churn(
    request: BatchPredictionRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Predict churn for multiple customers.

    Args:
        request: Batch prediction request
        predictor: Churn predictor instance
        metrics: Metrics collector

    Returns:
        Batch prediction response
    """
    start_time = time.time()

    try:
        # Convert to DataFrame
        customers_data = [customer.dict() for customer in request.customers]
        df = pd.DataFrame(customers_data)

        # Make batch predictions
        results = predictor.predict_batch(df)

        # Create response objects
        predictions = []
        churn_count = 0
        total_probability = 0.0

        for i, (customer, result) in enumerate(zip(request.customers, results)):
            probability = result['probability']
            total_probability += probability

            if result['prediction'] == 1:
                churn_count += 1

            # Calculate confidence
            if probability < 0.3:
                confidence = "Low"
            elif probability < 0.7:
                confidence = "Medium"
            else:
                confidence = "High"

            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                prediction=result['prediction'],
                probability=probability,
                confidence=confidence,
                prediction_time=datetime.now(),
                model_version=getattr(predictor, 'model_version', 'unknown'),
                features_used=getattr(predictor, 'feature_names', [])
            ))

        # Calculate metrics
        processing_time = time.time() - start_time
        avg_probability = total_probability / len(request.customers)

        # Record metrics
        metrics.record_batch_prediction(processing_time, len(request.customers))

        # Log predictions to database if available
        if db_manager:
            await asyncio.gather(*[
                log_prediction_to_db(customer.customer_id, result, processing_time / len(request.customers))
                for customer, result in zip(request.customers, results)
            ])

        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(request.customers),
            avg_probability=avg_probability,
            churn_count=churn_count,
            processing_time=processing_time,
            model_version=getattr(predictor, 'model_version', 'unknown')
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(metrics: MetricsCollector = Depends(get_metrics_collector)):
    """
    Get API metrics and performance statistics.

    Args:
        metrics: Metrics collector

    Returns:
        Current metrics
    """
    try:
        metrics_data = metrics.get_metrics()

        return MetricsResponse(
            predictions_count=metrics_data['predictions_count'],
            avg_latency=metrics_data['avg_latency'],
            error_rate=metrics_data['error_rate'],
            model_accuracy=metrics_data.get('model_accuracy'),
            last_updated=datetime.now()
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@app.get("/model/info")
async def get_model_info(predictor: ChurnPredictor = Depends(get_predictor)):
    """
    Get information about the loaded model.

    Args:
        predictor: Churn predictor instance

    Returns:
        Model information
    """
    try:
        model_info = {
            "model_type": type(predictor.model).__name__ if hasattr(predictor, 'model') else "unknown",
            "model_version": getattr(predictor, 'model_version', 'unknown'),
            "feature_count": len(getattr(predictor, 'feature_names', [])),
            "ensemble_method": getattr(predictor.model, 'ensemble_method', 'unknown') if hasattr(predictor, 'model') else "unknown",
            "models_in_ensemble": len(getattr(predictor.model, 'models', {})) if hasattr(predictor, 'model') else 0,
            "training_date": getattr(predictor, 'training_date', 'unknown'),
            "last_updated": getattr(predictor, 'last_updated', 'unknown')
        }

        return model_info

    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """
    Reload the model in the background.

    Args:
        background_tasks: Background task manager

    Returns:
        Reload status
    """
    try:
        background_tasks.add_task(reload_model_task)
        return {"status": "reloading", "message": "Model reload initiated in background"}

    except Exception as e:
        logger.error(f"Failed to initiate model reload: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reload model")

async def reload_model_task():
    """Background task to reload the model."""
    global predictor

    try:
        logger.info("Starting model reload...")

        if model_loader:
            new_predictor = model_loader.load_predictor(
                "models/ensemble_model.joblib",
                "models/preprocessor.joblib"
            )

            if new_predictor:
                predictor = new_predictor
                logger.info("Model reloaded successfully")
            else:
                logger.error("Failed to reload model")

    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")

async def log_prediction_to_db(customer_id: str, result: Dict[str, Any], processing_time: float):
    """
    Log prediction to database.

    Args:
        customer_id: Customer identifier
        result: Prediction result
        processing_time: Processing time in seconds
    """
    try:
        prediction_data = {
            'customer_id': customer_id,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'processing_time': processing_time,
            'timestamp': datetime.now(),
            'model_version': getattr(predictor, 'model_version', 'unknown')
        }

        await db_manager.insert_prediction(prediction_data)

    except Exception as e:
        logger.error(f"Failed to log prediction to database: {str(e)}")

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )