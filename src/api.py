import sys
import os

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from typing import Dict, Any

try:
    from src.config import (
        XGB_MODEL, SCALER_F, FEATURES_F,
        KM_MODEL_F, KM_SCALER_F, SEG_LABELS_F
    )
except ImportError:
    # Fallback paths if config is not available
    XGB_MODEL = "models/xgb_model.joblib"
    SCALER_F = "models/scaler.joblib"
    FEATURES_F = "models/feature_columns.joblib"
    KM_MODEL_F = "models/kmeans_model.joblib"
    KM_SCALER_F = "models/kmeans_scaler.joblib"
    SEG_LABELS_F = "models/segment_labels.joblib"

app = FastAPI(
    title="CLV Prediction API",
    description="Customer Lifetime Value prediction and segmentation API",
    version="1.0.0"
)

@app.get("/")
def root():
    """Root endpoint - redirect info"""
    return {
        "message": "CLV Prediction API",
        "endpoints": {
            "health": "/health",
            "predict_clv": "/predict (POST)",
            "predict_segment": "/segment (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    models_status = {}
    
    # Check if model files exist
    model_files = {
        "clv_model": XGB_MODEL,
        "clv_scaler": SCALER_F,
        "features": FEATURES_F,
        "segment_model": KM_MODEL_F,
        "segment_scaler": KM_SCALER_F
    }
    
    for name, path in model_files.items():
        models_status[name] = os.path.exists(path)
    
    all_healthy = all(models_status.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "models": models_status,
        "message": "All models loaded" if all_healthy else "Some models missing - run training pipeline"
    }

@app.post("/predict")
def predict_clv(features: Dict[str, Any]):
    """
    Predict Customer Lifetime Value
    
    Expected features:
    - recency_days: int (days since last purchase)
    - frequency: int (number of transactions)
    - monetary: float (total spent)
    - avg_order_value: float (average order value)
    - std_order_value: float (standard deviation of order values)
    - avg_days_between_txn: float (average days between transactions)
    """
    try:
        # Load models
        if not all(os.path.exists(f) for f in [XGB_MODEL, SCALER_F, FEATURES_F]):
            raise HTTPException(
                status_code=503, 
                detail="CLV models not available. Please run the training pipeline."
            )
        
        model = joblib.load(XGB_MODEL)
        scaler = joblib.load(SCALER_F)
        feature_cols = joblib.load(FEATURES_F)
        
        # Validate features
        missing_features = [col for col in feature_cols if col not in features]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Prepare data
        X = pd.DataFrame([features])[feature_cols]
        X = X.fillna(0)  # Handle any missing values
        X_scaled = scaler.transform(X)
        
        # Predict
        clv_prediction = float(model.predict(X_scaled)[0])
        
        return {
            "predicted_clv": clv_prediction,
            "currency": "USD",
            "model_version": "1.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/segment")
def predict_segment(features: Dict[str, Any]):
    """
    Predict customer segment
    
    Required features:
    - recency_days: int (days since last purchase)
    - frequency: int (number of transactions) 
    - monetary: float (total spent)
    """
    try:
        # Load models
        if not all(os.path.exists(f) for f in [KM_MODEL_F, KM_SCALER_F]):
            raise HTTPException(
                status_code=503,
                detail="Segmentation models not available. Please run the training pipeline."
            )
        
        km = joblib.load(KM_MODEL_F)
        scaler = joblib.load(KM_SCALER_F)
        
        # Load segment labels if available
        seg_labels_map = None
        if os.path.exists(SEG_LABELS_F):
            try:
                seg_labels_map = joblib.load(SEG_LABELS_F)
            except:
                pass
        
        # Validate required features for segmentation
        required_features = ['recency_days', 'frequency', 'monetary']
        missing_features = [col for col in required_features if col not in features]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features for segmentation: {missing_features}"
            )
        
        # Prepare data
        X = pd.DataFrame([features])[required_features]
        X = X.fillna(0)  # Handle any missing values
        X_scaled = scaler.transform(X)
        
        # Predict segment
        segment_label = int(km.predict(X_scaled)[0])
        
        response = {
            "segment_label": segment_label,
            "model_version": "1.0"
        }
        
        # Add friendly name if available
        if seg_labels_map and segment_label in seg_labels_map:
            response["segment_name"] = seg_labels_map[segment_label]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation error: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """Get information about loaded models"""
    try:
        info = {
            "clv_model": {
                "available": os.path.exists(XGB_MODEL),
                "path": XGB_MODEL
            },
            "segmentation_model": {
                "available": os.path.exists(KM_MODEL_F),
                "path": KM_MODEL_F
            }
        }
        
        # If models exist, get feature information
        if os.path.exists(FEATURES_F):
            features = joblib.load(FEATURES_F)
            info["required_features"] = features
        
        return info
    except Exception as e:
        return {"error": f"Could not load model info: {str(e)}"}
