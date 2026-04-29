import pandas as pd
import numpy as np
from fastapi import FastAPI
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import joblib

from model import train_model, predict as predict_fn  # rename import to avoid conflict
from schema import PredictionInput

app = FastAPI()

model = joblib.load("model.pkl")


@app.post("/train_model")
def initial_training():
    train_model()
    global model
    model = joblib.load("model.pkl") 
    return {"status": "training_complete"}


@app.post("/retrain")
def retrain():
    pass


@app.post("/predict")
def predict_endpoint(data: PredictionInput):
    predicted_minutes = predict_fn(data)
    
    now = datetime.now(ZoneInfo("Asia/Phnom_Penh"))
    eta = now + timedelta(minutes=predicted_minutes)
    
    return {
        "order_id": data.order_id,
        "predicted_minutes": round(predicted_minutes, 2),
        "eta": eta.isoformat()
    }
