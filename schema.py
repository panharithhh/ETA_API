from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionInput(BaseModel):
    order_id : int
    accept_time : datetime 
    accept_gps_lng: float
    accept_gps_lat: float
    delivery_gps_lng: float
    delivery_gps_lat: float

