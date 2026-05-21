from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class DeliveryItem(BaseModel):
    order_id: int
    accept_time: datetime | None = None
    accept_gps_lng: float
    accept_gps_lat: float
    delivery_gps_lng: float
    delivery_gps_lat: float

class BatchPredictionInput(BaseModel):
    courier_id: int
    accept_time: datetime | None = None
    deliveries: List[DeliveryItem]

class RetrainItem(BaseModel):
    order_id: int
    courier_id: int
    accept_time: datetime
    delivery_time: datetime
    accept_gps_lng: float
    accept_gps_lat: float
    delivery_gps_lng: float
    delivery_gps_lat: float
    package_count: int

class AcceptDeliveryInput(BaseModel):
    order_id: int

class PredictionInput(BaseModel):
    order_id: int
    accept_time: datetime
    accept_gps_lng: float
    accept_gps_lat: float
    delivery_gps_lng: float
    delivery_gps_lat: float

class PredictionInputPickUP(BaseModel):
    order_id: int
    accept_time: datetime
    accept_gps_lng: float
    accept_gps_lat: float
    pickup_gps_lng: float
    pickup_gps_lat: float

class StopInput(BaseModel):
    order_id: int
    accept_gps_lat: float
    accept_gps_lng: float
    delivery_gps_lat: float
    delivery_gps_lng: float
    accept_time: Optional[datetime] = None

class DriverInput(BaseModel):
    driver_id: int
    current_lat: float
    current_lng: float

class OrderInput(BaseModel):
    order_id: int
    accept_time: Optional[datetime] = None
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float

class AutoMapingInput(BaseModel):
    accept_time: Optional[datetime] = None
    drivers: List[DriverInput]
    stops: Optional[List[StopInput]] = None
    orders: Optional[List[OrderInput]] = None

class C2CPredictInput(BaseModel):
    order_id: int
    pickup_lat: float
    pickup_lon: float
    delivery_lat: float
    delivery_lon: float
    courier_rating: float

class C2CConfirmInput(BaseModel):
    order_id: int
