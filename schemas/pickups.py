from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PickupBookRequest(BaseModel):
    customer_phone: str = Field(..., description="Sender phone number")
    customer_email: Optional[str] = Field(None, description="Sender email — required for warehouse arrival confirmation")
    receiver_phone: str = Field(..., description="Receiver phone number")
    receiver_lat: float = Field(..., description="Receiver GPS latitude — pin drop accepted")
    receiver_lng: float = Field(..., description="Receiver GPS longitude — pin drop accepted")
    weight_kg: float = Field(..., gt=0, description="Package weight in kg")
    max_dimension_cm: float = Field(..., gt=0, description="Largest single dimension in cm")
    origin_branch_id: Optional[str] = Field(None, description="Branch the sender is nearest to")
    destination_branch_id: Optional[str] = Field(None, description="Branch nearest to the receiver")
    # Required only for container bookings (12-hour lead time)
    pickup_window_start: Optional[datetime] = Field(None, description="Container only: earliest pickup datetime")
    pickup_window_end: Optional[datetime] = Field(None, description="Container only: latest pickup datetime")


class ScanRequest(BaseModel):
    """Optional body for /scan — reserved for future QR payload fields."""
    pass


class DropoffRequest(BaseModel):
    branch_id: str = Field(..., description="ID of the origin branch where package is being checked in")
