import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class VehicleType(str, enum.Enum):
    motorbike = "motorbike"
    tuktuk = "tuktuk"
    van = "van"
    container = "container"


class PackageStatus(str, enum.Enum):
    created = "created"
    pending_pickup = "pending_pickup"
    picked_up = "picked_up"
    at_origin_branch = "at_origin_branch"
    arrived_at_warehouse = "arrived_at_warehouse"
    confirmed = "confirmed"
    rejected = "rejected"
    # Mid-mile statuses
    at_warehouse = "at_warehouse"
    in_transit_w2w = "in_transit_w2w"
    out_for_delivery = "out_for_delivery"


class ConfirmationAction(str, enum.Enum):
    confirm = "confirm"
    reject = "reject"


# native_enum=False → VARCHAR with CHECK on both PostgreSQL and SQLite (test-friendly)
_vehicle_type_col = SAEnum(VehicleType, name="vehicle_type", native_enum=False)
_pkg_status_col = SAEnum(PackageStatus, name="package_status", native_enum=False)
_confirm_action_col = SAEnum(ConfirmationAction, name="confirmation_action", native_enum=False)


class Branch(Base):
    __tablename__ = "branches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vehicle_type: Mapped[VehicleType] = mapped_column(_vehicle_type_col, nullable=False)
    license_plate: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    driver: Mapped[Optional["Driver"]] = relationship("Driver", back_populates="vehicle", uselist=False)


class Driver(Base):
    __tablename__ = "drivers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    current_lat: Mapped[Optional[float]] = mapped_column(Float)
    current_lng: Mapped[Optional[float]] = mapped_column(Float)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    vehicle_id: Mapped[Optional[int]] = mapped_column(ForeignKey("vehicles.id"))

    vehicle: Mapped[Optional[Vehicle]] = relationship("Vehicle", back_populates="driver")


class Package(Base):
    __tablename__ = "packages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_phone: Mapped[str] = mapped_column(String(20), nullable=False)
    receiver_phone: Mapped[str] = mapped_column(String(20), nullable=False)
    receiver_lat: Mapped[float] = mapped_column(Float, nullable=False)
    receiver_lng: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    max_dimension_cm: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[PackageStatus] = mapped_column(_pkg_status_col, default=PackageStatus.created, nullable=False)

    # Branch tracking — current_branch_id is separate from origin/destination
    origin_branch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("branches.id"))
    destination_branch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("branches.id"))
    current_branch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("branches.id"))

    assigned_driver_id: Mapped[Optional[int]] = mapped_column(ForeignKey("drivers.id"))
    assigned_vehicle_id: Mapped[Optional[int]] = mapped_column(ForeignKey("vehicles.id"))

    # Container-only scheduling fields
    pickup_window_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    pickup_window_end: Mapped[Optional[datetime]] = mapped_column(DateTime)

    customer_email: Mapped[Optional[str]] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    events: Mapped[list["PackageEvent"]] = relationship("PackageEvent", back_populates="package")


class PackageEvent(Base):
    """Append-only audit log — never update rows, only insert."""

    __tablename__ = "package_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    package_id: Mapped[int] = mapped_column(ForeignKey("packages.id"), nullable=False)
    status: Mapped[PackageStatus] = mapped_column(_pkg_status_col, nullable=False)
    driver_id: Mapped[Optional[int]] = mapped_column(ForeignKey("drivers.id"))
    branch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("branches.id"))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    package: Mapped[Package] = relationship("Package", back_populates="events")


class ConfirmationToken(Base):
    """Single-use token emailed to the customer for confirm/reject actions."""

    __tablename__ = "confirmation_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    package_id: Mapped[int] = mapped_column(ForeignKey("packages.id"), nullable=False)
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    action: Mapped[ConfirmationAction] = mapped_column(_confirm_action_col, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class WarehouseInventory(Base):
    """Tracks which packages are physically sitting at which branch."""

    __tablename__ = "warehouse_inventory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    package_id: Mapped[int] = mapped_column(ForeignKey("packages.id"), nullable=False, index=True)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), nullable=False, index=True)
    arrived_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    departed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    package: Mapped[Package] = relationship("Package")
    branch: Mapped[Branch] = relationship("Branch")
