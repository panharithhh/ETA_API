"""First-mile tables: branches, vehicles, drivers, packages, package_events

Revision ID: 001
Revises:
Create Date: 2026-05-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VEHICLE_TYPES = ("motorbike", "tuktuk", "van", "container")
PACKAGE_STATUSES = ("created", "pending_pickup", "picked_up", "at_origin_branch")


def upgrade() -> None:
    op.create_table(
        "branches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lng", sa.Float, nullable=False),
    )

    op.create_table(
        "vehicles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "vehicle_type",
            sa.String(20),
            sa.CheckConstraint(f"vehicle_type IN {VEHICLE_TYPES}"),
            nullable=False,
        ),
        sa.Column("license_plate", sa.String(20), nullable=False, unique=True),
        sa.Column("is_available", sa.Boolean, nullable=False, server_default="true"),
    )

    op.create_table(
        "drivers",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("phone", sa.String(20), nullable=False, unique=True),
        sa.Column("current_lat", sa.Float),
        sa.Column("current_lng", sa.Float),
        sa.Column("is_available", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("vehicle_id", sa.Integer, sa.ForeignKey("vehicles.id")),
    )

    op.create_table(
        "packages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("customer_phone", sa.String(20), nullable=False),
        sa.Column("receiver_phone", sa.String(20), nullable=False),
        sa.Column("receiver_lat", sa.Float, nullable=False),
        sa.Column("receiver_lng", sa.Float, nullable=False),
        sa.Column("weight_kg", sa.Float, nullable=False),
        sa.Column("max_dimension_cm", sa.Float, nullable=False),
        sa.Column(
            "status",
            sa.String(30),
            sa.CheckConstraint(f"status IN {PACKAGE_STATUSES}"),
            nullable=False,
            server_default="created",
        ),
        sa.Column("origin_branch_id", sa.Integer, sa.ForeignKey("branches.id")),
        sa.Column("destination_branch_id", sa.Integer, sa.ForeignKey("branches.id")),
        sa.Column("current_branch_id", sa.Integer, sa.ForeignKey("branches.id")),
        sa.Column("assigned_driver_id", sa.Integer, sa.ForeignKey("drivers.id")),
        sa.Column("assigned_vehicle_id", sa.Integer, sa.ForeignKey("vehicles.id")),
        sa.Column("pickup_window_start", sa.DateTime),
        sa.Column("pickup_window_end", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "package_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("package_id", sa.Integer, sa.ForeignKey("packages.id"), nullable=False),
        sa.Column(
            "status",
            sa.String(30),
            sa.CheckConstraint(f"status IN {PACKAGE_STATUSES}"),
            nullable=False,
        ),
        sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.id")),
        sa.Column("branch_id", sa.Integer, sa.ForeignKey("branches.id")),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("package_events")
    op.drop_table("packages")
    op.drop_table("drivers")
    op.drop_table("vehicles")
    op.drop_table("branches")
