"""Mid-mile statuses and warehouse_inventory table

Revision ID: 003
Revises: 002
Create Date: 2026-06-03
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_STATUSES = (
    "created", "pending_pickup", "picked_up", "at_origin_branch",
    "arrived_at_warehouse", "confirmed", "rejected",
    "at_warehouse", "in_transit_w2w", "out_for_delivery",
)
_OLD_STATUSES = (
    "created", "pending_pickup", "picked_up", "at_origin_branch",
    "arrived_at_warehouse", "confirmed", "rejected",
)


def upgrade() -> None:
    # Drop old status CHECK constraints (auto-named by PG, found by definition)
    op.execute("""
        DO $$
        DECLARE r RECORD;
        BEGIN
            FOR r IN (
                SELECT conname, conrelid::regclass::text AS tname
                FROM pg_constraint
                WHERE contype = 'c'
                  AND conrelid IN ('packages'::regclass, 'package_events'::regclass)
                  AND pg_get_constraintdef(oid) LIKE '%status%'
            ) LOOP
                EXECUTE 'ALTER TABLE ' || r.tname
                     || ' DROP CONSTRAINT ' || quote_ident(r.conname);
            END LOOP;
        END;
        $$;
    """)

    op.execute(
        f"ALTER TABLE packages ADD CONSTRAINT packages_status_check "
        f"CHECK (status IN {_NEW_STATUSES})"
    )
    op.execute(
        f"ALTER TABLE package_events ADD CONSTRAINT package_events_status_check "
        f"CHECK (status IN {_NEW_STATUSES})"
    )

    op.create_table(
        "warehouse_inventory",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("package_id", sa.Integer, sa.ForeignKey("packages.id"), nullable=False),
        sa.Column("branch_id", sa.Integer, sa.ForeignKey("branches.id"), nullable=False),
        sa.Column("arrived_at", sa.DateTime, nullable=False),
        sa.Column("departed_at", sa.DateTime),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )
    op.create_index("ix_warehouse_inventory_package", "warehouse_inventory", ["package_id"])
    op.create_index("ix_warehouse_inventory_branch", "warehouse_inventory", ["branch_id"])


def downgrade() -> None:
    op.drop_index("ix_warehouse_inventory_branch", "warehouse_inventory")
    op.drop_index("ix_warehouse_inventory_package", "warehouse_inventory")
    op.drop_table("warehouse_inventory")

    op.execute("ALTER TABLE packages DROP CONSTRAINT IF EXISTS packages_status_check")
    op.execute("ALTER TABLE package_events DROP CONSTRAINT IF EXISTS package_events_status_check")
    op.execute(
        f"ALTER TABLE packages ADD CONSTRAINT packages_status_check "
        f"CHECK (status IN {_OLD_STATUSES})"
    )
    op.execute(
        f"ALTER TABLE package_events ADD CONSTRAINT package_events_status_check "
        f"CHECK (status IN {_OLD_STATUSES})"
    )
