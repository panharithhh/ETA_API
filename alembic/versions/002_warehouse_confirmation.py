"""Warehouse arrival confirmation: new statuses, customer_email, confirmation_tokens

Revision ID: 002
Revises: 001
Create Date: 2026-05-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_STATUSES = (
    "created", "pending_pickup", "picked_up", "at_origin_branch",
    "arrived_at_warehouse", "confirmed", "rejected",
)
_OLD_STATUSES = ("created", "pending_pickup", "picked_up", "at_origin_branch")
_ACTIONS = ("confirm", "reject")


def upgrade() -> None:
    # Drop the old status CHECK constraints on both tables (auto-named by PG, so we
    # search by constraint definition rather than name)
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

    op.add_column("packages", sa.Column("customer_email", sa.String(255)))

    op.create_table(
        "confirmation_tokens",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("package_id", sa.Integer, sa.ForeignKey("packages.id"), nullable=False),
        sa.Column("token", sa.String(64), nullable=False, unique=True),
        sa.Column(
            "action",
            sa.String(10),
            sa.CheckConstraint(f"action IN {_ACTIONS}"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("used", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_confirmation_tokens_token", "confirmation_tokens", ["token"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_confirmation_tokens_token", "confirmation_tokens")
    op.drop_table("confirmation_tokens")
    op.drop_column("packages", "customer_email")

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
