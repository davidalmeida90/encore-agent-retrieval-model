"""make section columns text

Revision ID: d2e4b7a9c1f3
Revises: c5a8f1d6e2b9
Create Date: 2026-06-05 23:05:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d2e4b7a9c1f3"
down_revision: Union[str, Sequence[str], None] = "c5a8f1d6e2b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "document_chunks",
        "section",
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
    )
    op.alter_column(
        "message_citations",
        "section",
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "message_citations",
        "section",
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
    )
    op.alter_column(
        "document_chunks",
        "section",
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
    )
