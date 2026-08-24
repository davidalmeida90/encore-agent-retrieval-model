"""make document table title text

Revision ID: c5a8f1d6e2b9
Revises: b7f4a1c2d9e8
Create Date: 2026-06-05 22:58:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "c5a8f1d6e2b9"
down_revision: Union[str, Sequence[str], None] = "b7f4a1c2d9e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "document_tables",
        "title",
        existing_type=sa.String(length=512),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "document_tables",
        "title",
        existing_type=sa.Text(),
        type_=sa.String(length=512),
        existing_nullable=True,
    )
