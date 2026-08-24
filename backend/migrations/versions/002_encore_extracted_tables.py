"""add document tables

Revision ID: b7f4a1c2d9e8
Revises: 3c81ca3fef26
Create Date: 2026-06-05 17:52:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b7f4a1c2d9e8"
down_revision: Union[str, Sequence[str], None] = "3c81ca3fef26"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_tables",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("table_index", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("units", sa.String(length=255), nullable=True),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column(
            "table_data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("source_html_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["source_documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "table_index",
            name="uq_document_tables_document_table",
        ),
    )
    op.create_index(
        "ix_document_tables_document_id",
        "document_tables",
        ["document_id"],
        unique=False,
    )
    op.execute("ALTER TABLE document_tables ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY document_tables_select_authenticated
        ON document_tables
        FOR SELECT
        TO authenticated
        USING (true)
        """
    )


def downgrade() -> None:
    op.execute(
        'DROP POLICY IF EXISTS "document_tables_select_authenticated" ON document_tables'
    )
    op.execute("ALTER TABLE document_tables DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_document_tables_document_id", table_name="document_tables")
    op.drop_table("document_tables")
