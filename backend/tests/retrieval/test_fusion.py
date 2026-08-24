from __future__ import annotations

from uuid import UUID

from app.retrieval.fusion import reciprocal_rank_fusion

ID_A = UUID("00000000-0000-0000-0000-000000000001")
ID_B = UUID("00000000-0000-0000-0000-000000000002")
ID_C = UUID("00000000-0000-0000-0000-000000000003")
ID_D = UUID("00000000-0000-0000-0000-000000000004")


def test_reciprocal_rank_fusion_boosts_overlap() -> None:
    fused = reciprocal_rank_fusion(
        [
            [ID_A, ID_B, ID_C],
            [ID_B, ID_A, ID_D],
        ],
        k=60,
    )
    by_id = dict(fused)
    assert by_id[ID_B] > by_id[ID_D]
    assert by_id[ID_A] == by_id[ID_B]
    assert fused[0][0] in {ID_A, ID_B}


def test_reciprocal_rank_fusion_handles_empty_leg() -> None:
    fused = reciprocal_rank_fusion([[ID_A, ID_B], []], k=60)
    assert fused == [(ID_A, 1 / 61), (ID_B, 1 / 62)]


def test_reciprocal_rank_fusion_handles_both_empty() -> None:
    assert reciprocal_rank_fusion([[], []], k=60) == []
