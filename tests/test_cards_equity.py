from __future__ import annotations

import pytest

from fhp_evaluation.cards import card_id, canonical_starting_hand, five_card_rank
from fhp_evaluation.equity import published_hand_equity, published_hand_strength


def ids(*cards):
    return tuple(card_id(card) for card in cards)


def test_five_card_rank_orders_categories_and_wheel():
    royal = five_card_rank(ids("Ah", "Kh", "Qh", "Jh", "Th"))
    quads = five_card_rank(ids("As", "Ah", "Ad", "Ac", "2h"))
    full_house = five_card_rank(ids("Ks", "Kh", "Kd", "2c", "2d"))
    wheel = five_card_rank(ids("As", "2h", "3d", "4c", "5s"))
    six_high = five_card_rank(ids("2s", "3h", "4d", "5c", "6s"))
    assert royal > quads > full_house > six_high > wheel


def test_published_preflop_lookup_and_strength_scale():
    assert canonical_starting_hand(ids("As", "Ah")) == "AA"
    assert canonical_starting_hand(ids("As", "Kh")) == "AKo"
    assert canonical_starting_hand(ids("As", "Ks")) == "AKs"
    equity = published_hand_equity(ids("As", "Ah"), ())
    assert equity == pytest.approx(0.852037132245)
    assert published_hand_strength(ids("As", "Ah"), ()) == pytest.approx(
        equity * 1326.0 - 663.0
    )


def test_flop_equity_is_exact_for_unbeatable_royal_flush():
    equity = published_hand_equity(ids("Ah", "Kh"), ids("Qh", "Jh", "Th"))
    assert equity == 1.0

