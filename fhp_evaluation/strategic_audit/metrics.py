"""Blocker-aware rank moments and interpretable, overlapping hand tags."""

import numpy as np

from ..cards import five_card_rank


def win_tie_loss(space, weights):
    counts = np.bincount(space.codes, weights=weights, minlength=space.rank_count)
    prefix = np.cumsum(counts)
    wins = (prefix - counts)[space.codes]
    losses = (weights.sum() - prefix)[space.codes]
    card = np.bincount(space.card_rank_indices, weights=np.repeat(weights, 2),
                       minlength=space.deck_size * space.rank_count).reshape(space.deck_size, -1)
    cp = np.cumsum(card, axis=1)
    wins -= (cp - card)[space.hands, space.codes[:, None]].sum(axis=1)
    losses -= (cp[:, -1:] - cp)[space.hands, space.codes[:, None]].sum(axis=1)
    ties = space.mass(weights) - wins - losses
    # Roundoff from blocker subtraction can make exact zeros slightly negative.
    return tuple(np.maximum(a, 0.) for a in (wins, ties, losses))


def hand_tags(context):
    if hasattr(context, "_hand_tags"):
        return context._hand_tags
    suits = int(context.game.get_parameters()["numSuits"])
    board = [int(c) // suits for c in context.space.board]
    ranks = sorted(set(board), reverse=True)
    result = []
    for hand in context.space.hands:
        own = [int(c) // suits for c in hand]
        standard = [(int(c) // suits) * 4 + int(c) % suits for c in (*hand, *context.space.board)]
        rank = five_card_rank(standard)
        tags = ["category_" + str(rank[0]), "made_rank_" + str(rank[1])]
        if rank[0] == 1:
            tags.append("pair_kicker_" + str(rank[2]))
        if own[0] == own[1]:
            relation = "overpair" if own[0] > max(board) else (
                "underpair" if own[0] < min(board) else "pocket_pair_between_board_ranks")
            tags.append(relation)
        if len(ranks) == 3:
            for i, name in enumerate(("top_pair", "middle_pair", "bottom_pair")):
                if ranks[i] in own:
                    tags.append(name)
        result.append(tags)
    context._hand_tags = result
    return result
