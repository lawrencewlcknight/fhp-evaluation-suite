"""Vector equivalent of the frozen Exp16 information-state card features.

Only the exact v2 encoder is supported. No hand buckets, opponent cards, learned
approximations or cross-hand caches. Context is supplied by the native encoder
on one representative state, so betting-history semantics remain native.
"""
import numpy as np


def card_frames(hands, board):
    """Return N x 213 float32 inputs with [104:138] context left blank.

    Card IDs are OpenSpiel's full-deck rank-major IDs (2c=0 ... As=51).
    Suit signatures have exactly 26 bits: sorting integers is identical to
    lexicographically sorting the native hole-then-board binary signatures.
    """
    hands = np.asarray(hands)
    board = np.asarray(board, dtype=np.int64)
    if (hands.ndim != 2 or hands.shape[1] != 2 or not np.issubdtype(hands.dtype, np.integer)
            or board.ndim != 1 or len(board) not in (0, 3)
            or np.any(hands < 0) or np.any(hands >= 52)
            or np.any(board < 0) or np.any(board >= 52)
            or len(np.unique(board)) != len(board)
            or np.any(hands[:, 0] == hands[:, 1]) or np.isin(hands, board).any()):
        raise ValueError("Expected distinct full-deck private cards and zero/three disjoint board cards")
    n = len(hands)
    rows = np.arange(n)
    hole = np.zeros((n, 4, 13), dtype=np.float32)
    hole[rows[:, None], hands % 4, hands // 4] = 1
    public = np.zeros((4, 13), dtype=np.float32)
    public[board % 4, board // 4] = 1
    powers = 1 << np.arange(12, -1, -1, dtype=np.int64)
    signature = ((hole.astype(np.int64) @ powers) << 13) + (public.astype(np.int64) @ powers)
    order = np.argsort(-signature, axis=1, kind="stable")
    canonical_hole = hole[rows[:, None], order]
    canonical_board = public[order]
    hc = hole.sum(1).astype(np.int8)
    bc = public.sum(0).astype(np.int8)
    hs = canonical_hole.sum(2)
    bs = canonical_board.sum(2)
    x = np.zeros((n, 213), dtype=np.float32)
    x[:, :52] = canonical_hole.reshape(n, 52)
    x[:, 52:104] = canonical_board.reshape(n, 52)
    x[:, 138:151] = hc.astype(np.float32) / 2
    x[:, 151:164] = bc.astype(np.float32) / 3
    x[:, 164:168] = hs / 2
    x[:, 168:172] = bs / 3
    x[:, 172] = hc.max(1) == 2
    x[:, 173] = hs.max(1) == 2
    extra = x[:, 183:]
    ranks = np.sort(hands // 4, axis=1)
    low, high = ranks[:, 0], ranks[:, 1]
    extra[:, 5:8] = np.stack(((high + 2) / 14, (low + 2) / 14, (high - low) / 12), axis=1)
    if not len(board):
        return x

    counts = hc + bc
    present, board_present = counts > 0, bc > 0
    windows = np.asarray([*([*range(s, s + 5)] for s in range(9)), [12, 0, 1, 2, 3]])
    window_counts = present[:, windows].sum(2)
    straight_high = np.max(np.where(window_counts == 5, np.asarray([*range(4, 13), 3]), -1), axis=1)
    straight = straight_high >= 0
    suits = hs + bs
    flush = suits.max(1) == 5
    max_count = counts.max(1)
    pairs = (counts == 2).sum(1)
    category = np.select([straight & flush, max_count == 4, (max_count == 3) & (pairs == 1),
                          flush, straight, max_count == 3, pairs == 2, pairs == 1],
                         [8, 7, 6, 5, 4, 3, 2, 1], default=0)
    x[rows, 174 + category] = 1

    # Descending (multiplicity, rank), with absent ranks after all real ranks.
    tie_order = np.argsort(-(counts * 16 + np.arange(13)), axis=1, kind="stable")[:, :5]
    tie_counts = np.take_along_axis(counts, tie_order, axis=1)
    tie = np.where(tie_counts > 0, (tie_order + 2) / 14, 0.)
    tie[straight] = 0
    tie[straight, 0] = (straight_high[straight] + 2) / 14
    extra[:, :5] = tie
    board_ranks = np.sort(board // 4)[::-1]
    distinct = np.flatnonzero(bc)
    extra[:, 8:11] = (board_ranks + 2) / 14
    extra[:, 11 + 3 - len(distinct)] = 1
    extra[:, 14 + int(public.sum(1).max()) - 1] = 1
    pocket = low == high
    relation = np.select([high > distinct[-1], high < distinct[0], bc[high] > 0], [0, 1, 3], default=2)
    extra[rows[pocket], 17 + relation[pocket]] = 1
    extra[:, 21] = hc[:, distinct[-1]] > 0
    extra[:, 22] = hc[:, distinct[1]] > 0 if len(distinct) == 3 else 0
    extra[:, 23] = hc[:, distinct[0]] > 0
    extra[:, 24] = (ranks > distinct[-1]).sum(1) / 2
    extra[:, 25] = (bc[ranks] > 0).sum(1) / 2
    extra[:, 26] = suits.max(1).astype(np.float64) / 5
    extra[:, 27] = board_present[windows].sum(1).max() / 3
    extra[:, 28] = window_counts.max(1) / 5
    current, longest = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    # Fourteen rank positions, vectorised over all hands; ace may be low or high.
    for rank in (12, *range(13)):
        current = np.where(present[:, rank], current + 1, 0)
        longest = np.maximum(longest, current)
    extra[:, 29] = longest / 5
    return x
