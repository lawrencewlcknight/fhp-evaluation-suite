"""Deterministic board-class holdouts and independent reference deal streams."""

from collections import Counter, defaultdict
from itertools import combinations, permutations

import numpy as np

from . import PROTOCOL


def canonical_board(board):
    return min(tuple(sorted(4 * (c // 4) + p[c % 4] for c in board))
               for p in permutations(range(4)))


def texture(board):
    ranks = len({c // 4 for c in board})
    suits = len({c % 4 for c in board})
    return f"{ {3: 'unpaired', 2: 'paired', 1: 'trips'}[ranks]}_{suits}suit"


def build_library(seed=20261006, *, pilot=False, reference_deals=64):
    if reference_deals < 0:
        raise ValueError("reference_deals must be nonnegative")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for board in sorted({canonical_board(b) for b in combinations(range(52), 3)}):
        groups[texture(board)].append(board)
    # All six feasible textures; 96 boards, 64 development / 32 assessment.
    quotas = {"unpaired_1suit": (24, 16), "unpaired_2suit": (24, 16),
              "unpaired_3suit": (24, 16), "paired_2suit": (8, 5),
              "paired_3suit": (8, 5), "trips_3suit": (8, 6)}
    rows = []
    for kind, (count, development) in quotas.items():
        chosen = rng.choice(len(groups[kind]), size=count, replace=False)
        for i, index in enumerate(chosen):
            # Random label assignment prevents privileging low-index suits in
            # raw-input networks; holdout membership is still by suit class.
            labels = rng.permutation(4)
            board = sorted(4 * (c // 4) + int(labels[c % 4]) for c in groups[kind][index])
            rows.append(dict(board=board, texture=kind,
                             partition="development" if i < development else "assessment"))
    # Preserve the full holdout even when the pilot drops assessment rows.
    # Otherwise pilot reference deals could accidentally expose a locked board.
    heldout = {canonical_board(r["board"]) for r in rows if r["partition"] == "assessment"}
    if pilot:
        # Six textures plus two additional unpaired boards. No assessment access.
        selected = [next(r for r in rows if r["texture"] == kind) for kind in quotas]
        selected += [r for r in rows if r["texture"] == "unpaired_3suit"
                     and r not in selected][:2]
        rows = selected
    # Independent full-deck deals, not a uniform sample of suit classes. Exclude
    # locked board classes so the reference view cannot leak the assessment set.
    deals = []
    while len(deals) < reference_deals:
        cards = list(map(int, rng.choice(52, size=7, replace=False)))
        if canonical_board(cards[4:]) in heldout:
            continue
        deals.append(dict(cards=cards, action_seed=int(rng.integers(0, 2**63))))
    result = dict(protocol=PROTOCOL, seed=int(seed), pilot=bool(pilot), boards=rows,
                  assessment_board_classes=[list(b) for b in sorted(heldout)],
                  reference_deals=deals, reference_sampling="uniform deals conditional on non-assessment board class",
                  partition_counts=dict(Counter(r["partition"] for r in rows)))
    validate_library(result)
    return result


def validate_library(library):
    if library.get("protocol") != PROTOCOL or not library.get("boards"):
        raise ValueError("Unsupported or empty strategic library")
    seen = set()
    for row in library["boards"]:
        board = row["board"]
        if (len(board) != 3 or len(set(board)) != 3 or
                any(type(c) is not int or not 0 <= c < 52 for c in board)):
            raise ValueError("Invalid board")
        key = canonical_board(board)
        if key in seen or row["partition"] not in ("development", "assessment"):
            raise ValueError("Duplicate suit class or invalid partition")
        if row["texture"] != texture(board):
            raise ValueError("Wrong board texture")
        seen.add(key)
    heldout = {canonical_board(r["board"]) for r in library["boards"]
               if r["partition"] == "assessment"}
    for board in library.get("assessment_board_classes", []):
        if (len(board) != 3 or len(set(board)) != 3 or
                any(type(c) is not int or not 0 <= c < 52 for c in board)):
            raise ValueError("Invalid locked board class")
        heldout.add(canonical_board(board))
    if any(canonical_board(r["board"]) in heldout for r in library["boards"] if r["partition"] == "development"):
        raise ValueError("Development/assessment leakage")
    for deal in library.get("reference_deals", []):
        cards = deal["cards"]
        if (len(cards) != 7 or len(set(cards)) != 7 or
                any(type(c) is not int or not 0 <= c < 52 for c in cards)
                or canonical_board(cards[4:]) in heldout):
            raise ValueError("Invalid deal or assessment leakage")


def board_key(board):
    return "-".join(map(str, sorted(board)))


def board_schedule(library, partition, include_reference=False):
    validate_library(library)
    if partition not in ("development", "assessment"):
        raise ValueError("Select one partition explicitly")
    boards = {tuple(sorted(r["board"])) for r in library["boards"] if r["partition"] == partition}
    if include_reference:
        if partition != "development":
            raise ValueError("Reference sampling belongs to development, not locked assessment")
        boards.update(tuple(sorted(d["cards"][4:])) for d in library["reference_deals"])
    return sorted(boards)
