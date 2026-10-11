"""Fixed-work legacy/vectorised differential, separate from timed gameplay.

No playing-strength selection or budget tuning. Run only in feasibility mode;
each source seed contributes three public positions under the production pool.
"""
import numpy as np

from ..best_response.flop import validate_game
from ..strategic_audit.tables import preflop_nodes
from .solver import solve


def compare_preparation(game, adapter, seed, index, spec):
    rng = np.random.default_rng(spec['preflight_seed'] + 100000 + seed * 1000 + index)
    board = tuple(sorted(map(int, rng.choice(validate_game(game), 3, replace=False))))
    _, entries = preflop_nodes(game)
    pre, hero = sorted(entries)[index % len(entries)], index % 2
    results, strategies, problems = {}, {}, {}
    modes = ('legacy', 'vectorized') if (seed + index) % 2 else ('vectorized', 'legacy')
    encoder = getattr(adapter.native, 'feature_encoder', None)
    original_cache = getattr(encoder, 'cache_entries', None)
    try:
        if original_cache is not None:
            encoder.cache_entries = 0
        for mode in modes:
            # Neither implementation may borrow an already-encoded native frame.
            if encoder is not None:
                for name in ('_information_cache', '_full_state_cache'):
                    cache = getattr(encoder, name, None)
                    if cache is not None:
                        cache.clear()
            problems[mode], strategies[mode], results[mode] = solve(
                game, adapter, board, pre, hero, seconds=None, iterations=16,
                query_implementation=mode)
    finally:
        if original_cache is not None:
            encoder.cache_entries = original_cache
    old, new = problems['legacy'], problems['vectorized']
    # Same batch shapes/model/numerics, so this optimisation must be bitwise
    # identical at fixed work on one runtime (not necessarily across hardware).
    np.testing.assert_array_equal(old.reach, new.reach)
    for i in old.blueprint:
        np.testing.assert_array_equal(old.blueprint[i], new.blueprint[i])
    for key in ('iterations', 'checks', 'accepted', 'selected_iteration', 'rejected_checks'):
        if results['legacy'][key] != results['vectorized'][key]:
            raise AssertionError(f'Fixed-work solver mismatch: {key}')
    if strategies['legacy'] is not None:
        for i in strategies['legacy']:
            np.testing.assert_array_equal(strategies['legacy'][i], strategies['vectorized'][i])
    return dict(seed=seed, index=index, board=board, pre=pre, hero=hero, passed=True,
                fixed_iterations=16, execution_order=modes, results=results,
                preparation_speedup=results['legacy']['preparation_seconds'] /
                                    results['vectorized']['preparation_seconds'])
