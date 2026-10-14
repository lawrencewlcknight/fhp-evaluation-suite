from types import SimpleNamespace

import numpy as np
import pytest

from fhp_evaluation.strategic_audit import checks
from fhp_evaluation.strategic_audit.tables import BoardContext, build_tables
from fhp_evaluation.best_response.adapters import PolicyAdapter
from fhp_evaluation.best_response.validation import FixedPolicy
from fhp_evaluation.game import load_fhp_game


def test_fallback_rechecks_and_returns_precise_export_adapter(monkeypatch):
    precise = SimpleNamespace(backend="sd_cfr")
    adapter = SimpleNamespace(backend="sd_cfr", sd_float64_copy=lambda: precise)
    calls = []
    def check(context, value, **kwargs):
        calls.append((value, kwargs))
        if value is adapter:
            raise ValueError("Native/batched policy mismatch: 1.88e-5 > 3e-6")
        return dict(native_scalar_parity=dict(passed=True, tolerance=3e-6))
    monkeypatch.setattr(checks, "check_adapter", check)
    chosen, result = checks.checked_adapter(None, adapter, require_suit_invariance=True, float64_fallback=True)
    assert chosen is precise and len(calls) == 2
    assert all(k["require_suit_invariance"] for _, k in calls)
    assert result["numerical_fallback"]["tolerance_unchanged"]
    assert result["native_scalar_parity"]["tolerance"] == 3e-6


@pytest.mark.parametrize("message", ["Hidden-card or card-order invariance failure",
                                    "Required suit invariance failed", "Non-finite archived network prediction"])
def test_non_parity_failures_are_never_waived(monkeypatch, message):
    adapter = SimpleNamespace(backend="sd_cfr", sd_float64_copy=lambda: pytest.fail("unexpected fallback"))
    def fail(*args, **kwargs):
        raise ValueError(message)
    monkeypatch.setattr(checks, "check_adapter", fail)
    with pytest.raises(ValueError, match=message):
        checks.checked_adapter(None, adapter, float64_fallback=True)


def test_precise_failure_still_fails_and_disabled_fallback_is_unchanged(monkeypatch):
    adapter = SimpleNamespace(backend="sd_cfr", sd_float64_copy=lambda: adapter)
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("Native/batched policy mismatch: still fails")
    monkeypatch.setattr(checks, "check_adapter", fail)
    with pytest.raises(ValueError):
        checks.checked_adapter(None, adapter, float64_fallback=True)
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(ValueError):
        checks.checked_adapter(None, adapter)
    assert len(calls) == 1


def test_hand_major_complete_table_is_identical_and_batches_bounded():
    game = load_fhp_game()
    context = BoardContext(game, [23, 24, 42])
    # A deterministic state-dependent policy, not just uniform probabilities.
    class Policy:
        def action_probabilities(self, state, player):
            legal = state.legal_actions(player)
            values = np.array([1 + (sum(state.history()) + a) % 7 for a in legal], dtype=float)
            return dict(zip(legal, values / values.sum()))
    baseline = build_tables(context, PolicyAdapter(Policy(), batch_size=37))
    adapter = PolicyAdapter(Policy(), batch_size=37)
    expected_method = adapter.batch_probabilities
    def bounded(states, player):
        assert len(states) <= 37
        return expected_method(states, player)
    adapter.batch_probabilities = bounded
    actual = build_tables(context, adapter, query_order="hand_major")
    assert set(actual) == set(baseline)
    for name in baseline:
        np.testing.assert_array_equal(actual[name], baseline[name])
    with pytest.raises(ValueError, match="query order"):
        build_tables(context, adapter, query_order="unknown")


def test_exported_probabilities_are_checked_not_just_probe_queries():
    from fhp_evaluation.strategic_audit.tables import key
    context = BoardContext(load_fhp_game(), [23, 24, 42])
    adapter = PolicyAdapter(FixedPolicy(context.game, "uniform"))
    tables = build_tables(context, adapter, query_order="hand_major")
    assert checks.check_exported_parity(context, adapter, tables)["passed"]
    pre = context.entries[0]
    node = next(n for n in context.trees[pre] if n.player >= 0)
    row = tables[key(pre, node.path)][0]
    row[:] = [1., 0., 0.]
    with pytest.raises(ValueError, match="Export/reference"):
        checks.check_exported_parity(context, adapter, tables)


@pytest.mark.parametrize("precise_passes", [True, False])
def test_worker_rebuilds_table_before_marking_success(monkeypatch, tmp_path, precise_passes):
    from fhp_evaluation.strategic_audit import worker
    precise = SimpleNamespace(metadata=dict(inference_precision="float64"), stats={}, native=None)
    original = SimpleNamespace(metadata=dict(inference_precision="float32"), stats={}, native=None,
                               sd_float64_copy=lambda: precise)
    monkeypatch.setattr(worker, "policy_identity", lambda item: "policy")
    monkeypatch.setattr(worker, "load_target", lambda *args, **kwargs: original)
    monkeypatch.setattr(worker, "completed", lambda *args: False)
    checked, built, marked = [], [], []
    def check(context, adapter, **kwargs):
        checked.append(adapter)
        return adapter, {}
    def build(context, adapter, **kwargs):
        built.append(adapter)
        return {"fixture": np.ones((1, 3)) / 3}
    def parity(context, adapter, tables):
        if adapter is original or not precise_passes:
            raise ValueError("Export/reference policy mismatch: injected")
        return dict(passed=True)
    monkeypatch.setattr(worker, "checked_adapter", check)
    monkeypatch.setattr(worker, "build_tables", build)
    monkeypatch.setattr(worker, "check_exported_parity", parity)
    monkeypatch.setattr(worker, "mark", lambda *args: marked.append(args))
    spec = dict(policy=dict(family="sd_cfr", checkpoint="unused", sd_float64_fallback=True,
                            sd_query_order="hand_major"), expected_policy_identity="policy",
                boards=[[23, 24, 42]], cache=str(tmp_path), identity="audit")
    if precise_passes:
        assert worker.export(spec)["status"] == "succeeded"
        assert len(marked) == 1
    else:
        with pytest.raises(ValueError, match="Export/reference"):
            worker.export(spec)
        assert not marked
        assert not list(tmp_path.rglob("tables.npz"))
    assert checked == built == [original, precise]
