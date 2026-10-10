"""Algorithm, leakage, gate, native-reload and bounded FHP integration checks."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from fhp_evaluation.adversarial.config import ARMS, configuration, stream
from fhp_evaluation.adversarial.learning import (
    Episode, NetworkPolicy, TransitionReplay, attacker_model, collect,
    double_dqn_targets, masked_logs, model_hash, repair_loss, train_attacker,
)
from fhp_evaluation.adversarial.source import Replay, safe_path
from fhp_evaluation.adversarial.workflow import (
    Events, evaluate, pairs, qualification, require_gate, save_policy, train,
)
from fhp_evaluation.best_response.adapters import _native_root, load_target
from fhp_evaluation.game import FHP_GAME_PARAMETERS


@pytest.fixture(scope="module")
def native_root():
    root = Path(os.environ.get("FHP_UCV_REPO", Path(__file__).resolve().parents[2]
                               / "fhp_ucv_escher/fhp-ucv-escher-experiments"))
    if not root.is_dir():
        pytest.skip("Set FHP_UCV_REPO for native UCV integration tests")
    _native_root(root, ("fhp_escher", "vr_deep_cfr"))
    return root


@pytest.fixture
def target(game, native_root, tmp_path):
    from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
    from fhp_escher.features import StructuredFHPMLP
    encoder = FHPHandBoardFeatureEncoder()
    np.random.seed(123)
    model = StructuredFHPMLP(encoder.policy_layout, [16, 16], 3, branch_width=8)
    payload = dict(type="fhp_ucv_escher_policy_checkpoint", version=1, game=dict(parameters=dict(FHP_GAME_PARAMETERS)),
                   input_size=213, policy_network_layers=[16, 16], num_actions=3,
                   feature_encoder=encoder.metadata(), policy_model=model.checkpoint_metadata(),
                   policy_state_dict=model.state_dict(), experiment_id=10, seed=0,
                   checkpoint_target_seconds=86400, algorithm_id="fixture", outer_iteration=1)
    path = tmp_path / "source.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return load_target(game, path, family="ucv", repo_root=native_root)


@pytest.fixture
def replay(game, target):
    actor = NetworkPolicy(target.native.model, target.native.feature_encoder)
    episodes = collect(game, actor, [actor] * 32, [i % 2 for i in range(32)], np.random.default_rng(33))
    rows = [r for e in episodes for r in e.rows]
    x = np.stack([r[0] for r in rows])
    mask = np.stack([r[1] for r in rows])
    return Replay(x, mask / mask.sum(1)[:, None], mask, np.ones(len(x)), 1)


def test_seed_namespaces_and_budgets():
    assert stream(0, "train", 1) == stream(0, "train", 1)
    assert len({stream(s, p, a) for s in range(3) for p in ("train", "select", "test") for a in range(4)}) == 36
    cfg = configuration()
    assert cfg.rounds == 3 and cfg.attackers_per_seat == 2
    assert cfg.attack_seconds == 900 and not cfg.smoke
    assert cfg.fingerprint() != configuration(True).fingerprint()


def test_double_dqn_selects_online_but_evaluates_target_and_masks():
    online = nn.Linear(1, 3)
    target = nn.Linear(1, 3)
    with torch.no_grad():
        online.weight.zero_(); target.weight.zero_()
        online.bias.copy_(torch.tensor([999., 5., 2.]))
        target.bias.copy_(torch.tensor([999., 3., 8.]))
    values = double_dqn_targets(online, target, torch.zeros(2, 1),
                                torch.tensor([[False, True, True], [False, False, False]]),
                                torch.tensor([1., -2.]), torch.tensor([False, True]))
    assert torch.equal(values, torch.tensor([4., -2.]))


def test_replay_next_own_decision_and_terminal_returns():
    ring = TransitionReplay(2, 2)
    mask = np.array([True, True, False])
    ring.append([Episode([(np.array([1, 2]), mask, 0), (np.array([3, 4]), mask, 1)], 2.5)])
    assert ring.size == 2 and ring.cursor == 0
    assert ring.reward.tolist() == [0., 2.5]
    assert np.array_equal(ring.next_x[0], [3, 4])
    assert not ring.next_mask[1].any()
    ring.append([Episode([(np.array([5, 6]), mask, 1)], -1)])
    assert ring.size == 2 and ring.cursor == 1 and ring.reward[0] == -1


def test_grouped_replay_objective_and_cache(tmp_path):
    x = np.zeros((3, 213), np.float32)
    x[2, 0] = 1
    masks = np.array([[1, 1, 0]] * 3, dtype=bool)
    y = np.array([[.2, .8, 0], [.8, .2, 0], [.3, .7, 0]], np.float32)
    r = Replay(x, y, masks, np.array([1, 2, 2]), 2)
    weights = np.array([1., 4., 4.])
    assert len(r.x) == 2 and r.scale == 3
    assert np.allclose(r.y[0], (y[0] + 4 * y[1]) / 5)
    logits = torch.tensor([[.1, .7, 0], [.4, .2, 0]])
    logs = torch.log_softmax(logits[:, :2], -1).numpy()
    grouped = -(np.diff(np.r_[0, r.cdf]) * (r.y[:, :2] * logs).sum(1)).sum() * r.scale
    rowwise = -(weights * (y[:, :2] * logs[[0, 0, 1]]).sum(1)).mean()
    assert grouped == pytest.approx(rowwise, rel=1e-6)
    digest = r.save(tmp_path / "replay", "abc")
    restored = Replay.load(tmp_path / "replay", digest, "abc")
    assert np.array_equal(restored.y, r.y)
    with pytest.raises(ValueError, match="provenance"):
        Replay.load(tmp_path / "replay", digest, "wrong")
    (tmp_path / "replay/y.npy").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="SHA"):
        Replay.load(tmp_path / "replay", digest, "abc")


def test_pg_sign_and_trajectory_normalisation():
    model = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    row = (np.array([1, 0], np.float32), np.array([1, 1, 0], bool), 1)
    episodes = [Episode([row, row], 2), Episode([], -1)]
    loss, kl = repair_loss(model, deepcopy(model), episodes, 0, preserve=True)
    assert loss.item() == pytest.approx(2 * np.log(2))
    assert kl.item() == 0
    loss.backward()
    assert model.weight.grad[1, 0] < 0  # Gradient descent increases rewarding action.
    assert model.weight.grad[2, 0] == 0  # Illegal action receives no gradient.


def test_hidden_opponent_cards_do_not_enter_attacker_features(game, target):
    # Deal order in universal_poker is P0,P0,P1,P1. Verify information strings
    # as well as encodings instead of assuming a different dealing convention.
    states = []
    for cards in ((0, 4, 8, 12), (0, 4, 16, 20)):
        s = game.new_initial_state()
        for card in cards:
            s.apply_action(card)
        states.append(s)
    encoder = target.native.feature_encoder
    assert states[0].information_state_string(0) == states[1].information_state_string(0)
    assert np.array_equal(encoder.information_state(states[0], 0), encoder.information_state(states[1], 0))


def test_native_policy_reload_and_batched_parity(game, target, native_root, tmp_path):
    cfg = configuration(True)
    network = NetworkPolicy(deepcopy(target.native.model), target.native.feature_encoder)
    path = tmp_path / "post.pkl"
    save_policy(target, network, path, "preserved", cfg, target.metadata["checkpoint_sha256"])
    loaded = load_target(game, path, family="ucv", repo_root=native_root)
    s = game.new_initial_state()
    for c in (0, 4, 8, 12):
        s.apply_action(c)
    assert loaded.action_probabilities(s) == pytest.approx(network.action_probabilities(s), abs=1e-7)
    assert loaded.native.checkpoint["post_training"]["ucv_solver_resumable"] is False


def test_gate_fails_closed():
    cfg = configuration()
    gate = dict(phase="qualify", seed=0, config_hash=cfg.fingerprint(),
                source_policy_sha256="abc", bundle_sha256="def", gate_passed=False)
    with pytest.raises(ValueError, match="strength"):
        require_gate(gate, cfg, 0, "abc", "def")
    gate["gate_passed"] = True
    require_gate(gate, cfg, 0, "abc", "def")
    with pytest.raises(ValueError, match="provenance"):
        require_gate(gate, cfg, 1, "abc", "def")
    with pytest.raises(ValueError):
        require_gate(gate, configuration(True), 0, "abc", "def")


def test_unsafe_paths(tmp_path):
    for value in ("../escape", "/tmp/escape", "."):
        with pytest.raises(ValueError):
            safe_path(tmp_path, value)


def test_aggregate_selects_on_validation_not_final_payoff(tmp_path):
    from fhp_evaluation.adversarial.aggregate import aggregate
    cfg = configuration(True)
    for i, arm in enumerate(ARMS):
        directory = tmp_path / "input" / f"task_{i}"
        directory.mkdir(parents=True)
        record = dict(config_hash=cfg.fingerprint(), seed=0, arm=arm, phase="evaluate",
                      status="smoke_complete", bundle_sha256="same",
                      challenger_matrix={a: dict(selection_mean_mbb=10 if arm == "ordinary" else 0,
                                                pair_payoffs_mbb=[2 if arm == "ordinary" else 100] * 4)
                                         for a in ARMS},
                      h2h_vs_unchanged=dict(pair_payoffs_mbb=[0.] * 4), rule_panel={},
                      specialised_response=dict(pair_payoffs_mbb=dict(lbr=[1.] * 2, full_flop=[1.] * 2)))
        (directory / "result.json").write_text(json.dumps(record))
    summary = aggregate(tmp_path / "input", tmp_path / "out", smoke=True)
    assert summary["selected_challengers"]["preserved"] == ["ordinary"]
    assert summary["arms"]["preserved"]["learned_response"]["mean_mbb"] == 2
    assert summary["exact_exploitability"] is False


def test_historical_monolithic_source(game, native_root, tmp_path, monkeypatch):
    from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
    from fhp_escher.features import StructuredFHPMLP
    from fhp_evaluation.adversarial import source
    from fhp_evaluation.loaders import sha256_file
    encoder = FHPHandBoardFeatureEncoder()
    model = StructuredFHPMLP(encoder.policy_layout, [192, 192], 3, branch_width=64)
    policy = dict(type="fhp_ucv_escher_policy_checkpoint", version=1, game=dict(parameters=dict(FHP_GAME_PARAMETERS)),
                  input_size=213, policy_network_layers=[192, 192], num_actions=3,
                  feature_encoder=encoder.metadata(), policy_model=model.checkpoint_metadata(),
                  policy_state_dict=model.state_dict(), experiment_id=10, seed=0,
                  checkpoint_target_seconds=86400, algorithm_id="fixture", outer_iteration=1)
    with (tmp_path / "policy.pkl").open("wb") as handle:
        pickle.dump(policy, handle)
    payload = dict(schema_version=1, seed=0, checkpoint_id="time_24h",
                   config=dict(feature_encoder_id="fhp_lossless_hand_board_v2", gamma=2),
                   solver=dict(training_elapsed_seconds=86400, num_iteration=1),
                   average_policy_trainer=dict(model=model.state_dict(), buffer=dict(
                       size=3, infostate=np.zeros((3, 213), np.float32),
                       q_value=np.array([[.5, .5, 0]] * 3, np.float32),
                       q_value_mask=np.array([[1, 1, 0]] * 3, np.float32),
                       iteration=np.ones(3, np.float32))))
    torch.save(payload, tmp_path / "state.pt")
    row = dict(checkpoint_id="time_24h", path="policy.pkl", sha256=sha256_file(tmp_path / "policy.pkl"),
               training_state_path="state.pt", training_state_sha256=sha256_file(tmp_path / "state.pt"))
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps([row]))
    digest = sha256_file(tmp_path / "checkpoint_manifest.json")
    monkeypatch.setattr(source, "MANIFEST_HASHES", (digest, "", ""))
    (tmp_path / "source.json").write_text(json.dumps(dict(seed=0, source_run=source.SOURCE_RUN,
         checkpoint_manifest_sha256=digest, checkpoint=row)))
    target = load_target(game, tmp_path / "policy.pkl", family="ucv", repo_root=native_root)
    replay, _ = source.load_source(tmp_path, 0, target)
    assert replay.rows == 3 and len(replay.x) == 1
    (tmp_path / "state.pt").unlink()
    assert source.load_source(tmp_path, 0, target, need_replay=False)[0] is None


def test_real_fhp_attacker_and_repair_smoke(game, target, replay, native_root, tmp_path):
    torch.set_num_threads(2)
    cfg = configuration(True)
    reference = NetworkPolicy(target.native.model, target.native.feature_encoder)
    before = model_hash(reference.model)
    qualification_dir = tmp_path / "qualification"
    qualification_dir.mkdir()
    result = qualification(game, reference, cfg, 0, qualification_dir, Events(qualification_dir / "events.jsonl"))
    assert result["smoke_execution_only"] and not result["gate_passed"]
    assert all(t["updates"] > 0 for t in result["timings"])
    training = tmp_path / "training"
    training.mkdir()
    record = train(game, target, replay, cfg, 0, training, Events(training / "events.jsonl"))
    assert set(record["policy_hashes"]) == set(ARMS)
    assert model_hash(reference.model) == before
    assert record["arms"]["ordinary"]["seconds"] >= record["arms"]["preserved"]["seconds"]
    policies = {a: load_target(game, training / f"{a}.pkl", family="ucv", repo_root=native_root) for a in ARMS}
    assert model_hash(policies["unchanged"].native.model) == before
    assert model_hash(policies["preserved"].native.model) != before
    assert np.allclose(pairs(game, reference, reference, 4, 1234), 0)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    evaluated = evaluate(game, policies, cfg, 0, "preserved", evaluation, Events(evaluation / "events.jsonl"))
    assert evaluated["exact_exploitability"] is False
    assert set(evaluated["challenger_matrix"]) == set(ARMS)
    assert len(evaluated["specialised_response"]["pair_payoffs_mbb"]["lbr"]) == 2
