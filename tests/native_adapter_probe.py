"""Subprocess-only fixtures: native repos have overlapping package names."""

import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch

from fhp_evaluation.best_response.adapters import load_target
from fhp_evaluation.best_response.flop import FlopBestResponse, make_state
from fhp_evaluation.game import load_fhp_game, FHP_GAME_PARAMETERS


family, repo_root, output = sys.argv[1:]
sys.path.insert(0, repo_root)
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(421)
np.random.seed(421)
game = load_fhp_game()
paths = []

if family == "ucv":
    from fhp_escher.features import FHPFeatureEncoder, StructuredFHPMLP, POLICY_LAYOUT
    from fhp_escher.card_policy import ResidualCardPolicy
    from vr_deep_cfr.solver import MLP
    encoder = FHPFeatureEncoder()
    models = [("raw", MLP(190, [8], 3), None),
              ("encoded", MLP(183, [8], 3), encoder),
              ("structured", StructuredFHPMLP(POLICY_LAYOUT, [8], 3, branch_width=4), encoder)]
    models += [(kind, ResidualCardPolicy(kind), encoder) for kind in ("dense", "deepsets", "attention")]
    for name, model, feature in models:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * .03)
        payload = dict(type="fhp_ucv_escher_policy_checkpoint", version=1,
                       game=dict(parameters=dict(FHP_GAME_PARAMETERS)),
                       input_size=190 if feature is None else 183, num_actions=3,
                       policy_network_layers=[8], feature_encoder=None if feature is None else feature.metadata(),
                       policy_model=model.checkpoint_metadata() if hasattr(model, "checkpoint_metadata") else {"type": "mlp_v1"},
                       policy_state_dict=model.state_dict())
        path = output / f"{name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
        paths.append(path)
elif family == "vr_deep":
    from fhp_vr_deep.features import FHPFeatureEncoder
    from vr_deep_cfr.solver import MLP
    for feature in (None, FHPFeatureEncoder()):
        width = 190 if feature is None else 183
        path = output / f"policy_{width}.pt"
        payload = dict(type="vr_deep_cfr_policy_snapshot", version=2 if feature is None else 3,
                       game=dict(parameters=dict(FHP_GAME_PARAMETERS)), input_size=width, num_actions=3,
                       policy_network_layers=[8], policy_state_dict=MLP(width, [8], 3).state_dict())
        if feature:
            payload["feature_encoder"] = feature.metadata()
        torch.save(payload, path)
        paths.append(path)
else:
    from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive
    from deep_cfr_poker.networks import build_network
    from deep_cfr_poker.fhp_features import FHPFeatureEncoder
    for feature in (None, FHPFeatureEncoder()):
        width = 190 if feature is None else 183
        models = [build_network("mlp", width, (8,), 3) for _ in range(2)]
        metadata = {} if feature is None else dict(feature_encoder=feature.metadata())
        solver = SimpleNamespace(_num_players=2, _num_actions=3, _embedding_size=width,
                                 _game=game, _advantage_network_type="mlp", _advantage_network_layers=(8,),
                                 _advantage_networks=models, archive=SimpleNamespace(metadata=metadata))
        archive = DiskSDCFRArchive(solver, output / str(width), chunk_iterations=2)
        for iteration in range(1, 4):
            for player, model in enumerate(models):
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(torch.randn_like(parameter) * .15)
                    list(model.parameters())[-1].add_(torch.tensor([1., 3., 2.]) * .1)
                archive.capture_from_solver(solver, player, iteration)
        paths.append(archive.checkpoint(output / str(width) / "policy.json"))

states = []
deck = tuple(range(52))
for pre, board, post in [((), (), ()), ((2,), (), ()), ((2, 2, 2), (), ()),
                         ((1, 1), (16, 20, 32), ()), ((2, 1), (16, 20, 32), (2,))]:
    for player in (0, 1):
        for hand in ((0, 4), (48, 49), (6, 7)):
            states.append(make_state(game, deck, hand, player, board=board, preflop=pre, flop=post))
records = []
for path in paths:
    target = load_target(game, path, family=family, repo_root=repo_root,
                         batch_size=3, model_batch_size=2)
    record = target.check_scalar_parity(states)
    record.update(checkpoint=path.name, family=family)
    records.append(record)
print(json.dumps(records))
