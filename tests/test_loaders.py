from __future__ import annotations

import torch

from fhp_evaluation.loaders import LoadedCheckpointPolicy

from conftest import deal_preflop


def test_loader_supports_vr_and_deep_cfr_state_dict_layouts(game, tmp_path):
    layouts = (
        {
            "layers.0.weight": torch.zeros((4, 190)),
            "layers.0.bias": torch.zeros(4),
            "layers.1.weight": torch.zeros((3, 4)),
            "layers.1.bias": torch.tensor([0.0, 1.0, 2.0]),
        },
        {
            "model.0._weight": torch.zeros((4, 190)),
            "model.0._bias": torch.zeros(4),
            "model.1._weight": torch.zeros((3, 4)),
            "model.1._bias": torch.tensor([0.0, 1.0, 2.0]),
        },
    )
    state = deal_preflop(game)
    outputs = []
    for index, state_dict in enumerate(layouts):
        path = tmp_path / f"policy_{index}.pt"
        torch.save(
            {"game": "FHP", "policy_state_dict": state_dict, "version": 1},
            path,
        )
        outputs.append(LoadedCheckpointPolicy(game, path).action_probabilities(state, 0))
    assert outputs[0] == outputs[1]
    assert outputs[0][2] > outputs[0][1] > outputs[0][0]

