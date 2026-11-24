import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from worldflow.inference.weights import load_weights
from worldflow.models.network import WorldFlowNetwork
from worldflow.training.initialize import initialize_weights


class SmallNetwork(nn.Module):
    initialize_objectflow = WorldFlowNetwork.initialize_objectflow

    def __init__(self):
        super().__init__()
        self.hidden_size = 8
        self.objectflow_dim = 3
        self.vae2llm = nn.Linear(12, 8)
        self.objectflow2llm = nn.Linear(3, 8)
        self.llm2objectflow = nn.Linear(8, 3)
        self.objectflow_modality_embed = nn.Parameter(torch.zeros(8))
        self.objectflow_point_pe = nn.Sequential(nn.Linear(3, 8))
        self.objectflow_point_pe.abs_scale = 2.0
        self.objectflow_point_pe.rel_scale = 0.3
        for name in ("scale", "point_pe_abs_scale", "point_pe_rel_scale"):
            self.register_buffer(f"objectflow_{name}_buf", torch.tensor(0.0))


class BaseImportTests(unittest.TestCase):
    def test_stage1_loads_pretrained_weights_and_initializes_only_new_branch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = SmallNetwork()
            backbone = {"net_ema.vae2llm." + k: v for k, v in model.vae2llm.state_dict().items()}
            dcp.save(backbone, checkpoint_id=root / "base", no_dist=True)
            with patch("worldflow.training.initialize.WorldFlowNetwork", SmallNetwork):
                initialize_weights(root / "base", root / "model", seed=42)
            keys = json.loads((root / "model/model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(set(keys), set(model.state_dict()))
            loaded = SmallNetwork()
            load_weights(loaded, root / "model")
            torch.testing.assert_close(loaded.vae2llm.weight, model.vae2llm.weight, rtol=0, atol=0)
            self.assertEqual(loaded.llm2objectflow.weight.count_nonzero(), 0)
            self.assertGreater(loaded.objectflow2llm.weight.norm().item(), 0)
            self.assertAlmostEqual(loaded.objectflow_scale_buf.item(), 0.0155919)

    def test_refuses_incomplete_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dcp.save({"net_ema.vae2llm.weight": torch.zeros(8, 12)}, checkpoint_id=root / "base", no_dist=True)
            with patch("worldflow.training.initialize.WorldFlowNetwork", SmallNetwork):
                with self.assertRaisesRegex(ValueError, "missing 1 backbone"):
                    initialize_weights(root / "base", root / "model", seed=42)
