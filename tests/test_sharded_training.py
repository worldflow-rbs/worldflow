"""Run with torchrun --standalone --nproc-per-node=2 -m unittest discover -s tests -p test_sharded_training.py."""

import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard

from worldflow.training.checkpoint import PowerEMA, restore_checkpoint, save_checkpoint
from worldflow.training.distributed import initialize_distributed


@unittest.skipUnless(int(os.environ.get("WORLD_SIZE", 1)) >= 2, "run with torchrun and two CUDA GPUs")
class ShardedTrainingTests(unittest.TestCase):
    def test_bf16_adam_and_fp32_ema_survive_resume(self):
        device, mesh = initialize_distributed()
        rank = dist.get_rank()
        self.addCleanup(dist.destroy_process_group)

        def setup():
            torch.manual_seed(38)
            model = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3)).to(device=device, dtype=torch.bfloat16)
            fully_shard(model[0], mesh=mesh)
            fully_shard(model, mesh=mesh)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=True)
            return model, optimizer, PowerEMA(model)

        def advance(model, optimizer, ema, step):
            data = torch.full((4, 5), 0.5 * (rank + 1), device=device, dtype=torch.bfloat16)
            model(data).float().square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(step)
            for parameter, state in optimizer.state.items():
                self.assertEqual(parameter.dtype, torch.bfloat16)
                self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
                self.assertEqual(state["step"].dtype, torch.float32)

        with ExitStack() as stack:
            shared = [stack.enter_context(tempfile.TemporaryDirectory()) if rank == 0 else None]
            dist.broadcast_object_list(shared, src=0)
            model, opt, ema = setup()
            advance(model, opt, ema, 0)
            for name, value in model.state_dict().items():
                self.assertEqual(ema.shadow[name].dtype, torch.float32)
                torch.testing.assert_close(value.full_tensor().float(), ema.shadow[name].full_tensor(), rtol=0, atol=0)
            saved = save_checkpoint(Path(shared[0]), model, ema, opt, 1, {"stage": 2})
            advance(model, opt, ema, 1)
            expected = {name: value.full_tensor() for name, value in ema.model_state().items()}

            model, opt, ema = setup()
            restore_checkpoint(saved, model, ema, opt, {"stage": 2})
            advance(model, opt, ema, 1)
            for name, value in ema.model_state().items():
                torch.testing.assert_close(value.full_tensor(), expected[name], rtol=0, atol=0, msg=name)
            dist.barrier()
