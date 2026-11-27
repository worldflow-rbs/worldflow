"""Read Python 3.13 pathlib metadata on older supported runtimes."""

import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from worldflow.dcp import dcp_reader


class DCPCompatibilityTests(unittest.TestCase):
    def test_python313_path_on_trusted_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            expected = torch.tensor([0.25, 0.75])
            dcp.save({"net_ema.weight": expected}, checkpoint_id=path, no_dist=True)
            file = path / ".metadata"
            metadata = dcp_reader(path).read_metadata()
            metadata.storage_meta.checkpoint_id = path
            # Protocol 2 names globals explicitly without binary string length
            # prefixes. Substitute just pathlib's module as written by 3.13.
            data = pickle.dumps(metadata, protocol=2)
            public = b"cpathlib\nPosixPath\n"
            local = b"cpathlib._local\nPosixPath\n"
            self.assertTrue(public in data or local in data)
            expected_metadata = data.replace(public, local)
            file.write_bytes(expected_metadata)

            reader = dcp_reader(path)
            restored = reader.read_metadata()
            self.assertEqual(restored.storage_meta.checkpoint_id, path)
            self.assertEqual(restored.storage_meta.load_id, reader.load_id)
            state = {"net_ema.weight": torch.zeros_like(expected)}
            dcp.load(state, storage_reader=reader, no_dist=True)
            torch.testing.assert_close(state["net_ema.weight"], expected, rtol=0, atol=0)
            self.assertEqual(file.read_bytes(), expected_metadata)
