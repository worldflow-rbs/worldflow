"""Regression tests for canonical geometry, native admission and portable materialization."""

import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from worldflow.data import TrainingDataset, collate_training_samples
from worldflow.data.adapters import load_native
from worldflow.data.catalog import _compatible_mask, native_records
from worldflow.data.common import canonical_geometry
from worldflow.data.molmo import interval_to_49
from worldflow.data.prepare import prepare
from worldflow.data.realworld import load_any4d


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class DataTests(unittest.TestCase):
    def test_native_agibot_training_window_is_shared_by_rgb_and_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "agibot/any4d/demo/seg001"
            folder.mkdir(parents=True)
            index = base / "agibot/_meta/train_temporal_cap8.jsonl"
            index.parent.mkdir()
            length = 493
            dump(folder / "meta.json", {
                "scene_point_video_frame": "world", "scene_point_flow_frame": "world",
                "cam_poses_convention": "camera_to_world", "reference_frames": [0], "num_reference_frames": 1,
                "world_frame": {"index": 0, "convention": "camera_frame_at_t0"}, "video_fps": 16, "output_fps": 16,
            })
            dump(folder / "scene_point_flow_ref00000_h265_crf0.json", {
                "encoding": "ref_delta_4ch_v3", "bits": 10, "max_scale": 1, "shape_THW": [length, 8, 8],
            })
            np.save(folder / "scene_point_flow_ref00000.anchor.npy", np.zeros((8, 8, 3), np.float32))
            np.savez(base / "mask.npz", frame0=np.ones((8, 8), bool))
            identity = "demo/seg001::left"
            row = {"source": "agibot", "sample_id": identity, "any4d": str(folder.relative_to(base)),
                   "mask": "mask.npz", "video": "color.mp4"}

            def frames(*args):
                packed = np.zeros((8, 16, 3), dtype=np.uint16)
                packed[:, 8:, 0] = 1023 << 6
                for i in range(length):
                    packed[:, :8] = i << 6
                    yield packed

            def rgb(path, indices=None, native_resize=False):
                # The independent RGB timeline used without the sidecar can have another length.
                return (np.array([-1]) if indices is None else np.array(indices)), (8, 8)

            with (
                patch("worldflow.data.realworld._packed_frames", frames),
                patch("worldflow.data.realworld.decode_rgb", rgb),
                patch("worldflow.data.realworld.training_rgb", lambda x: x),
            ):
                original = load_any4d(row, base)
                self.assertEqual(original["video"].tolist(), [-1])
                dump(index, {"segment": "demo/seg001", "source_frames": length,
                             "last_source_frame": 384, "sample_ids": [identity]})
                changed = load_any4d(row, base)
                self.assertEqual(changed["video"].tolist(), list(range(0, 385, 8)))
                self.assertTrue(torch.allclose(changed["flow_m"][:, :, 0], torch.full((48, 64), 16/1023), atol=1e-6))
                for name in ("pixel_indices", "points_xyz"):
                    self.assertTrue(torch.equal(original[name], changed[name]))
                for other in (dict(row, sample_id="demo/seg001::right"), dict(row, source="sthv2")):
                    untouched = load_any4d(other, base)
                    self.assertEqual(untouched["video"].tolist(), [-1])
                    self.assertTrue(torch.equal(untouched["flow_m"], original["flow_m"]))

    def test_sparse_query_selection_keeps_pixel_geometry_and_increments_aligned(self):
        xx = np.arange(8, dtype=np.float32)
        xyz = np.zeros((49, 8, 3), np.float32)
        xyz[:, :, 0] = xx
        xyz[:, :, 1] = np.arange(49)[:, None] * xx
        uv = np.column_stack([xx, np.zeros(8)])
        sample = canonical_geometry(xyz, uv, (288, 480), "example")
        selected = sample["pixel_indices"]
        self.assertTrue(torch.equal(selected.float(), sample["points_xyz"][:, 0]))
        self.assertTrue(torch.equal(sample["flow_m"][:, :, 1], selected.expand(48, 64).float()))
        self.assertTrue(torch.equal(selected, canonical_geometry(xyz, uv, (288, 480), "example")["pixel_indices"]))
        with self.assertRaisesRegex(ValueError, "too_few"):
            canonical_geometry(xyz[:, :7], uv[:7], (288, 480), "too-small")

    def test_molmo_short_and_strided_intervals(self):
        short, padded, length = interval_to_49(8, 36, stride=4)
        self.assertTrue(padded)
        self.assertEqual(length, 8)
        self.assertEqual(short.tolist(), list(range(2, 10)) + [9] * 41)
        long, padded, _ = interval_to_49(10, 100)
        self.assertFalse(padded)
        self.assertEqual((long[0], long[-1], len(long)), (10, 100, 49))
        with self.assertRaisesRegex(ValueError, "short"):
            interval_to_49(5, 8)

    def test_sim7_qc_keys_count_and_motion_gates(self):
        def encode(array):
            stream = io.BytesIO()
            np.save(stream, array)
            return stream.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard-000001.tar"
            with tarfile.open(shard, "w") as file:
                for name, count, motion in (
                    ("kept", 20, 0.1),
                    ("invisible", 7, 0.1),
                    ("static", 20, 0.001),
                    ("qc", 20, 0.1),
                ):
                    xyz = np.zeros((49, 20, 3), np.float32)
                    xyz[-1, :, 2] = motion
                    fields = {
                        "rgb.mp4": b"opaque",
                        "meta.json": b'{"caption":"Move"}',
                        "object_mask_points_tgt.npy": encode(xyz),
                        "object_mask_points_src.npy": encode(xyz[0]),
                        "object_mask_points_valid.npy": encode(np.arange(20)[None] < count),
                        "object_mask_points_idx.npy": encode(np.arange(20)),
                    }
                    for key, data in fields.items():
                        entry = tarfile.TarInfo("task/" + name + "." + key)
                        entry.size = len(data)
                        file.addfile(entry, io.BytesIO(data))
            dump(root / "qc.jsonl", {"shard": shard.name, "sample_key": "task/qc", "status": "fail"})
            spec = {"format": "sim7", "subset": "rlbench", "qc": "qc.jsonl"}
            rows = list(native_records(spec, root, root))
            self.assertEqual([r["sample_id"] for r in rows], ["rlbench/shard-000001.tar:task/kept"])
            offset, length = rows[0]["members"]["rgb.mp4"]
            with shard.open("rb") as handle:
                handle.seek(offset)
                self.assertEqual(handle.read(length), b"opaque")

    def test_stage1_raw_uses_motion_qc_and_pinned_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "_t4w_flat" / "batch"
            capture.mkdir(parents=True)
            bundles = root / "wfm_export"
            bundles.mkdir()
            stems = ["episode_" + k for k in ("good", "qc", "pin", "static", "missing_video")]
            for stem in stems[:-1]:
                (capture / (stem + ".npz")).touch()
            dump(
                bundles / "prep_index.json",
                {
                    "gt_version": 2,
                    "motion_filter": [0.05, 3.0],
                    "episodes": [{"stem": s, "kept": "static" not in s} for s in stems],
                },
            )
            dump(bundles / ".fixed_val20.json", {"stems": [stems[2]]})
            dump(bundles / ".qc_exclude.json", {"stems": [stems[1]]})
            rows = list(native_records({"format": "demo5_raw"}, root, root))
            self.assertEqual([r["sample_id"] for r in rows], ["episode_good"])
            self.assertEqual(rows[0]["format"], "demo5_raw")

    def test_mask_gate_rejects_truncated_or_mismatched_frame_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "masks.npz"
            mask = np.ones((3, 4), dtype=bool)
            np.savez_compressed(path, frame000=mask, frame001=mask)
            self.assertTrue(_compatible_mask(path, (3, 4)))
            np.savez_compressed(path, frame000=mask, frame001=mask[:, :2])
            self.assertFalse(_compatible_mask(path, (3, 4)))
            path.write_bytes(b"truncated")
            self.assertFalse(_compatible_mask(path, (3, 4)))

    def test_warm_start_tops_up_small_post_outlier_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = np.zeros((49, 8, 3), np.float32)
            xyz[..., 0] = np.linspace(0, 0.2, 8)
            xyz[..., 2] = 1
            xyz[1:, 7, 0] += 1  # outlier removal leaves seven reliable tracks
            np.savez(root / "tracks.npz", obj_tgt=xyz, intrinsics=np.eye(3))
            np.savez(root / "episode.npz", rgb_camera1=np.zeros((2, 8, 8, 3), np.uint8), prompt="移动物体")
            row = {"sample_id": "test", "tracks": "tracks.npz", "episode": "episode.npz"}
            sample = load_native("demo5_raw", row, root)
            self.assertEqual(sample["points_xyz"].shape, (64, 3))
            self.assertEqual(len(torch.unique(sample["points_xyz"][:, 0])), 7)
            with self.assertRaisesRegex(ValueError, "too_few"):
                load_native("demo5", row, root)

    def test_custom_materialization_filters_and_reloads_every_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "clip.npz"
            np.savez(
                episode,
                prompt=np.asarray("把蓝色杯子放在黄色托盘上"),
                rgb_camera1=np.zeros((3, 12, 20, 3), np.uint8),
                intrinsics_camera1=np.eye(3),
            )
            tracks = np.zeros((3, 64, 3), np.float32)
            tracks[:, :, 2] = 1
            tracks[1:, :, 0] = 0.25
            np.savez(root / "geometry.npz", obj_tgt=tracks, intrinsics=np.eye(3))
            context = {"actor": "none", "view": "external_fixed", "origin": "real", "scene": "indoor"}
            row = {
                "format": "demo5",
                "source": "demo5",
                "sample_id": "train",
                "episode": "clip.npz",
                "tracks": "geometry.npz",
                "instruction_variants": ["移动杯子"],
                "prompt_context": context,
                "source_index": 17,
            }
            manifest = root / "source.jsonl"
            manifest.write_text("".join(json.dumps(dict(row, sample_id=s)) + "\n" for s in ("held", "train")))
            dump(root / "excluded.json", {"samples": [{"source": "demo5", "sample_id": "held"}]})
            config = {"exclude": ["excluded.json"], "sources": [{"format": "manifest", "manifest": "source.jsonl"}]}
            destination = root / "output" / "train.jsonl"
            self.assertEqual(prepare(config, root, destination, materialize=True), {"demo5": 1})
            native = TrainingDataset(manifest)[1]
            dataset = TrainingDataset(destination)
            sample = dataset[0]
            self.assertEqual(dataset.source_indices, {"demo5": [0]})
            self.assertEqual(sample["sample_id"], "train")
            for key in ("flow_m", "points_xyz", "pixel_indices", "video"):
                self.assertTrue(torch.equal(native[key], sample[key]), key)
            self.assertEqual(sample["video"].shape, (49, 288, 480, 3))
            self.assertEqual(sample["prompt_context"], context)
            self.assertEqual(sample["instruction_variants"], row["instruction_variants"])
            self.assertEqual(sample["source_index"], 17)
            self.assertEqual(collate_training_samples([sample])[0]["instruction"], native["instruction"])
            original = destination.read_bytes()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare(dict(config, sources=config["sources"] * 2), root, destination)
            self.assertEqual(destination.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
