# Training data

The trainer and held-out evaluator read a JSONL manifest with one object and instruction per row. A video can appear in several rows if its tracked objects differ. You can use your own clips or index the original training artifacts. Dataset owners supply the recordings, tracks, masks and selection metadata under their respective terms.

## Use your own clips

For each clip, supply an RGB video and a numeric NumPy archive with two arrays. `trajectories` contains absolute XYZ positions of the same points through time, in meters in the first camera's fixed coordinate frame. `pixels_uv` contains the corresponding frame-zero image coordinates, in native pixels ordered horizontally then vertically. The video and tracks must use the same timeline. Include at least eight finite, initially visible query points.

```python
import numpy as np

# xyz has shape [T, N, 3]. uv has shape [N, 2].
np.savez("cup-tracks.npz", trajectories=xyz.astype("float32"), pixels_uv=uv)
```

Create `clips.jsonl` beside the files. Set `source` to the source preset whose language context and motion scale apply to your data. Available presets are listed in [inference.md](inference.md).

```json
{"format":"tracks","source":"demo5","sample_id":"cup-001","instruction":"Move the cup to the left.","video":"cup.mp4","tracks":"cup-tracks.npz"}
```

Create `sources.json` in the same directory.

```json
{"sources":[{"format":"manifest","manifest":"clips.jsonl"}]}
```

```bash
worldflow-prepare --config sources.json --output data/train.jsonl --validate
```

`--validate` decodes every admitted clip and checks the tensor contract. To preprocess once for repeated training, use `--materialize`. This writes one compressed numeric NPZ per record into a versioned sibling directory. Copy the manifest and that directory together to relocate the dataset. Without materialization, keep source files and tar shards in place. The manifest refers to them by relative path.

### Prepared numeric samples

If you already have canonical training tensors, store these arrays with `np.savez` and use a row such as the following.

```json
{"format":"prepared","source":"sim7","sample_id":"block-001","instruction":"Slide the block forward.","sample":"block.npz"}
```

| NPZ key | Shape and dtype | Meaning |
| --- | --- | --- |
| `video` | `[49, 288, 480, 3] uint8` | RGB, channels last, values 0–255 |
| `points_xyz` | `[64, 3] float32` | Frame-zero queries, meters in the fixed frame-zero camera |
| `pixel_indices` | `[64] int64` | Observed frame-zero pixels, flattened as `v * 480 + u` |
| `flow_m` | `[48, 64, 3] float32` | Consecutive differences of the resampled absolute XYZ trajectories |

Point index `n` identifies the same query in every array and frame. Recover the full position at frame `t` by adding the first `t` increments to `points_xyz`. Source adapters return the original displacement units. The training objective applies the stage-specific scale. The original YT-VIS reconstruction uses relative geometry and has its own outdoor scale.

All row formats accept optional `prompt_context` with exactly `origin`, `actor`, `view` and `scene`, a list of equivalent `instruction_variants`, and a nonnegative `source_index` for a known source-local evaluation index. The evaluator uses the base instruction. Keep ground truth and future frames out of the instruction and context.

## Rebuild the original training indices

The two stages use different SpatialCog (Demo5) inputs. Both commands run on CPU and write manifests outside the package.

```bash
worldflow-prepare --root /path/to/worldflow_training_data --stage 1 --output data/stage1.jsonl
worldflow-prepare --root /path/to/worldflow_training_data --stage 2 --output data/stage2.jsonl
```

`--root` expects the following artifact layout. The paths below are relative to that root. The command does not download source data. Native Real3 decoding requires the FFmpeg executable with HEVC decoding and RGB48LE output. The other native readers use the Python dependencies installed with WorldFlow.

| Stage and source | Payloads and selection |
| --- | --- |
| 1, SpatialCog | `datasets/demo5/_t4w_flat/<batch>/episode_*.npz` and `datasets/demo5/wfm_export/<episode>/sample.npz`. The GT-v2 `prep_index.json` supplies the 5 cm to 3 m motion gate. The export's `.qc_exclude.json` and `.fixed_val20.json` remove tracking failures and 30 pinned validation episodes. The original index yields 82,789 training episodes with native Chinese capture text. |
| 2, SpatialCog | The same payloads, with the English `training.jsonl` in `metadata/objectflow_run/demo5_manifests/snapshots/demo5-en-20260903T140412Z-51c4ad14/`. It contains 82,588 episodes before the 400 benchmark exclusions, leaving 82,188 eligible. |
| 2, Sim7 | Uncompressed `shard-*.tar` WebDataset archives. Under `datasets/simulation`, ManiSkill, MiKASA and Meta-World use `webdataset/sim_v1_wbs_v0.15/<subset>/flatten`. RoboTwin uses the v0.16 counterpart. MetaFold and RLBench use `packed_shards/<subset>`, RoboCasa uses `packed_shards_new/robocasa/flat`. The index applies task, partial-visibility and per-shard QC gates. |
| 2, Real3 | `datasets/realworld/<source>/{_meta,time_seg,sam2,any4d,vlm_anno}` for `agibot`, `sthv2` and `epic`. The index expands each record of `_meta/complete_samples.jsonl.gz` into its compatible, nonempty SAM2 masks. Each mask supplies observed query pixels. Any4D supplies the frame-zero anchor and encoded reference-frame offsets. |
| 2, MolmoMotion | Original `videos/`, `tracks/` and source-specific `camera/` under `datasets/molmo_motion_1m/<subset>`. The object interval manifests in `metadata/qc/molmo_scale_qc_v1_20260915` cover DROID, EgoDex, Stereo4D and Xperience. MolmoSpaces uses `domain_scale_v3_simfix_20260916/molmospaces.jsonl`, and YT-VIS uses `outdoor_qc_v1_20260916/ytvis.jsonl`, both under `metadata/qc`. |

Stage 2 applies the 2,000-ID `metadata/objectflow_run/objectflow_benchmark_v1.json` and the source-keyed `metadata/qc/real3_qc_demo5_aligned_v1_20260915/real3_qc_exclude.json`. It verifies the benchmark, Real3 exclusion list and each MolmoMotion selection against the SHA-256 values in the published domain-scale contract. Sim7 reads its four subset QC lists. RoboTwin drops all flagged entries. MetaFold, RLBench and RoboCasa drop entries marked `fail`. Sim7 IDs include the subset, shard basename and sample key.

The adapters resample native trajectories and RGB to 49 aligned frames. For MolmoMotion, intervals with 8–48 usable frames repeat the final frame, and longer intervals are sampled uniformly. MolmoSpaces accounts for its stride-four tracks and uses the matching raw video frames. Query selection uses visible frame-zero candidates, deterministic replacement for 8–63 points, identity for 64, and voxel sampling above 64. Demo5 uses calibrated native camera intrinsics when available. RGB and XYZ always share the selected point ordering through the projected source pixels.

Stage 1 uses the earlier SpatialCog voxel policy. It removes trajectory outliers and selects voxel representatives, repeating evenly spaced representatives when fewer than 64 remain. In Stage 2, Sim7 additionally filters the maximum distance from observed partial points to their future positions to the range 0.05–3 m. For matching the original point selection, install `numpy==1.26.4` in the data-preparation environment. NumPy versions can order ties differently in dense voxel groups, producing different representative points.

An AgiBot dataset may supply `agibot/_meta/train_temporal_cap8.jsonl`. For the training sample IDs listed there, the native reader takes RGB and trajectory frames 0, 8, …, 384 from the 16 fps source. This covers 24 seconds of source motion in the three-second training window. The sidecar also applies to existing native manifests. Prepared NPZs already contain their sampled tensors.

To check a small selection of the original artifacts, add `--limit-per-source 1 --validate` to the stage-2 command. This decodes 17 clips, one from each of seven Sim7 subsets, three Real3 sources, six MolmoMotion subsets and SpatialCog. A full run iterates the complete source catalogs, keeping duplicate detection on disk. It publishes the output manifest after the preparation succeeds.

## Alternative source layouts

A `--config` JSON file can combine any number of source specifications. Paths in a specification resolve relative to the configuration file. Paths inside an existing `format: manifest` input resolve relative to that input file. Use the standalone `--root` recipe above when reproducing the complete selection.

```json
{
  "exclude": ["benchmark.json", "real3_qc_exclude.json"],
  "sources": [
    {"format":"demo5_raw","root":"datasets/demo5"},
    {"format":"sim7","subset":"robotwin","root":"shards/robotwin","qc":"qc/robotwin.jsonl","skip_statuses":"all"},
    {"format":"any4d","source":"sthv2","root":"datasets/realworld"},
    {"format":"molmo","root":"datasets/molmo_motion_1m","manifest":"qc/egodex.jsonl"},
    {"format":"manifest","manifest":"custom/clips.jsonl"}
  ]
}
```

Use `{"format":"demo5","root":"datasets/demo5","manifest":"snapshot/training.jsonl"}` to load a pinned SpatialCog snapshot. The reader relocates the recorded capture paths using the bundle's `wfm_export/<episode>/sample.npz` suffix. Optional `original_root` handles other export layouts. A Sim7 specification accepts a `glob` for a selection of uncompressed shards. Partition large jobs by non-overlapping shard or source manifests. Supply exclusions to each partition. Matching always uses the pair `(source, sample_id)`. Repeated sample IDs within one source raise an error.
