# Inference inputs and outputs

The released pipeline targets the joint WorldFlow Nano paper checkpoint. A scene contains one RGB image, an instruction, and a set of metric 3D query points with their matching pixels in that image. WorldFlow jointly samples future video and motion for the selected points.

## Supply your own scene

Write a JSON manifest alongside your image and NumPy archive. Paths in the manifest resolve from the manifest's directory.

```json
{
  "image": "frame.png",
  "queries": "queries.npz",
  "instruction": "Slide the blue block to the right.",
  "source": "sim7"
}
```

| Input | Shape or type | Convention |
| --- | --- | --- |
| Image | RGB image file | The model resizes the first frame to 480 × 288 pixels. |
| `points_xyz` | Float array, `(N, 3)` | Initial camera coordinates in meters, with x right, y down and z forward. |
| `pixels_uv` | Numeric array, `(N, 2)` | Zero-based column and row in the original image. Row `i` corresponds to 3D point `i`. |
| `instruction` | String | A plain-language description of the intended motion. |
| `source` | String | A preset for the scene context and metric scale, listed below. |

Save the points with `numpy.savez("queries.npz", points_xyz=points, pixels_uv=pixels)`. Supply at least one visible point and its pixel. The paper model uses 64 queries. The pipeline preserves all rows in order for exactly 64 inputs, uniformly selects rows when given more, and repeats rows when given fewer. It saves the chosen indices in the output.

## Source presets

The preset supplies the origin, actor, camera view and scene to the SPV1 prompt formatter. It also selects the scale that converts native model displacements to meters. Select the closest matching setting when preparing a new scene.

| Preset | Scene and camera context | Scale group |
| --- | --- | --- |
| `sim7`, `molmo_molmospaces` | Simulated indoor robot, fixed external view | Simulation |
| `demo5` | Real indoor robot, fixed external view | Demo5 |
| `agibot` | Real indoor robot, egocentric | Real3 |
| `sthv2` | Real indoor human, fixed external view | Real3 |
| `epic` | Real indoor human, egocentric | Real3 |
| `molmo_droid` | Real indoor robot, fixed external view | Molmo |
| `molmo_egodex` | Real indoor human, egocentric | Molmo |
| `molmo_stereo4d` | Real indoor human, moving external view | Molmo |
| `molmo_xperience` | Real indoor human, fixed external view | Molmo |
| `molmo_ytvis` | Real outdoor mixed actors, moving external view | Outdoor |

## Run and inspect a prediction

```bash
worldflow --checkpoint /path/to/worldflow-nano \
  --input my_scene/scene.json --output outputs/my_scene \
  --steps 40 --guidance 2.0 --shift 5.0 --seed 42
```

The defaults use 40 denoising steps, text guidance 2.0, scheduler shift 5.0 and seed 42. For an installation check you can use `--steps 2` and a separate output directory. Use the default sampling settings to assess model predictions.

| File | Contents |
| --- | --- |
| `video.mp4` | H.264 RGB prediction, 49 frames at 16 fps, 480 × 288. |
| `motion.npz` | Numeric arrays for the same joint sample, described below. |
| `prediction.json` | Instruction, source, scale, coordinate convention, seed and sampling settings. |

```python
import numpy as np

with np.load("outputs/my_scene/motion.npz", allow_pickle=False) as motion:
    xyz = motion["trajectories"]       # (49, 64, 3), meters in the initial camera frame
    offsets = motion["displacements"]  # (48, 64, 3), relative to frame zero
    initial = motion["points_xyz"]     # (64, 3), selected source points
    pixels = motion["pixels_uv"]       # (64, 2), original image coordinates
    rows = motion["selected_indices"]  # (64,), zero-based input query rows

    np.testing.assert_allclose(xyz[0], initial)
    np.testing.assert_allclose(xyz[1:], initial + offsets)
```

The network predicts frame-to-frame motion. The pipeline applies the preset's metric scale and accumulates those increments once to produce `displacements`. Row `t` in that array is the displacement at video frame `t + 1`.

## Checkpoint layout

```text
worldflow-nano/
├── model/
│   ├── model.safetensors.index.json
│   └── model-00001-of-00030.safetensors   (and the remaining shards)
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json              (plus the source tokenizer's vocabulary/template files)
└── Wan2.2_VAE.pth
```

Keep the original DCP and both auxiliary inputs until the converted bundle has been verified. The package reads the local tokenizer, checks generator tensor names and shapes, and loads the VAE from the same bundle. DCP conversion is intended for trusted training checkpoints.

For the full setup and bundled demo command, see the [repository README](../README.md).

## Verification

On a single H800 with PyTorch 2.8.0 and CUDA 12.8, the default 40-step block example took about 65 seconds including model loading and used 31.1 GiB of peak PyTorch-allocated GPU memory. Total device memory usage also includes the CUDA runtime and allocator reserve.

For the bundled example at seed 42, we compared the release's eager PyTorch SDPA backend with the frozen paper implementation using the same EMA weights and inputs. Across all 48 future frames and 64 queries, the mean Euclidean distance between predicted positions was 0.88 mm and the maximum was 2.02 mm. The mean absolute difference between decoded RGB samples before MP4 encoding was 0.44 on the 0–255 scale. These figures describe this single example.
