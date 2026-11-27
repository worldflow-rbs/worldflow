# Held-out trajectory evaluation

Use a local inference bundle and a held-out JSONL manifest in the same format as the training data. Keep evaluation clips out of the training manifest. The evaluator visits the rows in order and loads the checkpoint once.

```bash
worldflow-evaluate \
  --checkpoint checkpoints/worldflow-nano \
  --manifest data/heldout.jsonl \
  --output outputs/heldout
```

The model sees the first RGB frame, the instruction and context, and the 64 anchored 3D points with their image pixels. Ground-truth motion is used for scoring. Each of the 48 future frames is scored. For a small run, add `--limit 4`. Select a GPU with `--device cuda:1` or `CUDA_VISIBLE_DEVICES`.

The defaults are 40 denoising steps, scheduler shift 5 and base seed 42. Text guidance defaults to 1 for a stage-one bundle and 2 for a stage-two bundle. Set `--guidance`, `--steps`, `--shift` and `--seed` to reproduce a particular run. The noise seed for each clip is the base seed plus its zero-based `source_index`. A manifest can preserve the original benchmark indices by including this optional integer in each row. When absent, the evaluator uses the row's ordinal within its source.

## Outputs and metric definitions

`samples.jsonl` contains one record per clip, including its source, sample ID, selected seed and all six metrics. `summary.json` reports the unweighted mean over clips and the same breakdown for each source, together with the sampling settings. Both files are published together in a new output directory.

The dataset stores successive 3D displacements in meters. The evaluator sums these increments over time to recover offsets relative to the observed frame. It compares them with the predicted offsets for the same query points.

| Metric | Definition |
| --- | --- |
| `ade` | Mean Euclidean 3D position error across the future frames and all query points, in meters. |
| `fde` | Mean Euclidean 3D position error across the query points in the final frame, in meters. |
| `pwt_1cm`, `pwt_2cm`, `pwt_5cm` | Fraction of point-frame predictions with position error at or below the corresponding threshold. |
| `pwt` | Arithmetic mean of the three threshold fractions, on a scale from 0 to 1. |
