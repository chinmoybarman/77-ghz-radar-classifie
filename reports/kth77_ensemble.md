# Spectral CNN ensemble experiment

On 26 November 2026, two additional spectral CNNs were trained on the NVIDIA
GeForce RTX 5060 Laptop GPU using seeds 7 and 123. Each used all eligible training
segments, the existing architecture and optimizer, and the existing measurement
partitions. Early stopping selected epoch 9 for seed 7 and epoch 21 for seed 123.

The candidate set consisted of the deployed model, each new model individually,
and an equal-probability ensemble of all three. Checkpoints were selected using
single-scan validation macro F1. The existing validation search chose scan
history length and baseline blend weight. The experiment selected the candidate
with highest single-scan validation macro F1, using recent-scan macro F1 to break
ties. Selection was written to disk before evaluating the winning candidate on
the test partition.

The ensemble won validation selection. Its single-scan validation macro F1
was 0.9565, compared with 0.9510 for the deployed model.

| Test mode | Deployed model | Selected ensemble | Additional correct |
| --- | --- | --- | --- |
| Single scan | 94.17% | 94.17% | 0 |
| Recent scans | 96.84% | 96.63% | -3 |

The test contained 1,423 endpoints from the existing held-out measurement rows.
Paired recording bootstrap intervals for accuracy differences were
[-0.83, 0.77] percentage points for single scans and [-0.60, 0.14] for recent
scans. These intervals include zero. This experiment provides no evidence of
an accuracy improvement. The deployed checkpoint was retained.

The test partition has been inspected during earlier development. These
comparisons describe performance on that existing benchmark. An independent
recording set is needed to measure generalization after repeated development.
All histories use preceding scans from the same recording and reset at invalid
edges or timestamp gaps.

Run from the project folder with a CUDA-enabled Python environment:

```powershell
& ./venv/Scripts/python -m src.refine_spectral_ensemble --input "$env:USERPROFILE/Downloads/data_SAAB_SIRS_77GHz_FMCW.npy" --output experiments/kth77_ensemble_repeat
```

The output contains per-seed checkpoints, the validation candidate table,
selection, predictions, and metrics. The original run is stored locally in
`experiments/kth77_ensemble_20261126/`. The runner rejects an existing output
directory.
