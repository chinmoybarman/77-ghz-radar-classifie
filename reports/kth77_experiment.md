# First measured-data experiment: KTH/SAAB 77 GHz

The official `data_SAAB_SIRS_77GHz_FMCW.npy` archive was used to train and evaluate separate Random Forest and CNN models on 2 November 2026.

## Source verification and inventory

The file matches the [official Zenodo v2 archive](https://zenodo.org/records/5896641): 1,555,196,683 bytes, MD5 `01d66ba7b1ccc04477a9e69b2813f251`, SHA-256 `e3caa44ae92a56561d05ab51426a71c05f5d7eebf153fc7480fb3409c18f528d`. The adapter verifies both checksums before loading the source's nested NumPy object array. It does not enable pickle for arbitrary files.

The actual file contains 130 measurement rows and 75,868 segments:

| Category | Source segments | Treatment |
| --- | ---: | --- |
| Six drone types, D1–D6 | 58,768 | `drone` |
| Bird species | 7,792 | `bird` |
| Walking/running humans | 6,028 | `human` |
| Corner reflectors | 3,280 | Excluded |

There are 67 segments marked as truncated at the field-of-view edge; these are excluded before sampling. A corner reflector is not an empty scene, so no `background` class is inferred.

The source supplies range-compressed complex signals from five range cells, with 256 sweep samples per cell and a 17 kHz PRF. It does not supply this project's three-receiver 60 GHz ADC bursts. See the [source format description](https://zenodo.org/records/5845259/files/ReadMe.txt?download=1).

## Processing and validation protocol

`src/kth77_dataset.py` selects the third range cell (zero-based index 2), takes the central 150 samples (explicit Python slice `53:203`), removes the segment mean, and normalizes complex amplitude. It extracts 34 adapted EMD features plus eight spectral/envelope features, and a 32×32 slow-time STFT. No second range FFT, fabricated receive channels, AoA values, or fast-time interpolation is applied. The separate feature contract is `kth77-center150-mti-emd34-spectral8-v1`, with 42 values instead of the BGT model's 46.

An audit found that all 130 measurement rows span more than one supplied author subset. Those flags are retained, but this experiment uses a different protocol: entire measurement rows stay in one train, validation, or test partition. The split uses coarse-label stratification and seed 42. Measurement identity is the strongest grouping explicitly available here; it does not establish different days, scenes, or physical target instances.

For this first baseline, up to 64 evenly spaced non-edge segments are selected per measurement, with original segment indices and timestamps retained. Sparse selected segments are not stitched into a continuous slow-time recording or sent through temporal tracking.

| Partition | Measurements | Selected segments |
| --- | ---: | ---: |
| Train | 65 | 3,708 |
| Validation | 21 | 1,224 |
| Test | 25 | 1,423 |

Reflectors account for the other 19 measurement rows. The experiment processes 6,355 selected segments overall. This capped, recording-grouped experiment differs from the source paper and does not reproduce its reported performance.

RF uses the existing 250-tree balanced classifier. The CNN uses Adam, at most eight epochs, and validation loss for checkpoint selection. The held-out test subset was not used to choose parameters or tune thresholds. Source ranges are retained as metadata and are not input features; source labels are only training targets.

## Computed results

| Model | Test accuracy | Test balanced accuracy | Test macro-F1 | Measurement mean-score accuracy |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 83.20% | 81.72% | 0.8042 | 92.00% (23/25) |
| CNN | 83.35% | 83.61% | 0.8055 | 92.00% (23/25) |

Segment test counts are 591 bird, 640 drone, and 192 human. Mean-score aggregation averages predictions within each measurement; it is distinct from segment accuracy and from temporal filtering. All results are measured external 77 GHz results. They do not validate BGT60TR13C detection, bird rejection at 60 GHz, calibrated confidence, or generalization to new scenes. Range, SNR, scene, and target-type confounding remain possible, and segments within a measurement are correlated.

Both saved models were reloaded and reproduced all 1,423 exported test labels and confidence scores. The complete pytest suite passed **89 tests**, including 16 external-adapter tests and the existing 73 radar tests. Test format fixtures are synthetic; the experiment metrics above come from the user's measured archive. The training examples and RF confusion matrix were visually checked for legible axes and labels.

## Open the outputs

- [Measured waveform and micro-Doppler examples](../experiments/kth77_first/measured_examples.png)
- [RF confusion matrix](../experiments/kth77_first/rf_test_confusion.png)
- [CNN confusion matrix](../experiments/kth77_first/cnn_test_confusion.png)
- [Complete metrics](../experiments/kth77_first/metrics.json)
- [Measurement split manifest](../experiments/kth77_first/split_manifest.json)
- [Source inventory](../experiments/kth77_first/inventory.json)
- [RF test predictions](../experiments/kth77_first/rf_test_predictions.csv)
- [CNN test predictions](../experiments/kth77_first/cnn_test_predictions.csv)
- [Adapter test evidence](../results/verification/pytest_kth77.xml)

Separate checkpoints are `experiments/kth77_first/models/rf_kth77.joblib` and `cnn_kth77.pt`. The BGT predictor explicitly rejects these external-domain models. They do not overwrite the synthetic BGT checkpoints in `models/`.

## Reproduce or inspect

From the project directory, choose a new output folder for each run:

```powershell
python src/kth77_dataset.py --input "$env:USERPROFILE/Downloads/data_SAAB_SIRS_77GHz_FMCW.npy" --output-root experiments/kth77_second --max-segments-per-measurement 64 --models rf cnn --epochs 8
```

For inventory only:

```powershell
python src/kth77_dataset.py --input "$env:USERPROFILE/Downloads/data_SAAB_SIRS_77GHz_FMCW.npy" --output-root experiments/kth77_inventory --inspect-only
```

`--max-segments-per-measurement 0` processes all eligible segments and requires more computation. It is a new experiment; preserve the same grouping and avoid selecting changes based on the existing test results. No network or radar board is needed once the archive and dependencies are installed.
