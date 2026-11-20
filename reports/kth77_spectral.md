# Improving the measured 77 GHz classifier

## Results on recordings held out from training

| Model | Scans per decision | Correct / 1,423 | Accuracy | Macro F1 | Drone recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original forest and spectrogram CNN | 1 | 1,253 | 88.05% | 0.858 | 90.00% |
| Original models with score averaging | Up to 9 | 1,311 | 92.13% | 0.904 | 94.84% |
| Spectral CNN | 1 | 1,340 | 94.17% | 0.926 | 96.56% |
| Spectral CNN with score averaging | Up to 9 | 1,378 | 96.84% | 0.959 | 98.91% |

The expanded training set won on validation. Both selected modes assigned 100% of the score to the new CNN. The checkpoint has 50,319 parameters and occupies 211,257 bytes. Fitting the winning model took 8.7 s on an RTX 5060 Laptop GPU. The two training budgets and their evaluation together took 24.5 s after installation. The single-scan accuracy gain over the previous model was 6.11 percentage points, with an approximate 95% recording-bootstrap interval of +3.32 to +9.34. With recent scans the gain was 8.78 points, with an interval of +5.64 to +12.54. Comparing both models with recent scans, the new CNN made 67 additional correct decisions, a gain of 4.71 points with an interval of +2.48 to +7.24.

Recent scans correctly classified 633 of 640 drone endpoints. Eight of the 783 bird or human endpoints received a drone prediction. The median time between the first and last scan used was 0.798 s. A full nine scans were available at 972 of 1,423 endpoints. The other endpoints are included in every reported metric. The three classes in this benchmark already contain a known target.

## Which research applies here?

Karlsson and colleagues trained a small CNN on the Doppler spectrum of the same KTH recordings. They aligned the target's main spectral peak before classification. Their experiment used all five range cells and a different split and label scheme. For compatibility with existing exported samples, this app uses the central cell. Its CNN sees the full 150-sweep spectrum both before and after peak alignment, plus spectra from three shorter, overlapping sections. The five CNN input channels are derived from a single range cell. [Original IEEE paper](https://kth.diva-portal.org/smash/get/diva2%3A1658178/FULLTEXT01.pdf)

The archive stores five spatial range cells and 256 sweeps together in each 1,280-value column. Separate columns from one measurement provide successive observations. Their timestamps average about 100 ms apart and contain gaps. An earlier interpretation of the Ertekin and Güney comparison treated the five blocks in one column as five timestamps. The archive documentation identifies them as range cells. That paper's reported 97.72% therefore cannot establish the benefit of waiting for five successive scans. It also used the supplied segment split, which shares measurement rows across partitions. [KTH dataset and readme](https://zenodo.org/records/5896641) · [Ertekin and Güney, methods and Table 4](https://www.intechopen.com/journals/1/articles/709)

Two studies motivate an independent test of averaging actual successive observations. Karlsson's thesis measured how integrating observations affects classification across signal-to-noise levels. Yang and colleagues tested a compact network followed by temporal score fusion with recording-disjoint folds on a different radar dataset. Their gain varied with the observation design. Here the averaging window contains only the current scan and its available past. [Karlsson thesis, chapter 6](https://kth.diva-portal.org/smash/get/diva2%3A1797382/FULLTEXT01.pdf) · [Yang et al., 2026](https://pubmed.ncbi.nlm.nih.gov/42944518/)

## Local comparison

The training split has 65 recordings. The previous models used 3,708 clips from these recordings. Two spectral CNNs compare this cap against all 42,116 valid clips from the same rows. Each class and each recording within a class receives equal expected sampling weight. A secondary training objective predicts the original source type, such as drone model or human activity. Predictions use the signal alone. Range, timestamp, recording identity and reference label are excluded from the model input. Timestamps determine which previous scans are close enough to average.

The validation split contains 1,224 endpoints from 21 other recordings. Validation macro F1 determines the checkpoint, the spectral weight and the window. Single-scan validation macro F1 selects the training cap, with sequence macro F1 breaking a tie. Candidate windows contain up to 1, 3, 5 or 9 scans. A gap over 150 ms, a truncated scan or the start of a recording resets the history. Every test endpoint receives a prediction, including those with one available scan. Accuracy differences use the same 1,423 endpoints in 25 held-out recordings. The paired bootstrap resamples complete recordings 10,000 times.

The control reuses the previously trained forest and spectrogram CNN and averages their combined predictions across the history. The new sequence model averages spectral predictions across the history and blends this result with the original models' estimate for the current scan. These two experiments distinguish the effect of more observations from the effect of retraining.

For interactive prediction, the forest now uses one CPU thread per sample. Across 64 evenly spaced test clips after a warm-up, median CPU time for feature extraction and both original models fell from 41.4 ms to 17.8 ms. Their forest scores agreed exactly on these clips. Timings exclude model loading, browser rendering and time to acquire radar scans.

Using only the selected spectral checkpoint on CPU, median preprocessing and forward-pass time was 0.40 ms for one scan and 0.90 ms for nine scans. Each figure uses 64 samples after a warm-up. The dashboard also computes comparison models and plots. To use the saved model directly from the project directory, run `python -m src.kth77_spectral --input radar_sample.npz`. Add `--recent` for a timestamped sequence.

## Reproduce

First build the base models and the fusion experiment using the commands in the [project README](../README.md). From the project folder, run the following with a compatible CUDA PyTorch installation. The control runs on CPU. Use a new output folder to repeat an experiment with existing results.

```powershell
$trainer = './venv/Scripts/python.exe'
$dashboard = './.venv/Scripts/python.exe'
$archive = "$env:USERPROFILE/Downloads/data_SAAB_SIRS_77GHz_FMCW.npy"
& $trainer -m src.train_kth77_spectral --input $archive --device cuda
& $dashboard -m src.train_kth77_spectral --input $archive --legacy-control
```

The Windows trainer uses Python 3.14.6 and `torch==2.14.1+cu130` from the official CUDA 13.0 wheel index. On other machines, install a compatible wheel into the training environment and check `torch.cuda.is_available()`. In the initial Python 3.15 test, a CUDA backward pass produced a Windows access violation on interpreter shutdown after the checkpoint had been saved. The Python 3.14 GPU rerun exited cleanly and reproduced every test classification in both modes. PyTorch has documented a worker-thread issue affecting the 2.14.1 Windows Python 3.15 build. [PyTorch issue](https://github.com/pytorch/pytorch/issues/200779) · [Installation guide](https://pytorch.org/get-started/locally/)

The spectral run stores its selection, aggregate metrics, per-endpoint predictions, checkpoint and replay samples under `experiments/kth77_spectral`. The control writes to `experiments/kth77_temporal_control`. Inference on a trained checkpoint also runs on CPU. The original test split has been examined in earlier development. Another acquisition campaign would be needed to estimate performance on new sites or on the Infineon 60 GHz board.
