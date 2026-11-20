# Your project before the radar arrives

The software in this folder is a working offline research baseline. It processes simulated BGT60TR13C ADC recordings, trains models, predicts one file, and exports plots and CSVs. A separate KTH adapter now processes the user's measured 77 GHz archive and has trained and evaluated RF/CNN models. The USB acquisition script exists and has mock-device tests. Actual BGT60TR13C operation and classification remain unverified. See [the measured-data report](kth77_experiment.md) for the completed external experiment.

## What has been built and tested

| Component | Current state |
| --- | --- |
| JSON/XML configuration and raw-file inspection | Implemented; tested using generated fixtures and supported legacy formats |
| Three-receiver range FFT, MTI, Doppler, target-cell selection | Implemented; numerical tests verify known signals |
| Slow-time STFT and 34 EMD + 12 physical features | Implemented; output is a 32×32 image and a 46-value vector |
| Fast-time filtering and idle interpolation | Experimental front end implemented; no guaranteed high-frequency recovery |
| Random Forest, TERRM, and CNN | Synthetic training and saved-model inference verified |
| Session splitting and temporal label smoothing | Implemented; supplied timestamps preserved and smoothing resets after gaps |
| CLI, Streamlit interface, plots, and CSV exports | Verified on synthetic recordings |
| Public research dataset ingestion | KTH adapter and first measured RF/CNN experiment completed; other sources need adapters |
| Real BGT60TR13C capture and measured evaluation | Requires the board and actual recordings |

The feature extraction and TERRM settings are documented adaptations of the papers. This is not a claim of exact author-code reproduction.

## Corrections to the earlier advice

**The KTH dataset is already range-compressed.** Its README describes complex signals from five range cells, each with 256 samples across a mechanically scanned beam. It is useful for a separate micro-Doppler experiment starting at the complex slow-time signal. It cannot verify the board's raw ADC loader, real range FFT, or three-antenna phase features. [Dataset](https://zenodo.org/records/5896641), [format description](https://zenodo.org/records/5845259/files/ReadMe.txt?download=1).

**The second dataset's FMCW recordings are mostly range-Doppler maps.** Its authors provide a representative raw FMCW example as well. Maps can support a separate classifier, but full raw processing cannot be reconstructed from them. [Primary paper](https://www.nature.com/articles/s41597-026-06802-6).

**Public datasets use separate workflows.** The user has provided the KTH archive, and it has now been evaluated using `src/kth77_dataset.py`. Its models are in `experiments/kth77_first/models/`; BGT checkpoints in `models/` remain simulation-trained. Do not rename a downloaded file or attach the 60 GHz config to make it load. Preserve the source frequency, sampling rate, data representation, labels, and measurement identity.

**My earlier 5 m and 8 m capture examples do not fit the current profile.** With the nominal 1 MHz ADC rate and 59–61 GHz sweep in 128 μs, the real-IF ADC Nyquist range is about 4.80 m, and the configured analysis gate ends at 4.5 m. This is a sampling limit, not a measured detection range. Begin calibration at measured distances within the configured window, such as 1–3 m. Longer-range experiments require a redesigned, hardware-validated profile and compatible retraining.

**Classification is not calibrated detection.** The current DSP selects the strongest range candidate and a model assigns a class. A bright range cell, a pruning flag, or a high classifier score does not establish a drone's presence. Dedicated detection thresholds and false-alarm measurements remain future validation work. The receiver phase features measure stability rather than calibrated angles.

## Work through the project in this order

Open PowerShell in the project folder:

```powershell
python -m pytest tests/test_radar_pipeline.py -q
```

First inspect the existing synthetic sample and run its saved Random Forest:

```powershell
python src/inspect_raw_recording.py data/raw/synthetic_pretrain/synthetic_42_drone_000.npy
python src/predict_offline.py --input data/raw/synthetic_pretrain/synthetic_42_drone_000.npy --model models/rf_bgt60tr13c.joblib --output-dir results/predictions
python src/predict_offline.py --streamlit
```

Explain the raw-waveform, range, velocity, and spectrogram axes to yourself. Then study `radar_dsp.py`: fast-time FFT estimates range; complex changes across chirps carry Doppler; MTI removes the per-burst stationary component; STFT displays Doppler changes inside the burst. EMD and the physical features feed RF/TERRM; the slow-time spectrogram feeds the CNN. The experimental fast-time image is diagnostic and is not the default CNN input.

Next reproduce training in a separate experiment folder:

```powershell
python src/evaluate_pipeline.py --run-synthetic-benchmark --output-root experiments/practice_01
```

This generates simulations, splits complete sessions, selects middle-70% training bursts, trains the three models, and saves actual computed metrics. Inspect the split manifest, confusion matrices, and CNN loss curve. High synthetic scores verify the simulator experiment; they do not estimate field accuracy.

After that, study the completed KTH experiment in `experiments/kth77_first/`. The adapter preserves measurement IDs and supplied split indicators, verifies the complex-array layout, and uses the source PRF of 17 kHz. It extracts center-cell slow-time Doppler/STFT and EMD features with a separate 42-feature schema and external checkpoints. The default split holds out whole measurements because the supplied author splits share measurements. See the measured-data report before comparing scores or planning a larger run.

When the board arrives, verify the SDK environment, accepted settings, receiver order, range scale, Doppler sign, and data quality. Save original ADC data plus acquisition metadata. For an initial finite capture within the current range window:

```powershell
python src/acquire_bgt60tr13c.py --config config/bgt60tr13c_config.json --num-frames 20 --output data/raw/drone/day01_hover_2m.npy --session-id day01_drone_session --label drone
python src/inspect_raw_recording.py data/raw/drone/day01_hover_2m.npy
```

Twenty frames at the nominal frame rate is only a roughly 1.3-second capture test. Build a larger dataset across independent days, scenes, target instances, and conditions. Captures sharing an acquisition session keep the same session ID even if saved to separate files. Register raw files in `data/metadata/recordings.csv`, then train and evaluate on independent real sessions. Use only labels you have measured; a drone/background pilot does not demonstrate bird rejection.

Be ready in stages: synthetic software verified first, external measured-data experiment second, board acquisition and calibration third, and independently evaluated BGT60TR13C classification last.
