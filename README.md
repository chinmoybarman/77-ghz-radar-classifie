# 77 GHz radar sample classifier and 60 GHz board workflow

This local app classifies recorded KTH 77 GHz radar echoes as bird, drone or human. The trained spectral CNN reached **94.17% accuracy with one scan** and **96.84% with up to nine recent scans** on 1,423 endpoints from 25 held-out recordings. On Windows, launch `run_dashboard.cmd` to browse recordings or upload a compatible sample. The [spectral experiment](reports/kth77_spectral.md) gives class-level results and timing.

The separate **Infineon DEMO BGT60TR13C** workflow supports real IF capture, raw inspection, range-Doppler processing, MTI, slow-time STFT, experimental fast-time reconstruction, 46 engineered features, Random Forest, TERRM and a compact CNN. Its included 60 GHz benchmark uses synthetic data. Hardware detection accuracy still requires measurements from that board. The [research report](reports/research_documentation.md) documents the board workflow and its limitations.

Review on 2 November 2026: **108 tests passed**. The complete three-model synthetic benchmark, saved-model inference, and Streamlit flows work. A separate KTH adapter has trained RF and CNN models using the user-provided measured archive. The local visual classifier now includes uploads and a validation-tuned CNN. See [KTH experiment results](reports/kth77_experiment.md), [dashboard refinement](reports/dashboard_refinement.md), [verification results](reports/verification.md), and [the pre-board guide](reports/pre_board_guide.md). Start the measured-data website with `./run_dashboard.ps1`.

The KTH 77 GHz archive contains range-compressed complex signals and now has its own adapter, feature schema, models, and measurement-grouped evaluation. Run it with `src/kth77_dataset.py`, keeping it outside the BGT raw-ADC pipeline. The other multi-sensor dataset still needs a separate adapter. The default BGT profile's ADC Nyquist range is about 4.8 m, with a configured gate ending at 4.5 m; plan initial capture checks within that window.

## Test the provided measured KTH archive

The first run is saved in `experiments/kth77_first/`: 83.20% RF and 83.35% CNN segment accuracy on 1,423 selected segments from 25 held-out measurements. Classes are bird, drone, and human. Reflectors are excluded; this data has no empty-scene background class. These results apply to external 77 GHz measurements, not BGT60TR13C performance.

```powershell
python src/kth77_dataset.py --input "$env:USERPROFILE/Downloads/data_SAAB_SIRS_77GHz_FMCW.npy" --output-root experiments/kth77_second --max-segments-per-measurement 64 --models rf cnn --epochs 8
```

Choose a new output directory each time. The adapter verifies the official archive before loading, holds out complete measurement rows, excludes edge-truncated segments, and saves source indices, predictions, features, spectrograms, and checkpoints. Full protocol, source checks, limitations, and output links are in [the measured-data report](reports/kth77_experiment.md).

## Local visual sample classifier

On Windows, double-click `run_dashboard.cmd` from the project folder. It uses the project’s `.venv` if present, and opens the local dashboard in your browser. The standalone Documents copy already includes trained models and recorded samples. It needs a working Python environment with the packages in `requirements.txt`.

A fresh GitHub checkout contains source and instructions. Get the official `data_SAAB_SIRS_77GHz_FMCW.npy` from [the KTH Zenodo release](https://zenodo.org/records/5896641) and adjust the path below if you saved it outside Downloads. Run these commands from the checkout to build its local models and recorded examples.

```powershell
python -m venv .venv
$python = './.venv/Scripts/python.exe'
$archive = "$env:USERPROFILE/Downloads/data_SAAB_SIRS_77GHz_FMCW.npy"
& $python -m pip install -r requirements.txt
& $python src/kth77_dataset.py --input $archive --output-root experiments/kth77_first
& $python src/dashboard_data.py --input $archive
& $python src/refine_kth77.py
& $python src/refine_kth77.py --fusion-only
.\run_dashboard.cmd
```

Open **http://127.0.0.1:8501**. **Recorded samples** starts with all 1,423 test segments. Choose a target, held-out recording, and segment. The curated and disagreement views filter the recorded examples by the individual models’ results. Compare model scores, reference labels, the complex waveform, and the Doppler image.

Use **Previous** and **Next** to step through a recording. The **Signal**, **Model comparison**, and **Benchmark** views show interactive charts, individual predictions, and the saved evaluation results. Model settings are in the collapsible sidebar. The layout adapts to narrow screens, and sample/prediction exports retain the recording and segment identity. Recorded samples reuse validated saved features and cached models to avoid repeated feature extraction; uploads still run the complete processing path.

The original blend uses **45% Random Forest and 55% CNN** and reached **88.05% single-scan accuracy**. The spectral model reads the same central 150 complex sweeps and supports both single-scan and recent-scan decisions. The app defaults to a single scan. Choose **Recent scans** to include valid previous observations of the same target, or **Original RF + CNN** to reproduce the earlier result. The [spectral experiment report](reports/kth77_spectral.md) compares all modes on the same held-out endpoints.

After the base and fusion runs above, train the spectral model from the project folder. On the standalone Windows installation, `venv` contains the Python 3.14 CUDA trainer and `.venv` runs the dashboard on CPU. For a fresh checkout, point `$trainer` at the Python environment with CUDA-enabled PyTorch. A CPU-only installation can use `--device cpu`. For a repeat run, add `--output experiments/kth77_repeat`.

```powershell
$trainer = './venv/Scripts/python.exe'
& $trainer -m src.train_kth77_spectral --input $archive --device cuda
```

To try a timestamped sequence, choose **Recent scans** under **Recorded samples**, select a segment later in a recording, and click **Download these scans**. Select **Upload a sample** and **Recent scans** to replay it. A sequence archive contains `signal`, `metadata_json`, a complex `(N,150)` array named `history` ending at the current scan, and `N` strictly increasing `times_s` values. Each sequence must come from one recording of one target. Single-scan files also work.

Use **Download this sample**, switch to **Upload a sample**, and upload that `.npz` to try your own input flow. You can also upload one finite complex `.npy` segment shaped `(150,)`, `(256,)`, `(5,256)`, or `(1280,)`; plain arrays assume a 77 GHz carrier and 17 kHz sweep rate. Exported `.npz` files include settings that are checked against the models. Five-cell inputs use the center range cell and 256-sweep inputs use the central 150 samples. Uploads use numeric arrays with pickle disabled, are limited to 8 MB, and remain on this computer. Arbitrary ADC recordings require their own acquisition metadata and processing.

The app enables the spectral modes when both `kth77_spectral` and `kth77_fusion` are available. For earlier runs it can use `kth77_fusion`, `kth77_refined`, or `kth77_first`. Training preserves existing output directories. Class scores are uncalibrated. Background detection and transfer to the 60 GHz board remain untested. Stop the local server with Ctrl+C in its terminal. The separate BGT raw-recording dashboard remains available through `python src/predict_offline.py --streamlit`.

## Run the complete synthetic example

Run commands from this directory. The direct dependencies are pinned in `requirements.txt`; `ifxradarsdk` is optional and is not needed for the benchmark.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_radar_pipeline.py -q
python src/evaluate_pipeline.py --run-synthetic-benchmark
```

The benchmark generates labeled simulations, splits independent sessions into train/validation/test, extracts features and spectrograms, trains all three models, and saves metrics, predictions, and plots. The default CNN training is intentionally small for an executable demonstration. To run only classical models:

```powershell
python src/evaluate_pipeline.py --run-synthetic-benchmark --models rf terrm
```

Repeated runs may replace generated benchmark artifacts. Keep experiments in separate roots when retaining comparisons:

```powershell
python src/evaluate_pipeline.py --run-synthetic-benchmark --output-root experiments/run_02 --seed 43
```

Preparation requires package downloads. Processing and training run locally without a network connection once dependencies are installed. For an air-gapped machine, prepare a wheelhouse for the same Python version, operating system, and architecture, then install with `python -m pip install --no-index --find-links wheelhouse -r requirements.txt`. The supplied pins reflect the observed Python 3.15.1 Windows environment; platform/vendor wheel availability must be checked on a different machine.

## Inspect and predict one recording

Accepted inputs include `.npy` with companion configuration, supported legacy IFRB `.raw.bin`, and raw text exports. Canonical shape is `(frames, 3, chirps, samples)`. A companion JSON/XML or explicit `--config` must describe the acquisition. Do not supply tracking CSVs as ADC data.

After the benchmark, this PowerShell example selects one generated recording:

```powershell
$sample = (Get-ChildItem data/raw/synthetic_pretrain -Recurse -Filter *.npy | Select-Object -First 1).FullName
python src/inspect_raw_recording.py $sample --config config/bgt60tr13c_config.json
python src/predict_offline.py --input $sample --model models/rf_bgt60tr13c.joblib --config config/bgt60tr13c_config.json --output-dir results/predictions
```

The predictor processes a finite file and exits. It saves per-frame class/confidence output and range-Doppler/spectrogram diagnostics. A model trained with synthetic observations remains labeled as such even when predicting a real input. Model scores are not calibrated probabilities.

Supplied `frame_timestamps_s` are validated and preserved in exports. If absent, timestamps are explicitly labeled nominal. Temporal smoothing restarts after a gap greater than 1.5 configured frame periods; the gap rule is a heuristic, not a claim of hardware clock verification.

Use the same predictor with `models/terrm_bgt60tr13c.npz` or `models/cnn_bgt60tr13c.pt` after those models have been trained. Only load model files you trust: Python model serialization can contain executable objects.

## Acquire with the board

Install Infineon's [Radar Development Kit](https://softwaretools.infineon.com/tools/com.ifx.tb.tool.ifxradarsdk) and a matching SDK wheel for your host. The repository does not bundle the proprietary USB SDK or board firmware. With core dependencies installed, use the actual RDK wheel directory:

```powershell
python -m pip install --no-index --find-links "C:\path\to\RDK\python_wheels" -r requirements-hardware.txt
python src/acquire_bgt60tr13c.py --config config/bgt60tr13c_config.json --num-frames 20 --output data/raw/drone/capture.npy --session-id day01_drone01 --label drone
python src/inspect_raw_recording.py data/raw/drone/capture.npy
```

The finite capture closes the device and writes data plus metadata. Missing SDK/device errors explain what is needed. The default parameters are a proposed acquisition profile; the SDK/device must validate them. Keep effective ramp/timing metadata rather than substituting nominal values from an unrelated recording.

The physical azimuth pair is RX1–RX3; elevation is RX2–RX3 in the documented board orientation. Receiver phase stability is a feature, not calibrated angle-of-arrival output.

## Register and evaluate real observations

`data/metadata/recordings.csv` uses these fields:

```text
recording_id,filepath,config_path,label,session_id,is_synthetic,subject_id,environment,notes
```

Use `drone`, `bird`, `other`, or `background` labels. File/config paths resolve relative to the registry directory. Assign every related capture from one acquisition session the same `session_id`; changing a filename does not make an independent session. Preserve labels and metadata from acquisition. Populate multiple independent sessions per class so each subset has meaningful coverage.

```powershell
python src/evaluate_pipeline.py --registry data/metadata/recordings.csv
```

Real-only training rejects synthetic registry rows. `--allow-synthetic` explicitly permits synthetic or mixed data and preserves that provenance in outputs. Never relabel such a run as real-only evaluation. Center-70% burst selection happens only after the session split; preprocessing is fit on training samples only. Test frames never drive early stopping.

Generate simulations without training:

```powershell
python src/dataset_manager.py --generate-synthetic --output-root . --sessions-per-class 5 --frames-per-recording 6
```

## Project map

| Location | Purpose |
| --- | --- |
| `config/bgt60tr13c_config.json` | Nominal hardware profile and timing assumptions |
| `src/config.py` | Validated JSON/XML configuration adapter |
| `src/acquire_bgt60tr13c.py` | Optional finite USB recording |
| `src/inspect_raw_recording.py` | Strict raw loader, quality checks, four-panel plot |
| `src/dataset_manager.py` | Registry, physics-informed simulations, grouped splits |
| `src/kth77_dataset.py` | Separate measured 77 GHz archive adapter and grouped RF/CNN experiment |
| `src/radar_dsp.py` | Range/Doppler, MTI, FTMDSE experiment, EMD and features |
| `src/models_ml.py` | RF, TERRM and CNN training/serialization |
| `src/evaluate_pipeline.py` | Reproducible evaluation and temporal filter |
| `src/predict_offline.py` | Single-file predictions and optional dashboard |
| `tests/test_radar_pipeline.py` | Numerical, loader, provenance and integration tests |
| `data/raw/`, `data/processed/` | Original observations and derived data |
| `models/`, `results/` | Generated checkpoints, metrics, predictions and diagnostics |
| `reports/research_documentation.md` | Research method, validation protocol and references |

Legacy parsing supports declared 88/92-byte IFRB headers and frame counters with real float32 data. A uint16 payload requires an explicit `payload_dtype` sidecar; unknown encodings are rejected. XML import covers simple leaf/name-value configurations, not every possible vendor document schema.

Slow-time frequencies still alias above the chirp Nyquist rate. Fast-time linear interpolation does not guarantee recovery of missing high-frequency motion. Three-receiver power averaging does not promise a fixed 4.77 dB improvement. The software selects one dominant candidate and does not perform multi-target association. See the report before using these outputs in research claims.
