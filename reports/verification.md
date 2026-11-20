# Implementation verification — 2 November 2026

The BGT project was completed as a runnable offline research baseline and checked against the supplied master prompt. Its benchmark observations are synthetic. A later user-provided KTH archive has been evaluated in a separate measured-data workflow; see [the KTH report](kth77_experiment.md). A local sample classifier and validation-only model refinement have since been added; see [the dashboard report](dashboard_refinement.md). No physical BGT60TR13C device was exercised. The current combined suite passes **108 tests**, including the radar, external-adapter, upload, filtering, checkpoint consistency and Streamlit checks. Evidence: [current pytest XML](../results/verification/pytest_dashboard_refined.xml). Earlier test counts below describe historical runs.

## Follow-up review and corrections

The review suite completed with **73 passed, no failures, no skips, and no pytest warnings**. Evidence: [review pytest XML](../results/verification/pytest_review.xml). The full RF/TERRM/CNN benchmark was rerun in the separate `experiments/review_20261102` directory, preserving earlier generated artifacts. Evidence: [review benchmark log](../results/verification/synthetic_benchmark_review.log) and [review metrics](../experiments/review_20261102/results/synthetic_benchmark/metrics.json). Test accuracies remained RF 24/24, TERRM 24/24, and CNN 22/24. These remain synthetic-only results.

Additional corrections covered by tests:

- ADC validation now checks each frame and receiver for clipping. One bad frame previously could be hidden by averaging over many clean frames.
- Direct SDK ADC validation rejects complex values, invalid shapes, and fractional values declared as ADC counts.
- Supplied frame timestamps are validated and preserved in prediction/evaluation exports. Missing timestamps are explicitly labeled nominal.
- Temporal smoothing restarts after timestamp gaps greater than 1.5 configured frame periods. The first three frames after a reset abstain again. This is an interruption heuristic; nominal timestamps do not prove that hardware frames were not dropped.
- Invalid SDK configuration readback raises a validation error rather than silently falling back to requested settings. Missing readback APIs still produce the documented unverified-settings notice.
- External object-array research files are rejected with an explanation that a dedicated adapter is needed, without unpickling them in the ADC loader.

The Streamlit Analyze recording flow was rerun successfully, including timestamp-provenance export. Its output is in `results/predictions/review_dashboard/`.

The [pre-board guide](pre_board_guide.md) fixes prior guidance: KTH data are already range-compressed complex signals, and most FMCW data in the multi-sensor source are range-Doppler maps. The KTH adapter and first measured-data experiment have since been completed with a separate schema and split audit; the multi-sensor adapter remains future work. Earlier 5 m/8 m capture examples were inappropriate for the default profile's roughly 4.8 m ADC Nyquist limit and 4.5 m analysis gate.

## Executed checks

| Check | Result |
| --- | --- |
| `python -m pytest tests/test_radar_pipeline.py -q --junitxml=results/verification/pytest.xml` | 58 passed, no failures or skips |
| `python src/evaluate_pipeline.py --run-synthetic-benchmark` | Completed RF, TERRM, and CNN training and held-out evaluation |
| Raw inspection CLI on `synthetic_42_drone_000.npy` | Loaded `(6, 3, 64, 128)` data and saved a four-panel diagnostic |
| Prediction CLI using each of the three saved models | Completed with predictions, CSV, JSON, STFT/FTMDSE arrays, and plots |
| Streamlit AppTest, including an Analyze recording button click | Completed synthetic inference without app exceptions or error elements |
| Visual inspection of raw and prediction diagnostic PNGs | Legible axes, legends, and synthetic watermarks |
| USB capture tests with a mock SDK | Finite capture, effective metadata readback, and disconnect on success/failure checked |

The bare `pytest` executable was not on this host's PATH; invoking the same pytest package through `python -m pytest` succeeded. Dependencies were already installed. The observed environment is Python 3.15.1 on Windows, with the direct dependency versions recorded in `requirements.txt` and CPU PyTorch 2.14.1.

## Actual synthetic benchmark output

Seed 42 generated 20 independent simulated sessions, five per class, with six frames per session. The session split contains 12 training, four validation, and four test sessions. Middle-70% selection provides 48 training bursts; validation and test each contain 24 bursts. Track filtering emits on 12 of the 24 test frames after its per-recording warmup.

| Model | Test single-scan accuracy | Test track-filtered accuracy |
| --- | ---: | ---: |
| Random Forest | 100.00% (24/24) | 100.00% (12/12) |
| TERRM | 100.00% (24/24) | 100.00% (12/12) |
| Compact CNN | 91.67% (22/24) | 100.00% (12/12) |

These small, idealized simulator scores are software checks. They are not estimates of real drone/bird classification performance. Track scores cover fewer frames; the metrics file also reports single-scan accuracy on those same eligible frames. The example CLI prediction is a functional check and is not an additional held-out experiment.

Machine-readable evidence: [pytest XML](../results/verification/pytest.xml), [benchmark log](../results/verification/synthetic_benchmark.log), [metrics](../results/synthetic_benchmark/metrics.json), and [session assignments](../results/synthetic_benchmark/split_manifest.json).

## Fixes covered by regression tests

- Use effective SDK settings when a capture sidecar also contains the originally requested settings.
- Reject acquisition conflicts across explicit and recorded configurations, including chirp slope, and reject a nonexistent explicit configuration path.
- Preserve recording-specific synthetic provenance when an explicit configuration contains other provenance fields.
- Reject NaN/Inf ASCII rows instead of skipping them as header text.
- Accept float32 representation error in legacy timing fields while rejecting materially invalid timing or nonintegral sample grids.
- Compare chirp slope during saved-model compatibility checks so a changed ramp cannot silently change the range scale.
- Emit balanced accuracy as documented, and direct optional dependency errors to an existing requirements file.
- Plot the resized STFT on its actual window-center time and frequency coordinates.

The suite additionally covers real-IF signed Doppler and aliasing, MTI cancellation, conjugate FTMDSE filtering, interpolation timestamps, EMD reconstruction and numerical feature oracles, quality guards, legacy endian/layout variants, session isolation, synthetic provenance, temporal-filter state, model serialization, CNN training, and exported predictions.

## Research guidance and implementation boundaries

The three primary manuscripts were consulted online. The main report contains the full method and reference list. The source checks confirmed:

- Passafiume et al. validate their model using 77 GHz measurements. Applying the motion model at this project's 60 GHz acquisition profile requires separate hardware validation. [Publisher-provided full text](https://www.researchgate.net/publication/350304139_Modeling_Small_UAV_Micro-Doppler_Signature_Using_Millimeter-Wave_FMCW_Radar)
- Peter and Reddy's method includes range-Doppler filtering, idle interpolation, and subsequent EMD processing. This implementation provides the requested filtering/interpolation front end; it does not claim to reproduce their complex-EMD mode-selection stage or guarantee unaliased recovery. [Primary manuscript, Sections III.A–D](https://arxiv.org/html/2204.09621v1)
- Oh et al. define a binary sign in Eq. (8). Their printed classifier target equation and reported parameter setting require an explicit implementation choice; this package documents its 0/1 target adaptation. [Author-uploaded manuscript](https://www.researchgate.net/publication/332996672_A_UAV_Classification_System_based_on_FMCW_Radar_Micro-Doppler_Signature_Analysis)
- Infineon identifies RX1–RX3 as the azimuth baseline and RX2–RX3 as the elevation baseline in the documented orientation. [Official technical response](https://community.infineon.com/t5/Radar-sensors/How-does-BGT60-Radar-resolve-multiple-targets-in-E-H-plane/td-p/1230063)

Paper ambiguities, hardware corrections, feature conventions, diagnostic-only pruning, and simulator limitations are explicitly documented in [research_documentation.md](research_documentation.md). The software is a runnable research baseline; hardware capture, transfer to real data, calibrated AoA, and detection performance still require measured recordings.

