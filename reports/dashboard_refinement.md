# Local classifier and model refinement — 2 November 2026

The local website at `http://127.0.0.1:8501` browses measured KTH/SAAB 77 GHz samples, accepts small numeric sample uploads, runs saved RF/CNN checkpoints, and displays class scores, waveform and Doppler image. Launch it with `./run_dashboard.ps1` from the project directory.

## Actual model results

The original 3,708 training segments, 1,224 validation segments and 1,423 test segments remain unchanged. Complete measurement rows remain disjoint across subsets. The refinement retains the 42-feature contract and 32×32 CNN input; it introduces no extra labels, test-set training, or changed signal preprocessing.

| Model | Validation accuracy | Validation macro F1 | Full-test accuracy | Full-test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Original RF, retained | 89.46% | 0.862 | 83.20% | 0.804 |
| CNN, up to 32 epochs | 92.24% | 0.899 | 87.56% | 0.854 |
| Combined, 25% RF + 75% CNN | 92.73% | 0.903 | 87.56% | 0.853 |

The original eight-epoch CNN scored 83.35% on the complete test subset. Longer training increased that result by 4.22 percentage points (rounded). This is still a development comparison using the previously inspected test set, not a new independent final evaluation or proof of generalization to another radar.

`src/refine_kth77.py` writes its protocol before fitting: three 500-tree RF candidates with `(max_features, min_samples_leaf)` of `(0.5,1)`, `(1.0,1)` and `(0.5,2)`; one CNN run capped at 32 epochs with validation-loss early stopping and patience 7; seed 42. It compares each candidate against its baseline using validation segment macro F1. All RF candidates scored worse on validation, so the baseline RF is retained. The longer CNN scored better and is selected. Fusion weights are chosen from RF weights `[0,.25,.5,.75,1]` using validation macro F1. Selection is written before test features and labels are loaded. No further selection follows the test evaluation.

The new run is in [kth77_refined](../experiments/kth77_refined/). See [protocol](../experiments/kth77_refined/protocol.json), [selection](../experiments/kth77_refined/selected_models.json), [metrics](../experiments/kth77_refined/metrics.json), and [CNN history](../experiments/kth77_refined/cnn_history.json). Original checkpoints and reports are retained.

## Correct-example collection

At the user's request, the website starts with **Correct demo examples**: 1,124 of 1,423 selected test segments where both selected models match the recorded label. The other 299 are available under **Model mistakes / disagreements**; **All test samples** restores the entire subset.

This filter uses reference labels solely to choose which examples to display. Every displayed classification is computed from the complex signal and saved models. It never substitutes the reference label for a prediction. Uploads bypass the collection filter. Optional labels carried in an uploaded export are explicitly shown as user-supplied reference metadata.

The **Full test results** panel always includes all 1,423 test segments. Correct-example browsing must not be reported as 100% model accuracy. No model can be promised error-free performance on unseen data.

## Input and layout corrections

- A single recommended prediction is shown with each model's separate result and readable percentage scores. The previously cramped table is now full width.
- Mean-removed complex waveforms and Doppler images use separate stacked axes so their labels remain legible in the narrow app browser.
- Settings start collapsed; sample selection and upload are visible on the main page.
- Uploads accept one finite complex `(150,)`, `(256,)`, `(5,256)`, or `(1280,)` array. Expanded archives are limited to 8 MB and object arrays are rejected without pickle. Exported samples carry checked carrier, sweep rate and preprocessing version.
- A failed upload or empty model selection clears the result view. Invalid data cannot leave an old prediction visible.
- Signal samples and prediction JSON can be downloaded. The upload page also offers a correctly classified example from each class.

## Verification

`python -m pytest -q --junitxml=results/verification/pytest_dashboard_refined.xml` completed with **108 passed**. Tests cover exact center-cell/crop selection, export/upload identity, incompatible formats and settings, object rejection, zero/constant signals, model-domain compatibility, full-bank identity, actual baseline/refined checkpoint consistency, validation selection, score fusion, correct/mistake filters, Streamlit sample/model switches, a valid upload, and invalid-upload result clearing.

Evidence: [pytest XML](../results/verification/pytest_dashboard_refined.xml). The website runs on loopback with usage telemetry disabled. Its classification domain remains measured 77 GHz range-compressed segments; 60 GHz hardware, empty scenes, and independent-day/target-instance generalization remain untested.

The actual browser flow was also exercised: download a recorded segment, select Upload a sample, select the downloaded `.npz` through the browser file chooser, and verify the same Drone prediction and class scores. A running-server import cache was cleared by restarting the task-owned server before this check. Health endpoint returned `ok`. The finished page is left open with the uploaded sample. Evidence: [browser screenshot](../results/verification/radar_dashboard.jpg). The example file is available at `C:\Users\chinm\Downloads\radar_sample.npz`.

