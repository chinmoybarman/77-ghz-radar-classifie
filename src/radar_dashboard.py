"""Local measured-data sample browser. Run with streamlit run src/radar_dashboard.py."""
from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_data import (EXPERIMENT_ROOT, PRF_HZ, analyze_signal, check_model,
    export_sample, load_demo_bank, read_upload, model_selection, combined_prediction,
    demo_positions, REFINED_ROOT)
from src.kth77_dataset import FEATURE_NAMES
from src.models_ml import load_model, model_probabilities

MODEL_FILES = {"Random Forest": "rf_kth77.joblib", "CNN": "cnn_kth77.pt"}


@st.cache_resource(show_spinner=False)
def dashboard_model(path):
    bundle = load_model(path)
    check_model(bundle)
    # Streamlit sessions can share this model. Inference uses eval/no_grad only.
    if bundle["kind"] == "cnn":
        import torch
        torch.set_num_threads(2)
    return bundle


@st.cache_data(show_spinner=False)
def samples():
    return load_demo_bank()


@st.cache_data(show_spinner=False, max_entries=128)
def classify(values, model_paths):
    return analyze_signal(values, {name: dashboard_model(path) for name, path in model_paths})


@st.cache_data(show_spinner=False)
def bank_scores(model_paths):
    with np.load(EXPERIMENT_ROOT / "processed" / "test.npz", allow_pickle=False) as data:
        features, spectrograms = data["features"], data["spectrograms"]
        scores = {}
        for name, path in model_paths:
            bundle = dashboard_model(path)
            order = [list(bundle["classes"]).index(label) for label in ("bird", "drone", "human")]
            scores[name] = model_probabilities(bundle, features, spectrograms)[:, order]
        return scores


def plots(extracted):
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), constrained_layout=True)
        fig.set_facecolor("#0e1525")
        for ax in axes:
            ax.set_facecolor("#0e1525")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
        time_ms = np.arange(150) / PRF_HZ * 1000
        axes[0].plot(time_ms, extracted["signal"].real, color="#5eead4", lw=1, label="Real (I)")
        axes[0].plot(time_ms, extracted["signal"].imag, color="#a5b4fc", lw=1, alpha=.8, label="Imaginary (Q)")
        axes[0].set(xlabel="Time within sample (ms)", ylabel="Normalized amplitude", title="Complex echo after mean removal")
        axes[0].legend(loc="upper right", fontsize=8, frameon=False)
        image = axes[1].pcolormesh(extracted["time_s"] * 1000, extracted["frequency_hz"] / 1000,
            extracted["spectrogram"], shading="auto", cmap="magma", vmin=0, vmax=1)
        axes[1].set(xlabel="STFT window center (ms)", ylabel="Doppler frequency (kHz)", title="Micro-Doppler image · 32 × 32")
        fig.colorbar(image, ax=axes[1], label="Normalized power", shrink=.85)
        return fig


def main():
    st.set_page_config(page_title="µDoppler Lab · Radar classifier", page_icon="📡", layout="wide",
        initial_sidebar_state="collapsed")
    st.html("""<style>
      .stApp { background: #0b1120; color: #e5edf8; }
      [data-testid="stHeader"] { background: #0b1120; }
      [data-testid="stSidebar"] { background: #111b2e; }
      .stMainBlockContainer { max-width: 1100px; padding: 1.5rem 2rem 3rem; }
      .lab-kicker { color: #5eead4; letter-spacing: .14em; font-size: .8rem; font-weight: 600; margin-bottom: .65rem; }
      h1 { letter-spacing: -.035em; font-size: clamp(1.6rem, 3vw, 2.2rem) !important; }
      [data-testid="stMetricValue"] { color: #5eead4; }
      [data-testid="stMetric"] { background: #111b2e; padding: 1rem 1.2rem; border-radius: .7rem; }
      [data-testid="stCaptionContainer"] { color: #aebbd0; }
    </style>""")
    with st.sidebar:
        st.markdown("### Model settings")
        st.caption("µDOPPLER LAB · LOCAL WORKBENCH")
        st.divider()
        model_names = st.multiselect("Models", list(MODEL_FILES), default=list(MODEL_FILES))
        st.divider()
        st.markdown("**77 GHz · 17 kHz sweep rate**")
        st.caption("KTH / SAAB measured radar data. Predictions run on this computer.")
        st.caption("These models classify bird, drone, or human within the selected signal. Empty-scene detection and 60 GHz board performance are untested.")
        with st.expander("Experiment results"):
            st.markdown("**1,423 test segments · 25 held-out measurement rows**")
            if (REFINED_ROOT / "metrics.json").exists():
                report = json.loads((REFINED_ROOT / "metrics.json").read_text())
                for name, result in report["models"].items():
                    st.write(f"{name}: {result['test']['segments']['accuracy']:.2%} accuracy")
            else:
                st.write("Random Forest: 83.20% accuracy")
                st.write("CNN: 83.35% accuracy")
            st.caption("Full test scores include every evaluated test segment, regardless of the demo filter. Scores are uncalibrated.")

    st.html('<div class="lab-kicker">MEASURED RADAR / OFFLINE INFERENCE</div>')
    st.title("Radar sample classifier")
    st.caption("77 GHz measured echoes · bird / drone / human · local processing")
    source = st.radio("Sample source", ["Recorded samples", "Upload a sample"], horizontal=True)
    selection = model_selection()
    all_model_paths = tuple((name, row["path"]) for name, row in selection["models"].items())
    model_paths = tuple((name, path) for name, path in all_model_paths if name in model_names)
    if not model_paths:
        st.info("Select at least one model in the sidebar.")
        st.stop()
    missing = [name for name, path in model_paths if not Path(path).is_file()]
    if missing:
        st.warning(f"Train the KTH checkpoints before opening the dashboard. Missing: {', '.join(missing)}.")
        st.code('python src/kth77_dataset.py --input "<path-to-KTH-archive.npy>" --output-root experiments/kth77_first')
        st.stop()

    record = None
    if source == "Recorded samples":
        try:
            values, records = samples()
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Could not open the demo samples: {exc}")
            st.code('python src/dashboard_data.py --input "<path-to-KTH-archive.npy>"')
            st.stop()
        view = st.selectbox("Example collection", ["All test samples", "Correct demo examples", "Model mistakes / disagreements"])
        if view == "All test samples":
            available = range(len(records))
        else:
            with st.spinner("Preparing the sample collection…"):
                available = demo_positions(records, bank_scores(all_model_paths), view)
        if view == "Correct demo examples":
            st.caption(f"Curated demo · {len(available):,} / {len(records):,} samples: both models match the known label. This filter does not change full-test accuracy.")
        else:
            st.caption(f"{len(available):,} samples available · model outputs and original labels are preserved.")
        first, second = st.columns([1, 2])
        target = first.selectbox("Recorded target", ["drone", "bird", "human"], format_func=str.title)
        identities = list(dict.fromkeys(records[i]["measurement_id"] for i in available if records[i]["label"] == target))
        if not identities:
            st.info("No samples for this target in this collection. Choose another collection or target.")
            st.stop()
        identity = second.selectbox("Held-out recording", identities,
            format_func=lambda identity: f"{identity} · {next(r['source_label'] for r in records if r['measurement_id'] == identity)}")
        positions = [i for i in available if records[i]["measurement_id"] == identity]
        local_position = st.select_slider("Sample within recording", options=list(range(len(positions))),
            value=0, key=f"segment_{view}_{identity}",
            format_func=lambda i: f"Segment {records[positions[i]]['segment_index']}")
        position = positions[local_position]
        value, record = values[position], records[position]
        st.caption(f"Segment {record['segment_index']} · timestamp {record['time_s']:.3f} s · "
                   f"source range {record['range_m']:.1f} m · reference label: {record['label']}")
    else:
        st.info("Upload one complex, range-compressed 77 GHz sample at a 17 kHz sweep rate. You can download one from Recorded samples first.")
        uploaded = st.file_uploader("Drop a sample here", type=["npz", "npy"], max_upload_size=8,
            help="Recorded samples contains the segments used in the held-out evaluation.")
        with st.expander("Accepted sample formats"):
            st.markdown("**.npz:** a sample exported by this page, including signal settings.\n\n"
                "**.npy:** a complex array shaped `(150,)`, `(256,)`, `(5,256)` or `(1280,)`. "
                "Plain arrays assume 77 GHz and 17 kHz. For five range cells, the center cell is selected; "
                "256 sweeps are cropped to the central 150.\n\n"
                "The Infineon board's raw ADC frames need the separate BGT workflow. Object arrays and model files cannot be uploaded here.")
        if uploaded is None:
            st.caption("Waiting for a sample. Your file stays on this computer.")
            if (EXPERIMENT_ROOT / "dashboard_samples.npz").is_file() and all(Path(path).is_file() for _, path in all_model_paths):
                with st.expander("Need a sample? Download a recorded example", expanded=True):
                    show_example_downloads(all_model_paths)
            st.stop()
        try:
            value, metadata = read_upload(uploaded.getvalue(), uploaded.name)
            record = metadata.get("source_record")
            if metadata.get("settings_assumed"):
                st.caption("Using assumed settings: 77 GHz carrier, 17 kHz sweep rate. Confirm your sample uses these settings.")
            else:
                st.caption("Signal settings match the saved models.")
        except (ValueError, OSError) as exc:
            st.error(f"Cannot classify this file: {exc}")
            st.stop()

    try:
        with st.spinner("Reading the Doppler signature…"):
            extracted, predictions = classify(value, model_paths)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        st.error(f"Classification could not finish: {exc}")
        st.stop()

    st.divider()
    recommended = combined_prediction(predictions, selection["fusion_rf_weight"])
    st.metric("Prediction", recommended["prediction"].title())
    st.caption(f"{recommended['method']} · class score {recommended['scores'][recommended['prediction']]:.1%}")
    st.caption(" · ".join(f"{name}: {result['prediction'].title()} ({max(result['scores'].values()):.1%})"
        for name, result in predictions.items()))
    labels = {p["prediction"] for p in predictions.values()}
    if len(labels) > 1:
        st.info("Review suggested: the models disagree. The prediction above follows the validation-selected score combination.")
    if record and record.get("label"):
        correct = [name for name, p in predictions.items() if p["prediction"] == record["label"]]
        label_source = "Recorded label" if source == "Recorded samples" else "Reference label included in upload"
        st.caption(f"{label_source}: {str(record['label']).title()} · " +
            (f"Matched by {', '.join(correct)}." if correct else "No selected prediction matches this label."))

    st.markdown("**Class scores**")
    frame = pd.DataFrame({name: {label.title(): f"{score:.1%}" for label, score in result["scores"].items()}
                          for name, result in predictions.items()})
    st.table(frame)
    st.caption("Scores compare the three supported classes; they are uncalibrated. Full test accuracy is in Model settings.")
    st.markdown("**Inside the signal**")
    st.caption("150 complex sweeps → remove mean → normalize → 42 features (RF) / Doppler image (CNN)")
    fig = plots(extracted)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    first, second = st.columns(2)
    first.download_button("Download this sample", export_sample(value, record),
        file_name="radar_sample.npz", mime="application/octet-stream", width="stretch")
    payload = {"signal_domain": "kth77_range_compressed", "sample": record, "models": predictions,
        "recommended": recommended, "model_paths": dict(model_paths),
        "score_meaning": "uncalibrated class score", "features": dict(zip(
            FEATURE_NAMES, map(float, extracted["features"])))}
    second.download_button("Download prediction", json.dumps(payload, indent=2),
        file_name="radar_prediction.json", mime="application/json", width="stretch")
    with st.expander("Inspect the 42 numerical features"):
        st.dataframe(pd.DataFrame(payload["features"].items(), columns=["Feature", "Value"]), hide_index=True, width="stretch")
    with st.expander("Full test results · all 1,423 samples"):
        if (REFINED_ROOT / "metrics.json").exists():
            report = json.loads((REFINED_ROOT / "metrics.json").read_text())
            st.table(pd.DataFrame({name: {"Accuracy": f"{result['test']['segments']['accuracy']:.2%}",
                "Macro F1": f"{result['test']['segments']['macro_f1']:.3f}"}
                for name, result in report["models"].items()}).T)
            st.caption("Candidates and score weights were selected using validation data. The original test results were already inspected during development; new independent recordings are needed for a fresh final evaluation.")
        else:
            st.write("Random Forest: 83.20% · CNN: 83.35%")
        st.caption("Every evaluated test segment is included here. The correct-example filter affects only which demonstration samples you browse. These measurements do not validate the 60 GHz board or empty-scene detection.")


def show_example_downloads(model_paths):
    example_values, example_records = samples()
    correct = demo_positions(example_records, bank_scores(model_paths), "Correct demo examples")
    for column, label in zip(st.columns(3), ("drone", "bird", "human")):
        position = next((i for i in correct if example_records[i]["label"] == label), None)
        if position is not None:
            column.download_button(f"{label.title()} sample", export_sample(example_values[position], example_records[position]),
                file_name=f"{label}_sample.npz", mime="application/octet-stream", width="stretch")
    st.caption("Both models match the recorded label on these examples.")


if __name__ == "__main__":
    main()
