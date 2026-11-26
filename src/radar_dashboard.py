"""Local radar workbench. Run with streamlit run src/radar_dashboard.py."""
from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_data import (EXPERIMENT_ROOT, REFINED_ROOT, combined_prediction,
    demo_positions, export_sample, model_selection, read_upload_with_history)
from src.dashboard_runtime import (SPECTRAL_DIR, classify, classify_recorded,
    collection_scores, replay, samples, spectral_prediction)
from src.dashboard_ui import benchmark_panel, footer, metadata_row, page_style, scores_panel, signal_chart
from src.kth77_dataset import CLASSES, FEATURE_NAMES

MODEL_NAMES = ("Random Forest", "CNN")


def read_report(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def move_sample(key, step, count):
    st.session_state[key] = min(count - 1, max(0, st.session_state.get(key, 0) + step))


def recorded_input(all_model_paths, choice):
    values, records = samples()
    st.subheader("Recording")
    target = st.selectbox("Recorded target", ["drone", "bird", "human"], format_func=str.title)
    view = st.selectbox("Example collection", ["All test samples", "Correct demo examples", "Model mistakes / disagreements"],
                        help="Curated examples require every component model to match the recorded label.")
    if view == "All test samples":
        available = range(len(records))
    else:
        with st.spinner("Finding examples…"):
            available = demo_positions(records, collection_scores(all_model_paths, choice), view)
    if view == "Correct demo examples":
        st.caption(f"Curated demo · {len(available):,} / {len(records):,} samples. Every model matches the recorded label.")
    identities = list(dict.fromkeys(records[i]["measurement_id"] for i in available if records[i]["label"] == target))
    if not identities:
        st.info("No matching samples. Choose another collection or target.")
        return None
    labels = {r["measurement_id"]: r["source_label"] for r in records}
    identity = st.selectbox("Held-out recording", identities,
        format_func=lambda item: f"{labels[item]} · {item.removeprefix('kth_measurement_')}")
    positions = [i for i in available if records[i]["measurement_id"] == identity]
    key = f"segment_{view}_{identity}"
    if st.session_state.get(key, 0) >= len(positions):
        st.session_state[key] = 0
    if len(positions) > 1:
        local = st.select_slider("Sample within recording", options=list(range(len(positions))), value=0, key=key,
            format_func=lambda i: f"Segment {records[positions[i]]['segment_index']}")
    else:
        local = 0
        st.caption("One sample in this collection.")
    previous, following = st.columns(2)
    previous.button("Previous", width="stretch", disabled=local == 0,
                    on_click=move_sample, args=(key, -1, len(positions)))
    following.button("Next", width="stretch", disabled=local == len(positions)-1,
                     on_click=move_sample, args=(key, 1, len(positions)))
    st.caption(f"Sample {local + 1} of {len(positions)} in this recording")
    position = positions[local]
    record = records[position]
    st.divider()
    metadata_row([("Range", f"{record['range_m']:.1f} m"), ("Timestamp", f"{record['time_s']:.3f} s"),
                  ("Segment", record["segment_index"])])
    return values[position], record, position


def upload_input():
    st.subheader("Your sample")
    st.caption("Complex 77 GHz radar echoes, sampled at a 17 kHz sweep rate.")
    uploaded = st.file_uploader("Drop a sample here", type=["npz", "npy"], max_upload_size=8,
                               help="Choose an exported sample or a compatible numeric array. Maximum 8 MB.")
    with st.expander("Accepted sample formats"):
        st.markdown("**Exported .npz** includes the signal and radar settings. Sequence exports also include "
                    "complex `history` shaped `(N,150)` and increasing `times_s`, ending at the current scan.")
        st.markdown("**Numeric .npy** accepts complex arrays shaped `(150,)`, `(256,)`, "
                    "`(5,256)` or `(1280,)`. The center range cell and central 150 sweeps are selected.")
        st.caption("Plain arrays assume a 77 GHz carrier and 17 kHz sweep rate. Use valid scans of one target per sequence.")
    return uploaded


def show_example_downloads(model_paths, choice):
    values, records = samples()
    correct = demo_positions(records, collection_scores(model_paths, choice), "Correct demo examples")
    for column, label in zip(st.columns(3), CLASSES):
        position = next((i for i in correct if records[i]["label"] == label), None)
        if position is not None:
            history = None
            if choice and choice["window"] > 1:
                signals, counts, ends, times = replay()
                section = slice(ends[position]-counts[position], ends[position])
                history = signals[section], times[section]
            column.download_button(f"{label.title()} sample", export_sample(values[position], records[position], history),
                file_name=f"{label}_sample.npz", mime="application/octet-stream", width="stretch")
    st.caption("Each model matches the recorded label on these example downloads.")


def inference(value, position, model_paths, choice, mode, upload_history):
    extracted, predictions = classify(value, model_paths) if position is None else classify_recorded(position, model_paths)
    selection = model_selection()
    recommended = combined_prediction(predictions, selection["fusion_rf_weight"])
    used_history = None
    if choice and choice["spectral_weight"] > 0:
        from src.kth77_spectral import blend_scores, recent_history
        echoes = value[None]
        if mode == "sequence":
            if position is not None:
                signals, counts, ends, times = replay()
                section = slice(ends[position]-counts[position], ends[position])
                echoes, times = signals[section], times[section]
            else:
                echoes, times = recent_history(*upload_history, choice["window"])
            used_history = echoes, times
        spectral = spectral_prediction(echoes)
        result = dict(zip(CLASSES, map(float, spectral)))
        predictions["Spectral CNN"] = {"prediction": max(result, key=result.get), "scores": result}
        baseline = np.array([recommended["scores"][label] for label in CLASSES])
        mixed = blend_scores(baseline, spectral, choice["spectral_weight"])
        recommended = {"method": "Spectral CNN" if choice["spectral_weight"] == 1 else "Spectral blend",
                       "prediction": CLASSES[int(mixed.argmax())], "scores": dict(zip(CLASSES, map(float, mixed)))}
    return extracted, predictions, recommended, used_history


def main():
    st.set_page_config(page_title="µDoppler Lab · Radar workbench", page_icon="📡", layout="wide",
                       initial_sidebar_state="collapsed")
    page_style()
    st.title("Radar workbench")
    st.caption("Explore a recorded echo. Identify its signature.")
    st.html('<div class="lab-specs"><span><b>77 GHz</b> carrier</span><span><b>17 kHz</b> sweep rate</span>'
            '<span><b>Bird · Drone · Human</b> classes</span></div>')
    report = read_report(REFINED_ROOT / "metrics.json")
    spectral_report = None
    if REFINED_ROOT.name == "kth77_fusion" and (SPECTRAL_DIR / "spectral.pt").is_file():
        completed = read_report(SPECTRAL_DIR / "metrics.json")
        if completed and {"single", "sequence"} <= completed.get("test", {}).keys():
            spectral_report = completed
    with st.sidebar:
        st.subheader("Model settings")
        model_names = st.multiselect("Models", list(MODEL_NAMES), default=list(MODEL_NAMES),
                                    help="Both comparison models enable the spectral modes. Choose one to inspect it alone.")
        st.caption("The classification mode chooses the final prediction. Individual model scores appear under Model comparison.")
        st.divider()
        st.markdown("**About this dataset**")
        st.caption("KTH / SAAB recorded 77 GHz echoes. Each sample contains a known target.")
        st.caption("The 60 GHz board requires its own capture and evaluation workflow.")
    source = st.radio("Sample source", ["Recorded samples", "Upload a sample"], horizontal=True,
                       key="source", label_visibility="collapsed")
    selection = model_selection()
    all_model_paths = tuple((name, row["path"]) for name, row in selection["models"].items())
    model_paths = tuple((name, path) for name, path in all_model_paths if name in model_names)
    if not model_paths:
        st.info("Select at least one model in Model settings.")
        return
    missing = [name for name, path in model_paths if not Path(path).is_file()]
    if missing:
        st.warning(f"Model files are missing for {', '.join(missing)}.")
        with st.expander("Set up this checkout"):
            st.markdown("Follow the local setup commands in the project README to generate models and samples.")
        return
    controls, output = st.columns([1, 2.3], gap="large")
    mode, choice = "single", None
    value = record = position = upload_history = None
    with controls, st.container(border=True, key="recording_panel"):
        if spectral_report and set(model_names) == set(MODEL_NAMES):
            modes = ["Single scan"]
            if spectral_report["selection"]["sequence"]["window"] > 1 and (
                    source == "Upload a sample" or (SPECTRAL_DIR / "replay.npz").is_file()):
                modes.append("Recent scans")
            modes.append("Original RF + CNN")
            decision = st.radio("Classification mode", modes, key="decision")
            if decision != "Original RF + CNN":
                mode = "sequence" if decision == "Recent scans" else "single"
                choice = spectral_report["selection"][mode]
            st.caption("Uses the current scan and available previous scans." if mode == "sequence"
                       else "Classifies the selected scan independently.")
            st.divider()
        try:
            if source == "Recorded samples":
                selected = recorded_input(all_model_paths, choice)
                if selected is None:
                    return
                value, record, position = selected
            else:
                uploaded = upload_input()
                if uploaded is not None:
                    value, metadata, upload_history = read_upload_with_history(uploaded.getvalue(), uploaded.name)
                    record = metadata.get("source_record")
                    if mode == "sequence" and upload_history is None:
                        mode, choice = "single", spectral_report["selection"]["single"]
                        st.caption("This file contains one scan. Using the single-scan model.")
                    st.caption("Using assumed settings of 77 GHz and 17 kHz." if metadata.get("settings_assumed")
                               else "Signal settings match the saved models.")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            output.error(f"Could not open this sample. {exc}")
            return
    with output:
        if value is None:
            with st.container(border=True):
                st.subheader("Bring your signal into focus")
                st.write("Upload a sample to see its predicted class, Doppler signature and model scores.")
                st.caption("Start with an example below, or download a sequence from Recorded samples.")
                if (EXPERIMENT_ROOT / "dashboard_samples.npz").is_file():
                    try:
                        example_choice = choice
                        if mode == "sequence" and not (SPECTRAL_DIR / "replay.npz").is_file():
                            example_choice = spectral_report["selection"]["single"]
                        show_example_downloads(all_model_paths, example_choice)
                    except (OSError, ValueError, RuntimeError, KeyError) as exc:
                        st.info(f"Example downloads are unavailable. {exc}")
            footer()
            return
        try:
            with st.spinner("Reading the Doppler signature…"):
                extracted, predictions, recommended, history = inference(
                    value, position, model_paths, choice, mode, upload_history)
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            st.error(f"Classification could not finish. {exc}")
            return
        used_scans, span = (len(history[0]), float(history[1][-1]-history[1][0])) if history else (1, 0.)
        with st.container(border=True, key="prediction_panel"):
            scores_panel(recommended, record, source)
            if mode == "sequence":
                st.caption(f"Used {used_scans} of up to {choice['window']} consecutive scans spanning {span:.3f} s. History restarts after a gap.")
        detail = st.radio("Analysis view", ["Signal", "Model comparison", "Benchmark"], horizontal=True,
                           key="detail", label_visibility="collapsed")
        with st.container(border=True, key="analysis_panel"):
            if detail == "Signal":
                chart = st.selectbox("Signal view", ["Micro-Doppler", "Waveform"], label_visibility="collapsed")
                signal_chart(extracted, chart)
            elif detail == "Model comparison":
                st.subheader("Model comparison")
                frame = pd.DataFrame({name: {"Prediction": p["prediction"].title(),
                    **{label.title(): f"{score:.1%}" for label, score in p["scores"].items()}}
                    for name, p in predictions.items()}).T
                st.dataframe(frame, width="stretch")
                if len({p["prediction"] for p in predictions.values()}) > 1:
                    st.info("The models disagree. The main prediction uses the validation-selected model and score weights.")
                st.caption("Class scores compare the three supported targets and are uncalibrated.")
                with st.expander("Numerical features"):
                    st.dataframe(pd.DataFrame({"Feature": FEATURE_NAMES, "Value": extracted["features"]}),
                                 hide_index=True, width="stretch")
            else:
                st.subheader("Evaluation on held-out recordings")
                benchmark_panel(report, spectral_report)
        payload = {"signal_domain": "kth77_range_compressed", "sample": record, "models": predictions,
                   "recommended": recommended, "model_paths": dict(model_paths), "scans_used": used_scans,
                   "span_s": span, "score_meaning": "uncalibrated class score",
                   "features": dict(zip(FEATURE_NAMES, map(float, extracted["features"])))}
        if choice:
            payload["spectral_settings"] = choice
            if choice["spectral_weight"] > 0:
                payload["model_paths"]["Spectral CNN"] = str(SPECTRAL_DIR / "spectral.pt")
        stem = f"{record['measurement_id']}_{record['segment_index']}" if position is not None else "radar_sample"
        download, prediction = st.columns(2)
        download.download_button("Download these scans" if history else "Download this sample",
            export_sample(value, record, history), file_name=f"{stem}.npz", mime="application/octet-stream",
            width="stretch", on_click="ignore")
        prediction.download_button("Export prediction", json.dumps(payload, indent=2),
            file_name=f"{stem}_prediction.json", mime="application/json", width="stretch", on_click="ignore")
    footer()


if __name__ == "__main__":
    main()
