"""Presentation components for the local radar workbench."""
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from src.kth77_dataset import CLASSES, PRF_HZ


def page_style():
    st.html("""
    <style>
      :root { --radar-accent: #62e5c2; }
      .stApp { background: #0b1018; color: #e6edf5; }
      [data-testid="stHeader"] { background: #0b1018; }
      [data-testid="stSidebar"] { background: #101722; border-right: 1px solid #233042; }
      .stMainBlockContainer { max-width: 1440px; padding: 4.3rem 2.5rem 2.5rem; }
      h1, h2, h3 { letter-spacing: -.035em; font-weight: 600 !important; }
      h1 { font-size: 2rem !important; padding-top: .1rem !important; }
      h3 { font-size: 1.15rem !important; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #9baec3; opacity: 1; }
      [data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #253246 !important; border-radius: 16px !important; }
      [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 600; color: #62e5c2; letter-spacing: -.045em; }
      [data-testid="stMetricLabel"] { color: #a8b8ca; }
      [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] { background: #131e2c; border-color: #2b3d52; }
      [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { border-color: #314358; border-radius: 9px; }
      button:focus-visible, input:focus-visible, [role="radio"]:focus-visible { outline: 2px solid #62e5c2 !important; outline-offset: 3px; }
      .st-key-source [role="radiogroup"], .st-key-detail [role="radiogroup"] { gap: .35rem; }
      .st-key-source [data-testid="stRadioOption"], .st-key-detail [data-testid="stRadioOption"] {
        border: 1px solid #2b3b50; border-radius: 8px; padding: .35rem .85rem; margin: 0;
      }
      .st-key-source [data-testid="stRadioOption"]:has(input:checked),
      .st-key-detail [data-testid="stRadioOption"]:has(input:checked) { background: #17332f; border-color: #62e5c2; }
      .lab-brand { display: flex; align-items: center; gap: 12px; margin: .4rem 0 1.8rem; }
      .lab-mark { height: 31px; width: 31px; border: 1px solid #62e5c2; border-radius: 50%;
        background: radial-gradient(circle, #62e5c2 0 2px, transparent 3px 8px, #2a6b5d 9px 10px, transparent 11px),
        conic-gradient(from 20deg, #62e5c233, transparent 100deg); }
      .lab-brand strong { font-size: 1rem; letter-spacing: .04em; }
      .lab-brand span { margin-left: auto; font-size: .85rem; color: #a3b3c6; }
      .lab-specs { display: flex; flex-wrap: wrap; gap: 1.25rem; padding: .8rem 0 1rem; color: #9baec3; font-size: .85rem; }
      .lab-specs b { color: #e4edf6; font-weight: 500; }
      .lab-status { color: #b5c4d5; font-size: .9rem; margin: .5rem 0 .9rem; }
      .lab-status.ok { color: #62e5c2; }
      .lab-status.review { color: #f2c77b; }
      .lab-score { display: grid; grid-template-columns: 55px 1fr 56px; gap: 12px; align-items: center;
        margin: .65rem 0; font-size: .9rem; color: #b9c8d9; }
      .lab-score .track { background: #202f43; height: 7px; border-radius: 8px; overflow: hidden; }
      .lab-score .fill { background: #4a6882; height: 100%; border-radius: 8px; }
      .lab-score.active { color: #e6edf5; }
      .lab-score.active .fill { background: #62e5c2; }
      .lab-score .value { text-align: right; font-variant-numeric: tabular-nums; }
      .lab-meta { display: flex; flex-wrap: wrap; gap: 1.4rem; padding: .75rem 0; }
      .lab-meta div { color: #9baec3; font-size: .82rem; }
      .lab-meta b { display: block; margin-top: .25rem; color: #e3edf8; font-size: .95rem; font-weight: 500; }
      .lab-footer { border-top: 1px solid #263345; margin-top: 2rem; padding-top: 1rem;
        display: flex; justify-content: space-between; flex-wrap: wrap; gap: .6rem; color: #8da2b9; font-size: .8rem; }
      @media (max-width: 760px) {
        .stMainBlockContainer { padding: 4rem 1rem 2rem; }
        .lab-brand { margin-bottom: 1rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stColumn"] { min-width: min(100%, 320px); flex: 1 1 320px; }
        h1 { font-size: 1.65rem !important; }
      }
    </style>
    <div class="lab-brand"><div class="lab-mark" aria-hidden="true"></div>
      <strong>µDOPPLER LAB</strong><span>Local radar analysis</span></div>
    """)


def metadata_row(items):
    cells = "".join(f"<div>{escape(label)}<b>{escape(str(value))}</b></div>" for label, value in items)
    st.html(f'<div class="lab-meta">{cells}</div>')


def scores_panel(recommended, record, source):
    first, second = st.columns([1, 1.4], gap="large")
    with first:
        st.metric("Prediction", recommended["prediction"].title())
        st.caption(recommended["method"])
        if record and record.get("label"):
            matches = record["label"] == recommended["prediction"]
            status = "Matches recorded label" if matches else "Differs from recorded label"
            st.html(f'<div class="lab-status {"ok" if matches else "review"}">{status}</div>')
            label_source = "Recorded label" if source == "Recorded samples" else "Reference label included in upload"
            st.caption(f"{label_source}: {record['label'].title()}")
    with second:
        st.markdown("**Class scores**")
        bars = []
        for label in CLASSES:
            score = recommended["scores"][label]
            active = "active" if label == recommended["prediction"] else ""
            bars.append(f'<div class="lab-score {active}"><span>{label.title()}</span>'
                    f'<div class="track" role="meter" aria-label="{label.title()} class score" '
                    f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{score * 100:.1f}">'
                    f'<div class="fill" style="width:{score * 100:.4f}%"></div></div>'
                    f'<span class="value">{score:.1%}</span></div>')
        st.html("".join(bars))
        st.caption("Scores are uncalibrated model outputs.")


def signal_chart(extracted, view):
    common = {"background": "transparent", "height": 245,
              "config": {"view": {"stroke": None},
                         "axis": {"labelColor": "#a4b6ca", "titleColor": "#b8c8db",
                                  "gridColor": "#263345", "domain": False, "tickColor": "#36465a",
                                  "labelFontSize": 12, "titleFontSize": 12, "titlePadding": 12},
                         "legend": {"labelColor": "#bfd0e2", "title": None}}}
    if view == "Waveform":
        values = extracted["signal"]
        data = pd.DataFrame({"Time (ms)": np.tile(np.arange(len(values)) / PRF_HZ * 1000, 2),
                             "Amplitude": np.r_[values.real, values.imag],
                             "Component": np.repeat(["I · real", "Q · imaginary"], len(values))})
        spec = {**common, "mark": {"type": "line", "strokeWidth": 1.5},
                "encoding": {"x": {"field": "Time (ms)", "type": "quantitative"},
                             "y": {"field": "Amplitude", "type": "quantitative"},
                             "color": {"field": "Component", "type": "nominal",
                                       "scale": {"range": ["#62e5c2", "#a8a7ff"]}},
                             "tooltip": [{"field": "Time (ms)", "format": ".3f"},
                                         {"field": "Amplitude", "format": ".3f"}, {"field": "Component"}]},
                "params": [{"name": "view", "select": "interval", "bind": "scales"}]}
    else:
        time = extracted["time_s"] * 1000
        freq = extracted["frequency_hz"] / 1000
        data = pd.DataFrame({"Time (ms)": np.tile(time, len(freq)),
                             "Doppler (kHz)": np.repeat(freq, len(time)),
                             "Power": extracted["spectrogram"].ravel()})
        dt = float(np.diff(time).mean()) if len(time) > 1 else 1.
        df = float(np.diff(freq).mean()) if len(freq) > 1 else 1.
        data["Time start"] = data["Time (ms)"] - dt / 2
        data["Time end"] = data["Time (ms)"] + dt / 2
        data["Frequency start"] = data["Doppler (kHz)"] - df / 2
        data["Frequency end"] = data["Doppler (kHz)"] + df / 2
        spec = {**common, "mark": "rect", "encoding": {
            "x": {"field": "Time start", "type": "quantitative", "title": "STFT window centre (ms)",
                  "scale": {"zero": False, "nice": False}},
            "x2": {"field": "Time end"},
            "y": {"field": "Frequency start", "type": "quantitative", "title": "Doppler (kHz)",
                  "scale": {"nice": False}},
            "y2": {"field": "Frequency end"},
            "color": {"field": "Power", "type": "quantitative", "scale": {"scheme": "magma", "domain": [0, 1]},
                      "legend": {"title": "Normalized power", "titleColor": "#a4b6ca", "orient": "right"}},
            "tooltip": [{"field": "Time (ms)", "format": ".3f"}, {"field": "Doppler (kHz)", "format": ".2f"},
                        {"field": "Power", "format": ".3f"}]}}
    st.vega_lite_chart(data, spec, width="stretch", theme=None, key=f"signal_{view}")
    st.caption("Mean removed and amplitude normalized." if view == "Waveform"
               else "Short-time Fourier transform of the selected scan. Color shows normalized power.")


def benchmark_panel(report, spectral_report):
    rows = []
    if report:
        for name, result in report["models"].items():
            metric = result["test"]["segments"]
            rows.append({"Model": name, "Accuracy": metric["accuracy"], "Macro F1": metric["macro_f1"], "History": "1 scan"})
    if spectral_report:
        for label, key in (("Spectral CNN", "single"), ("Spectral CNN · recent scans", "sequence")):
            metric = spectral_report["test"][key]["segments"]
            window = spectral_report["selection"][key]["window"]
            rows.append({"Model": label, "Accuracy": metric["accuracy"], "Macro F1": metric["macro_f1"],
                         "History": "1 scan" if window == 1 else f"Up to {window} scans"})
    if not rows:
        st.info("Run an evaluation to see benchmark results here.")
        return
    metric = spectral_report["test"]["single"]["segments"] if spectral_report else next(iter(report["models"].values()))["test"]["segments"]
    st.caption(f"{metric['n']:,} evaluated samples · complete measurement rows held out from training")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Accuracy": st.column_config.NumberColumn(format="percent"),
                                "Macro F1": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Model selection used validation recordings. This test split has been inspected during development. New recordings are needed for an independent estimate.")
    st.caption("These results cover bird, drone and human targets in KTH 77 GHz recordings. Background detection and transfer to the 60 GHz board require separate evaluation.")


def footer():
    st.html('<div class="lab-footer"><span>KTH / SAAB · 77 GHz recorded echoes</span>'
            '<span>Processing and sample uploads stay on this computer</span></div>')
