"""Strict real-IF import and diagnostic CLI. Detection/track tables are not ADC data."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import sys
import warnings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.config import RadarConfig, PROJECT_ROOT

SYNTHETIC_WATERMARK = "[SYNTHETIC 60 GHz BENCHMARK - NOT REAL HARDWARE DATA]"
FORBIDDEN_FIELDS = {"range_m", "velocity_mps", "target_id", "azimuth_deg"}


@dataclass
class Recording:
    frames: np.ndarray
    config: RadarConfig
    metadata: dict
    quality: dict


def _is_synthetic(metadata):
    value = metadata.get("is_synthetic", False)
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError("is_synthetic must be true or false")


def _companion(path, explicit=None):
    if explicit and not Path(explicit).is_file():
        raise FileNotFoundError(explicit)
    candidates = [path.with_suffix(".json"), path.parent / "config.json", path.parent / "config", path.parent / "config.xml"] + ([Path(explicit)] if explicit else [])
    meta = {}
    config = None
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if candidate.suffix.lower() != ".xml":
            item = json.loads(candidate.read_text(encoding="utf-8-sig"))
            if not isinstance(item, dict):
                raise ValueError(f"Expected a JSON object: {candidate}")
            # File-specific provenance wins over directory-level metadata.
            meta = {**item, **meta}
        try:
            candidate_config = RadarConfig.from_file(candidate)
        except ValueError as exc:
            # A provenance-only sidecar is allowed; malformed acquisition
            # settings must never be replaced silently by another profile.
            if (explicit and candidate == Path(explicit)) or not str(exc).startswith("Missing required acquisition parameter"):
                raise
            continue
        if config is not None:
            _check_config_match(config, candidate_config)
        else:
            config = candidate_config
    return config, meta


def _guard_fields(names):
    if any(str(n).strip().lower() in FORBIDDEN_FIELDS for n in names):
        raise ValueError("Rule 8: processed target/detection logs are not raw ADC recordings")


def _reshape(data, cfg, layout=None):
    if data.dtype.names:
        _guard_fields(data.dtype.names)
        raise ValueError("Structured arrays are not supported as raw ADC samples")
    if data.dtype.kind not in "fiu":
        raise ValueError("Expected real numeric I-only samples; complex/object data are not supported")
    if data.ndim == 2 and data.shape[1] == 4:
        raise ValueError("Rule 8: four-column tracking tables are not raw ADC recordings")
    l, n = cfg.num_chirps_per_frame, cfg.num_samples_per_chirp
    layout = layout or cfg.array_layout
    if data.ndim == 4 and data.shape[1:] == (3, l, n):
        return data
    if data.ndim == 3:
        if data.shape == (3, l, n):
            return data[None, ...]
        if data.shape[1:] == (3 * l, n):
            return data.reshape(-1, 3, l, n)
        if data.shape[1:] == (l, n) and data.shape[0] % 3 == 0:
            return data.reshape(-1, 3, l, n)
        if data.shape[1:] == (3, l * n):
            return data.reshape(-1, 3, l, n)
    if data.ndim == 2 and data.shape[1] == n and data.shape[0] % (3 * l) == 0:
        return data.reshape(-1, 3, l, n)
    if data.ndim == 1 and data.size % (3 * l * n) == 0:
        if layout == "interleaved_rx":
            return data.reshape(-1, l, n, 3).transpose(0, 3, 1, 2)
        if layout == "chirps_rx_samples":
            return data.reshape(-1, l, 3, n).transpose(0, 2, 1, 3)
        if layout == "frames_rx_chirps_samples":
            return data.reshape(-1, 3, l, n)
    raise ValueError(f"Unsupported/ambiguous raw shape {data.shape}; expected (frames,3,{l},{n}) or documented receiver-major flattening")


def validate_raw_data(data, cfg, *, saturation_limit=0.01):
    """Return normalized data and per-channel quality; reject nonfinite/dead/clipped RX."""
    data = np.asarray(data)
    expected = (3, cfg.num_chirps_per_frame, cfg.num_samples_per_chirp)
    if data.ndim != 4 or data.shape[1:] != expected:
        raise ValueError(f"Expected ADC shape (frames,{expected[0]},{expected[1]},{expected[2]})")
    if data.dtype.kind not in "fiu":
        raise ValueError("ADC samples must be real numeric I-only values")
    if not np.isfinite(saturation_limit) or not 0 < saturation_limit <= 1:
        raise ValueError("saturation_limit must be in (0,1]")
    if not data.size or data.shape[0] == 0:
        raise ValueError("Recording contains no frames")
    if not np.isfinite(data).all():
        raise ValueError("Recording contains NaN or Inf")
    data = np.asarray(data, dtype=np.float64)
    adc_format = cfg.adc_format
    lo, hi = float(data.min()), float(data.max())
    if adc_format == "auto":
        if lo >= 0 and hi <= 1:
            adc_format = "normalized_unsigned"
        elif lo >= -1 and hi <= 1:
            adc_format = "normalized_signed"
        elif lo >= 0 and hi <= 2**cfg.adc_bits - 1 and np.allclose(data, np.rint(data), atol=1e-6, rtol=0):
            adc_format = "counts"
        else:
            raise ValueError("Cannot infer ADC scaling; declare adc_format explicitly (counts, normalized_signed, normalized_unsigned, arbitrary)")
    if adc_format == "counts":
        if not np.allclose(data, np.rint(data), atol=1e-6, rtol=0):
            raise ValueError("ADC counts must be integer-valued")
        rails = (0.0, float(2**cfg.adc_bits - 1))
    elif adc_format == "normalized_unsigned":
        rails = (0.0, 1.0)
    elif adc_format == "normalized_signed":
        rails = (-1.0, 1.0)
    else:
        rails = None
    fractions = np.zeros(3)
    frame_fractions = np.zeros(data.shape[:2])
    if rails is not None:
        if lo < rails[0] - 1e-7 or hi > rails[1] + 1e-7:
            raise ValueError(f"Samples are outside declared ADC rails {rails}")
        clipped = (data <= rails[0] + 1e-9) | (data >= rails[1] - 1e-9)
        fractions = clipped.mean(axis=(0, 2, 3))
        frame_fractions = clipped.mean(axis=(2, 3))
        if np.any(frame_fractions >= saturation_limit):
            raise ValueError(f"ADC saturation/clipping exceeds {saturation_limit:.1%} in at least one frame/RX")
        if np.any(fractions > 0):
            warnings.warn("Some ADC samples touch the rails; inspect quality metrics", RuntimeWarning)
    if adc_format == "counts":
        data = data / (2**cfg.adc_bits - 1)
    standard_deviations = data.std(axis=(0, 2, 3))
    if np.any(standard_deviations < 1e-7):
        raise ValueError(f"Dead/stuck RX channel: per-channel std={standard_deviations.tolist()}")
    # Detect a frozen channel in an individual frame as well.
    if np.any(data.std(axis=(2, 3)) < 1e-7):
        raise ValueError("Dead/stuck RX channel in at least one frame")
    return data, {"adc_format": adc_format, "rx_std": standard_deviations.tolist(), "rx_clipping_fraction": fractions.tolist(), "rx_max_frame_clipping_fraction": frame_fractions.max(axis=0).tolist(), "input_min": lo, "input_max": hi, "frames": len(data)}


def _recording_times(metadata, frame_count, cfg):
    """Keep supplied frame timestamps; generate explicitly nominal times otherwise."""
    if "frame_timestamps_s" not in metadata:
        metadata["timestamps_source"] = "nominal configured frame period; not measured timestamps"
        return np.arange(frame_count, dtype=float) * cfg.frame_repetition_time_s
    raw_times = np.asarray(metadata["frame_timestamps_s"])
    if raw_times.dtype.kind not in "fiu" or raw_times.shape != (frame_count,):
        raise ValueError("frame_timestamps_s must contain one numeric timestamp per frame")
    times = raw_times.astype(float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("frame_timestamps_s must be finite and strictly increasing")
    metadata.setdefault("timestamps_source", "provided recording timestamps; clock accuracy unverified")
    return times


_HEADER_FIELDS = [
    ("Num_Tx_Antennas", "i"), ("Num_Rx_Antennas", "i"),
    ("Mask_Tx_Antennas", "i"), ("Mask_Rx_Antennas", "i"),
    ("Are_Rx_Antennas_Interleaved", "i"), ("Modulation_Type_Enum", "i"),
    ("Chirp_Shape_Enum", "i"), ("Lower_RF_Frequency_kHz", "f"),
    ("Upper_RF_Frequency_kHz", "f"), ("Sampling_Frequency_kHz", "f"),
    ("ADC_Resolution_Bits", "i"), ("Are_ADC_Samples_Normalized", "i"),
    ("Data_Format_Enum", "i"), ("Chirps_per_Frame", "i"),
    ("Samples_per_Chirp", "i"), ("Samples_per_Frame", "i"),
    ("Chirp_Time_sec", "f"), ("Pulse_Repetition_Time_sec", "f"),
    ("Frame_Period_sec", "f"),
]


def _legacy_config(header):
    h = dict(header)
    if int(h.get("Num_Rx_Antennas", 3)) != 3 or int(h.get("Num_Tx_Antennas", 1)) != 1:
        raise ValueError("Legacy header must describe one TX and three RX")
    h["adc_format"] = "normalized_unsigned" if int(h.get("Are_ADC_Samples_Normalized", 0)) else "counts"
    h["array_layout"] = "interleaved_rx" if int(h.get("Are_Rx_Antennas_Interleaved", 0)) else "chirps_rx_samples"
    return RadarConfig.from_dict(h)


def _read_ifrb(path, metadata):
    raw = path.read_bytes()
    if len(raw) < 88 or raw[:4] not in (b"IFRB", b"BRFI"):
        raise ValueError("Invalid IFRB magic or truncated header")
    possible = [e for e in ("<", ">") if struct.unpack_from(e + "i", raw, 8)[0] in (88, 92)]
    if len(possible) != 1:
        raise ValueError("Unsupported IFRB header size/endian; supported declared sizes are 88 and 92 bytes")
    endian = possible[0]
    version, size = struct.unpack_from(endian + "fi", raw, 4)
    if not np.isfinite(version) or not (0 < version < 10):
        raise ValueError("Unsupported IFRB version")
    header = {name: struct.unpack_from(endian + dtype, raw, 12 + 4 * i)[0] for i, (name, dtype) in enumerate(_HEADER_FIELDS)}
    cfg = _legacy_config(header)
    count = 3 * cfg.num_chirps_per_frame * cfg.num_samples_per_chirp
    if header["Samples_per_Frame"] != count:
        raise ValueError("IFRB Samples_per_Frame does not match RX/chirp/sample dimensions")
    dtype_name = metadata.get("payload_dtype", "float32")
    if dtype_name not in ("float32", "uint16"):
        raise ValueError("IFRB payload_dtype must be float32 or explicitly declared uint16")
    dtype = np.dtype(endian + ("f4" if dtype_name == "float32" else "u2"))
    record_bytes = 4 + count * dtype.itemsize
    if (len(raw) - size) <= 0 or (len(raw) - size) % record_bytes:
        raise ValueError("Truncated/unsupported IFRB payload; expected uint32 frame counter + sample payload per frame")
    frames, counters = [], []
    for offset in range(size, len(raw), record_bytes):
        counters.append(struct.unpack_from(endian + "I", raw, offset)[0])
        values = np.frombuffer(raw, dtype=dtype, count=count, offset=offset + 4)
        frames.append(_reshape(values, cfg)[0])
    if len(counters) > 1 and np.any(np.diff(np.asarray(counters, dtype=np.int64)) != 1):
        raise ValueError("IFRB has missing/reordered frames; split into contiguous recordings before analysis")
    return np.stack(frames), cfg, {"legacy_header": header, "frame_numbers": counters, "ifrb_header_size": size}


def _read_text(path, cfg):
    text = path.read_text(encoding="utf-8-sig")
    for field in FORBIDDEN_FIELDS:
        if re.search(r"\b" + field + r"\b", text, flags=re.I):
            raise ValueError("Rule 8: processed detection fields found in text recording")
    header, values, frame_numbers = {}, [], []
    numeric_rows = []
    for line in text.splitlines():
        line = line.strip().lstrip("#%").strip()
        if not line:
            continue
        match = re.match(r"([A-Za-z_][A-Za-z_0-9 ]*)\s*[:=]\s*([-+0-9.eE]+)\s*$", line)
        if match:
            key, value = match.groups()
            if key.lower().replace(" ", "_") in ("frame_number", "frame"):
                frame_numbers.append(int(float(value)))
            else:
                header[key] = float(value)
            continue
        if re.search(r"(?:^|[\s,;])[-+]?(?:nan|inf(?:inity)?)(?:$|[\s,;])", line, flags=re.I):
            raise ValueError("Recording contains NaN or Inf")
        if re.match(r"^[A-Za-z]", line) or "{" in line:
            # Human-readable format legends are allowed only as comments/header.
            continue
        tokens = re.split(r"[\s,;]+", line)
        try:
            row = [float(x) for x in tokens if x]
        except ValueError as exc:
            raise ValueError(f"Malformed raw ASCII row: {line[:80]}") from exc
        numeric_rows.append(row)
    if numeric_rows and all(len(row) == 4 for row in numeric_rows):
        raise ValueError("Rule 8: four-column tracking tables are not raw ADC recordings")
    embedded = _legacy_config(header) if "Samples_per_Chirp" in header else None
    cfg = cfg or embedded
    if cfg is None:
        raise ValueError("Raw text requires acquisition config or a complete legacy header")
    count = 3 * cfg.num_chirps_per_frame * cfg.num_samples_per_chirp
    flat = np.asarray([v for row in numeric_rows for v in row], dtype=float)
    # Legacy text may place bare frame counters before each frame, rather than Frame_Number=.
    if not frame_numbers and header and flat.size % (count + 1) == 0:
        framed = flat.reshape(-1, count + 1)
        counters = framed[:, 0]
        if not np.all(counters == np.rint(counters)):
            raise ValueError("Invalid ASCII frame counters")
        frame_numbers = counters.astype(int).tolist()
        flat = framed[:, 1:].reshape(-1)
    if frame_numbers and (len(frame_numbers) != flat.size // count or any(b != a + 1 for a, b in zip(frame_numbers, frame_numbers[1:]))):
        raise ValueError("ASCII frame count/sequence mismatch")
    # Embedded header determines legacy RX ordering, independent of external config.
    layout = embedded.array_layout if embedded else cfg.array_layout
    return _reshape(flat, cfg, layout), embedded, {"legacy_header": header, "frame_numbers": frame_numbers}


def _check_config_match(a, b):
    for field in ("start_frequency_Hz", "end_frequency_Hz", "sample_rate_Hz", "num_samples_per_chirp", "num_chirps_per_frame", "chirp_repetition_time_s", "frame_repetition_time_s", "rx_mask", "tx_mask", "data_format", "slope_Hz_s"):
        if not np.isclose(getattr(a, field), getattr(b, field), rtol=1e-6, atol=1e-12):
            raise ValueError(f"Explicit and recorded config disagree on {field}")


def load_recording(path, cfg=None, config_path=None):
    """Load documented raw layouts, preserving provenance. No default config fallback."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    companion_cfg, metadata = _companion(path, config_path)
    if cfg is not None and companion_cfg is not None:
        _check_config_match(cfg, companion_cfg)
    cfg = cfg or companion_cfg
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            data = np.load(path, allow_pickle=False)
        except ValueError as exc:
            if "Object arrays cannot be loaded" in str(exc):
                raise ValueError("Object-array .npy is not raw BGT60TR13C ADC data. For the KTH archive use the dedicated adapter src/kth77_dataset.py; see reports/pre_board_guide.md.") from exc
            raise
        if data.dtype.names:
            _guard_fields(data.dtype.names)
        if cfg is None:
            raise ValueError("Missing acquisition metadata; provide --config or a companion config.json/.xml")
        frames = _reshape(data, cfg)
    elif suffix == ".bin":
        frames, embedded, extra = _read_ifrb(path, metadata)
        if cfg:
            _check_config_match(cfg, embedded)
        cfg = cfg or embedded
        metadata.update(extra)
    elif suffix in (".txt", ".csv"):
        frames, embedded, extra = _read_text(path, cfg)
        if cfg and embedded:
            _check_config_match(cfg, embedded)
        cfg = cfg or embedded
        metadata.update(extra)
    else:
        raise ValueError("Supported inputs: .npy, .raw.bin/.bin, .raw.txt/.txt")
    frames, quality = validate_raw_data(frames, cfg)
    metadata["frame_timestamps_s"] = _recording_times(metadata, len(frames), cfg).tolist()
    metadata["is_synthetic"] = _is_synthetic(metadata)
    metadata["source_path"] = str(path)
    return Recording(frames, cfg, metadata, quality)


def save_diagnostic(recording, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert
    from src.radar_dsp import compute_range_doppler_and_target_bin
    burst, cfg = recording.frames[0], recording.config
    result = compute_range_doppler_and_target_bin(burst, cfg)
    t_us = np.arange(cfg.num_samples_per_chirp) / cfg.sample_rate_Hz * 1e6
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for rx in range(3):
        axes[0, 0].plot(t_us, burst[rx, 0], label=f"RX{rx + 1}", alpha=.8)
    axes[0, 0].set(title="A · Real IF, first chirp", xlabel="Fast time (µs)", ylabel="ADC amplitude")
    axes[0, 0].legend()
    centered = burst[0, 0] - burst[0, 0].mean()
    axes[0, 1].plot(t_us, centered, label="DC-removed RX1")
    axes[0, 1].plot(t_us, np.abs(hilbert(centered)), label="Hilbert envelope")
    axes[0, 1].set(title="B · Beat signal and analytic envelope", xlabel="Fast time (µs)")
    axes[0, 1].legend()
    axes[1, 0].imshow(burst[0], aspect="auto", origin="lower", extent=[0, t_us[-1], 0, cfg.num_chirps_per_frame - 1])
    axes[1, 0].set(title="C · RX1 chirp matrix", xlabel="Fast time (µs)", ylabel="Chirp index")
    for rx in range(3):
        axes[1, 1].plot(result["range_axis_m"], 20*np.log10(np.maximum(np.abs(result["range_fft"][rx, 0]), 1e-12)), label=f"RX{rx + 1} single", alpha=.55)
        mti_profile = np.sqrt(np.mean(np.abs(result["range_fft_mti"][rx])**2, axis=0))
        axes[1, 1].plot(result["range_axis_m"], 20*np.log10(np.maximum(mti_profile, 1e-12)), linestyle="--", label=f"RX{rx + 1} MTI RMS")
    axes[1, 1].set(title="D · Range FFT and MTI residual", xlabel="Range (m)", ylabel="Amplitude (dB, arbitrary reference)")
    axes[1, 1].legend(fontsize=7, ncol=2)
    fig.suptitle(SYNTHETIC_WATERMARK if recording.metadata["is_synthetic"] else "BGT60TR13C raw-data diagnostic", fontsize=10)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "phase1_inspection")
    args = parser.parse_args(argv)
    try:
        recording = load_recording(args.input, config_path=args.config)
        output = save_diagnostic(recording, args.output_dir / f"{args.input.stem}_raw_diagnostic.png")
        print(json.dumps({"shape": list(recording.frames.shape), "quality": recording.quality, "is_synthetic": recording.metadata["is_synthetic"], "diagnostic": str(output)}, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Inspection failed: {exc}\n")


if __name__ == "__main__":
    main()

