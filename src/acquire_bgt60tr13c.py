"""Finite BGT60TR13C USB capture using Infineon's separately installed Radar SDK."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.config import DEFAULT_CONFIG, RadarConfig
from src.dataset_manager import CLASSES
from src.inspect_raw_recording import validate_raw_data


def _sequence_config(sequence, requested):
    """Read the SDK's simple frame-loop/chirp-loop/chirp representation."""
    frame_loop = sequence.loop
    chirp_loop = frame_loop.sub_sequence.contents.loop
    chirp = chirp_loop.sub_sequence.contents.chirp
    return replace(requested,
        start_frequency_Hz=float(chirp.start_frequency_Hz),
        end_frequency_Hz=float(chirp.end_frequency_Hz),
        sample_rate_Hz=float(chirp.sample_rate_Hz),
        num_samples_per_chirp=int(chirp.num_samples),
        num_chirps_per_frame=int(chirp_loop.num_repetitions),
        chirp_repetition_time_s=float(chirp_loop.repetition_time_s),
        frame_repetition_time_s=float(frame_loop.repetition_time_s),
        rx_mask=int(chirp.rx_mask), tx_mask=int(chirp.tx_mask))


def acquire(cfg, num_frames, output, session_id, label, overwrite=False):
    if num_frames < 1:
        raise ValueError("--num-frames must be positive")
    if not session_id.strip() or label not in CLASSES:
        raise ValueError("A session ID and a supported label are required")
    output = Path(output).resolve()
    if output.suffix.lower() != ".npy":
        raise ValueError("--output must end in .npy")
    companion = output.with_suffix(".json")
    if not overwrite and (output.exists() or companion.exists()):
        raise ValueError("Output already exists; select a new name or use --overwrite")
    try:
        from ifxradarsdk.fmcw import DeviceFmcw
        from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
    except (ImportError, OSError) as exc:
        raise RuntimeError("Infineon ifxradarsdk/native runtime is unavailable. Install the wheel matching your Python/OS from Infineon Radar Development Kit; offline processing does not need the board or SDK.") from exc
    sequence_config = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=cfg.frame_repetition_time_s,
        chirp_repetition_time_s=cfg.chirp_repetition_time_s,
        num_chirps=cfg.num_chirps_per_frame,
        tdm_mimo=False,
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=cfg.start_frequency_Hz, end_frequency_Hz=cfg.end_frequency_Hz,
            sample_rate_Hz=cfg.sample_rate_Hz, num_samples=cfg.num_samples_per_chirp,
            rx_mask=cfg.rx_mask, tx_mask=cfg.tx_mask, tx_power_level=cfg.tx_power_level,
            if_gain_dB=cfg.if_gain_dB, hp_cutoff_Hz=cfg.hp_cutoff_Hz, lp_cutoff_Hz=cfg.lp_cutoff_Hz))
    frames = []
    readback_verified = False
    readback_note = ""
    with DeviceFmcw() as device:
        sensor = str(device.get_sensor_type())
        if "BGT60TR13C" not in sensor.upper():
            raise ValueError(f"Connected sensor is {sensor!r}; only BGT60TR13C is supported")
        sequence = device.create_simple_sequence(sequence_config)
        device.set_acquisition_sequence(sequence)
        effective = cfg
        try:
            readback = device.get_acquisition_sequence()
            effective = _sequence_config(readback, cfg)
            readback_verified = True
        except (AttributeError, TypeError) as exc:
            readback_note = f"SDK sequence readback not available ({type(exc).__name__}); saved config is requested, not hardware-verified."
            print(f"Warning: {readback_note}", file=sys.stderr)
        expected = (3, effective.num_chirps_per_frame, effective.num_samples_per_chirp)
        for index in range(num_frames):
            batch = device.get_next_frame()
            if isinstance(batch, (tuple, list)):
                if len(batch) != 1:
                    raise ValueError("Expected one simple-sequence radar frame per SDK read")
                frame = np.asarray(batch[0])
            else:
                frame = np.asarray(batch)
                if frame.shape == (1, *expected):
                    frame = frame[0]
            if frame.shape != expected:
                raise ValueError(f"Unexpected SDK frame shape {frame.shape}; expected {expected}")
            frames.append(frame.copy())
            print(f"Captured {index + 1}/{num_frames}", flush=True)
    data = np.stack(frames)
    # Validate before writing, but retain the SDK's original samples and scaling.
    _, quality = validate_raw_data(data, effective)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, data, allow_pickle=False)
    metadata = {"config": effective.to_dict(), "requested_config": cfg.to_dict(),
                "is_synthetic": False, "recording_id": output.stem, "session_id": session_id,
                "label": label, "created_utc": datetime.now(timezone.utc).isoformat(),
                "sensor": sensor, "configuration_readback_verified": readback_verified,
                "readback_note": readback_note, "quality": quality,
                "frame_timestamps_s": (np.arange(num_frames)*effective.frame_repetition_time_s).tolist(),
                "timestamps_source": "nominal configured frame period; not independently measured hardware timestamps",
                "range_scale_note": "N/fs ramp-duration approximation unless ramp_duration_s is supplied from verified timing"}
    companion.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--label", choices=CLASSES, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = acquire(RadarConfig.from_file(args.config), args.num_frames, args.output, args.session_id, args.label, args.overwrite)
        print(f"Saved real hardware capture: {result}")
    except KeyboardInterrupt:
        parser.exit(130, "Capture interrupted; device disconnected.\n")
    except Exception as exc:
        # Vendor USB errors have SDK-version-specific exception classes.
        parser.exit(2, f"Capture failed: {exc}\nCheck the BGT60TR13C connection, close Radar Fusion GUI, and verify the SDK/USB driver.\n")


if __name__ == "__main__":
    main()

