"""Dataset registry, independent-session splits, and explicitly synthetic fixtures."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.config import C0, DEFAULT_CONFIG, PROJECT_ROOT, RadarConfig

CLASSES = ("drone", "bird", "other", "background")
REGISTRY_FIELDS = ("recording_id", "filepath", "config_path", "label", "session_id", "is_synthetic", "subject_id", "environment", "notes")


def _boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid is_synthetic value: {value!r}")


def read_registry(path, allow_synthetic=False):
    """Validate paths, labels, identity, and provenance; never split a recording into sessions."""
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(REGISTRY_FIELDS[:6])
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Registry missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Registry is empty; acquire real data or explicitly generate synthetic fixtures")
    identities, sources = set(), set()
    for row in rows:
        for field in REGISTRY_FIELDS[:6]:
            if not str(row.get(field, "")).strip():
                raise ValueError(f"Empty registry field {field}")
        if row["label"] not in CLASSES:
            raise ValueError(f"Invalid target label {row['label']!r}; expected {CLASSES}")
        if row["recording_id"] in identities:
            raise ValueError(f"Duplicate recording_id: {row['recording_id']}")
        identities.add(row["recording_id"])
        for field in ("filepath", "config_path"):
            candidate = Path(row[field])
            candidate = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
            if not candidate.is_file():
                raise ValueError(f"Missing {field}: {candidate}")
            row[field] = str(candidate)
        if row["filepath"] in sources:
            raise ValueError("One recording file cannot appear in multiple rows/sessions")
        sources.add(row["filepath"])
        row["is_synthetic"] = _boolean(row["is_synthetic"])
        config = RadarConfig.from_file(row["config_path"])
        sidecar = Path(row["filepath"]).with_suffix(".json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if "is_synthetic" in metadata and _boolean(metadata["is_synthetic"]) != row["is_synthetic"]:
                raise ValueError("Registry and recording disagree on synthetic provenance")
            for field in ("label", "session_id", "recording_id"):
                if field in metadata and metadata[field] != row[field]:
                    raise ValueError(f"Registry and recording disagree on {field}")
        if row["is_synthetic"] and not allow_synthetic:
            raise ValueError("Synthetic rows require --allow-synthetic; synthetic/mixed metrics are not real-data validation")
    return rows


def split_sessions(rows, seed=42, train_fraction=0.6, val_fraction=0.2):
    """Split whole sessions and require class coverage in every partition.

    The pure-class-session case is stratified directly. Mixed-class sessions use a
    bounded seeded group search; impossible protocols fail rather than leak rows.
    """
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Train/validation fractions must leave a positive test fraction")
    if not rows:
        raise ValueError("Cannot split an empty registry")
    groups = {}
    for row in rows:
        groups.setdefault(row["session_id"], []).append(row)
    labels = {row["label"] for row in rows}
    for label in labels:
        if sum(any(r["label"] == label for r in group) for group in groups.values()) < 3:
            raise ValueError(f"At least three independent sessions containing {label!r} are required for train/val/test")
    rng = np.random.default_rng(seed)
    partition = {"train": [], "val": [], "test": []}
    if all(len({r["label"] for r in group}) == 1 for group in groups.values()):
        for label in sorted(labels):
            ids = sorted(s for s, group in groups.items() if group[0]["label"] == label)
            rng.shuffle(ids)
            nt = min(max(1, int(len(ids) * train_fraction)), len(ids) - 2)
            nv = min(max(1, int(len(ids) * val_fraction)), len(ids) - nt - 1)
            for name, selected in zip(partition, (ids[:nt], ids[nt:nt+nv], ids[nt+nv:])):
                partition[name].extend(r for s in selected for r in groups[s])
    else:
        ids = sorted(groups)
        nt = min(max(1, int(len(ids) * train_fraction)), len(ids) - 2)
        nv = min(max(1, int(len(ids) * val_fraction)), len(ids) - nt - 1)
        for _ in range(2000):
            rng.shuffle(ids)
            candidate = dict(zip(partition, (ids[:nt], ids[nt:nt+nv], ids[nt+nv:])))
            if all({r["label"] for s in selected for r in groups[s]} == labels for selected in candidate.values()):
                partition = {name: [r for s in selected for r in groups[s]] for name, selected in candidate.items()}
                break
        else:
            raise ValueError("Cannot find a three-way session split with class coverage; collect more independent sessions")
    sets = [{r["session_id"] for r in part} for part in partition.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "Session leakage"
    return partition


def passafiume_displacement(t, amplitudes_m, angular_frequencies_rad_s, phases_rad):
    """Eq(9–11)-inspired four-component vibration trajectory, per motor."""
    a, w, p = map(np.asarray, (amplitudes_m, angular_frequencies_rad_s, phases_rad))
    if a.shape != w.shape or a.shape != p.shape or a.shape[-1] != 4:
        raise ValueError("Each motor needs four amplitudes/frequencies/phases")
    return np.sum(a[..., None, None] * np.cos(w[..., None, None] * t + p[..., None, None]), axis=-3)


def passafiume_literal_equation11(t, cfg, range_m, amplitudes_m, angular_frequencies_rad_s, phases_rad, motor_weights):
    """Literal requested Eq11 transcription; retained for traceability, not range calibration.

    Its 4*mu/c beat coefficient differs from the conventional monostatic 2*mu/c
    coefficient. The dataset uses the displacement with the carrier-aware Peter
    phase model instead. This function is not presented as a validated ADC model.
    """
    displacement = passafiume_displacement(np.asarray(t), amplitudes_m, angular_frequencies_rad_s, phases_rad)
    weights = np.asarray(motor_weights)[:, None, None]
    return np.sum(weights * np.cos(2*np.pi * (4*cfg.slope_Hz_s/C0) * (range_m + displacement) * t), axis=0)


def peter_rotor_return(fast_time, absolute_time, cfg, *, range_m, velocity_mps, blade_length_m, rpm, blades=2, elevation_rad=0.0, initial_phase_rad=0.0, body_amplitude=1.0, blade_amplitude=1.0):
    """Peter/Reddy Eq4–7 analytic IF; sinc(x)=sin(x)/x, hence np.sinc(x/pi).

    This ideal scatterer model omits RF/ADC transfer functions and uses f_start
    as the zero-fast-time carrier. Taking its real part gives I-only samples.
    """
    if blades < 1 or blade_length_m <= 0 or rpm < 0:
        raise ValueError("Rotor geometry and RPM must be nonnegative with at least one blade")
    phase_scale = 4*np.pi * (cfg.start_frequency_Hz + cfg.slope_Hz_s * fast_time) / C0
    distance = range_m + velocity_mps * absolute_time
    signal = body_amplitude * np.exp(1j * phase_scale * distance)
    omega = rpm * 2*np.pi/60
    for blade in range(blades):
        angle = omega * absolute_time + initial_phase_rad + 2*np.pi*blade/blades
        projection = blade_length_m/2 * np.cos(elevation_rad) * np.cos(angle)
        x = phase_scale * projection
        signal = signal + blade_length_m * blade_amplitude * np.exp(1j*phase_scale*(distance+projection)) * np.sinc(x/np.pi)
    return signal


def _synthesize_recording(cfg, label, frames, rng):
    n, l = cfg.num_samples_per_chirp, cfg.num_chirps_per_frame
    fast = np.arange(n)[None, :] / cfg.sample_rate_Hz
    slow = np.arange(l)[:, None] * cfg.chirp_repetition_time_s
    beat_nyquist_range = cfg.sample_rate_Hz / 2 * C0 / (2*cfg.slope_Hz_s)
    range_low = max(0.5, cfg.range_min_m)
    range_high = min(cfg.range_max_m or beat_nyquist_range*.8, beat_nyquist_range*.8, 3.2)
    if range_high <= range_low:
        raise ValueError("Synthetic generator needs a nonempty 0.5m+ usable range window")
    distance = rng.uniform(range_low, range_high)
    velocity = rng.uniform(-.2, .2)
    azimuth, elevation = rng.uniform(-.45, .45, 2)
    # Infineon AN600: RX1-RX3 horizontal, RX2-RX3 vertical, physical spacing2.5mm.
    phase = 2*np.pi * .0025 / cfg.wavelength_m * np.array([np.sin(azimuth), np.sin(elevation), 0.0])
    motors = int(rng.choice([1, 4]))
    amplitudes = rng.uniform(.67e-3, 4.76e-3, (motors, 4))
    wb = rng.uniform(1.2e3, 5.0e3, motors)
    wa = rng.uniform(.35e3, 1.0e3, motors)
    frequencies = np.stack([wb, wa, wb+wa, np.abs(wb-wa)], axis=1)
    phases = rng.uniform(0, 2*np.pi, (motors, 4))
    rpm = rng.uniform(6000, 14000, motors)
    blade_lengths = rng.uniform(.04, .08, motors)
    noise_sigma = rng.uniform(.025, .08)
    gain = rng.uniform(.2, .4)
    flap_hz = rng.uniform(3, 12)
    phase_scale = 4*np.pi*(cfg.start_frequency_Hz + cfg.slope_Hz_s*fast)/C0
    clutter_ranges = rng.uniform(range_low, range_high, 3)
    clutter_weights = rng.uniform(.05, .25, 3)
    result = np.empty((frames, 3, l, n), dtype=np.float32)
    for f in range(frames):
        absolute = f*cfg.frame_repetition_time_s + slow + fast
        moving_range = distance + velocity * absolute
        if label == "drone":
            vibrations = passafiume_displacement(absolute, amplitudes, frequencies, phases)
            analytic = np.zeros((l, n), dtype=complex)
            for motor in range(motors):
                analytic += np.exp(1j*phase_scale*(moving_range + vibrations[motor])) / motors
                analytic += peter_rotor_return(fast, absolute, cfg, range_m=distance,
                    velocity_mps=velocity, blade_length_m=blade_lengths[motor], rpm=rpm[motor],
                    initial_phase_rad=phases[motor, 0], body_amplitude=0,
                    blade_amplitude=8/motors, elevation_rad=elevation)
        elif label == "bird":
            # Deliberately simple flapping surrogate, not a validated bird model.
            wing = .02*np.sin(2*np.pi*flap_hz*absolute)
            analytic = .55*np.exp(1j*phase_scale*moving_range) + .45*np.exp(1j*phase_scale*(moving_range+wing))
        elif label == "other":
            analytic = np.exp(1j*phase_scale*moving_range)
        else:
            analytic = np.zeros((l, n), dtype=complex)
        clutter = sum(a*np.exp(1j*phase_scale*r) for a, r in zip(clutter_weights, clutter_ranges))
        for rx in range(3):
            # Small independent receiver gain/noise; all nuisance ranges shared across labels.
            result[f, rx] = gain * np.real(analytic*np.exp(1j*phase[rx]) + clutter) + rng.normal(0, noise_sigma, (l, n))
    return result, {"range_m": distance, "radial_velocity_mps": velocity, "motor_count": motors if label == "drone" else 0,
                    "rpm": rpm.tolist() if label == "drone" else [], "vibration_amplitudes_m": amplitudes.tolist() if label == "drone" else [],
                    "generator": "carrier-aware Peter rotor + Passafiume displacement; non-drone heuristic surrogates",
                    "limitations": "Ideal scatterers; no measured RCS, ADC transfer function, validated bird model or demonstrated hardware accuracy."}


def generate_synthetic_pretrain_dataset(output_root, cfg=None, sessions_per_class=5, frames_per_recording=6, seed=42):
    if sessions_per_class < 3 or frames_per_recording < 4:
        raise ValueError("Benchmark needs >=3 sessions per class and >=4 frames per recording")
    root = Path(output_root).resolve()
    cfg = replace(cfg or RadarConfig.from_file(DEFAULT_CONFIG), adc_format="arbitrary")
    directory = root / "data" / "raw" / "synthetic_pretrain"
    directory.mkdir(parents=True, exist_ok=True)
    registry = root / "data" / "metadata" / "synthetic_recordings.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    children = np.random.SeedSequence(seed).spawn(len(CLASSES)*sessions_per_class)
    for ci, label in enumerate(CLASSES):
        for session in range(sessions_per_class):
            index = ci*sessions_per_class+session
            rng = np.random.default_rng(children[index])
            recording_id = f"synthetic_{seed}_{label}_{session:03d}"
            path = directory / f"{recording_id}.npy"
            data, physics = _synthesize_recording(cfg, label, frames_per_recording, rng)
            np.save(path, data, allow_pickle=False)
            metadata = {"config": cfg.to_dict(), "is_synthetic": True, "recording_id": recording_id,
                        "session_id": recording_id, "label": label, "seed": seed, "seed_spawn_key": list(children[index].spawn_key),
                        "frame_timestamps_s": (np.arange(frames_per_recording)*cfg.frame_repetition_time_s).tolist(), "simulation": physics}
            companion = path.with_suffix(".json")
            companion.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
            rows.append({"recording_id": recording_id, "filepath": os.path.relpath(path, registry.parent),
                         "config_path": os.path.relpath(companion, registry.parent), "label": label,
                         "session_id": recording_id, "is_synthetic": "True", "subject_id": f"simulated_{label}_{session:03d}",
                         "environment": "idealized_simulation", "notes": "SYNTHETIC ONLY; each recording independently randomized; non-drone classes are surrogates"})
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sessions-per-class", type=int, default=5)
    parser.add_argument("--frames-per-recording", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.generate_synthetic:
            registry = generate_synthetic_pretrain_dataset(args.output_root, RadarConfig.from_file(args.config), args.sessions_per_class, args.frames_per_recording, args.seed)
            print(f"SYNTHETIC ONLY registry: {registry}")
        elif args.registry:
            rows = read_registry(args.registry, args.allow_synthetic)
            parts = split_sessions(rows, args.seed)
            print(json.dumps({name: {"recordings": len(part), "sessions": len({r['session_id'] for r in part})} for name, part in parts.items()}, indent=2))
        else:
            parser.error("Choose --generate-synthetic or --registry PATH")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Dataset error: {exc}\n")


if __name__ == "__main__":
    main()

