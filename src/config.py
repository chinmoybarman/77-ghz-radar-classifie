"""Validated acquisition metadata; no radar constants are hidden in DSP functions."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

C0 = 299_792_458.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "bgt60tr13c_config.json"


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _flatten(value):
    result = {}
    if isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                nested = _flatten(val)
                for nested_key, nested_val in nested.items():
                    if nested_key in result and result[nested_key] != nested_val:
                        raise ValueError(f"Ambiguous multi-profile config field: {nested_key}")
                    result[nested_key] = nested_val
            else:
                name = _key(key)
                if name in result and result[name] != val:
                    raise ValueError(f"Conflicting config field: {key}")
                result[name] = val
    elif isinstance(value, list):
        for val in value:
            for key, item in _flatten(val).items():
                if key in result and result[key] != item:
                    raise ValueError(f"Ambiguous multi-profile config field: {key}")
                result[key] = item
    return result


@dataclasses.dataclass(frozen=True)
class RadarConfig:
    start_frequency_Hz: float
    end_frequency_Hz: float
    sample_rate_Hz: float
    num_samples_per_chirp: int
    num_chirps_per_frame: int
    chirp_repetition_time_s: float
    frame_repetition_time_s: float
    rx_mask: int
    tx_mask: int = 1
    sensor: str = "BGT60TR13C"
    adc_bits: int = 12
    adc_format: str = "auto"
    data_format: int = 0
    array_layout: str = "frames_rx_chirps_samples"
    tx_power_level: int = 31
    if_gain_dB: int = 33
    hp_cutoff_Hz: int = 80000
    lp_cutoff_Hz: int = 500000
    ramp_duration_s: float | None = None
    range_min_m: float = 0.0
    range_max_m: float | None = None

    def __post_init__(self):
        numeric = [self.start_frequency_Hz, self.end_frequency_Hz, self.sample_rate_Hz,
                   self.chirp_repetition_time_s, self.frame_repetition_time_s]
        if not all(math.isfinite(x) and x > 0 for x in numeric):
            raise ValueError("Acquisition parameters must be positive and finite")
        if not (58e9 <= self.start_frequency_Hz < self.end_frequency_Hz <= 63.5e9):
            raise ValueError("Only BGT60TR13C up-chirps in 58–63.5 GHz are supported")
        for field in ("num_samples_per_chirp", "num_chirps_per_frame", "rx_mask", "tx_mask", "adc_bits", "data_format"):
            val = getattr(self, field)
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"{field} must be an integer")
        if self.num_samples_per_chirp < 8 or self.num_chirps_per_frame < 4:
            raise ValueError("At least 8 samples/chirp and 4 chirps/frame are required")
        if self.rx_mask != 7 or self.tx_mask != 1 or self.data_format != 0:
            raise ValueError("This pipeline requires real I-only data from TX1 and all three RX (rx_mask=7)")
        if "BGT60TR13C" not in self.sensor.upper():
            raise ValueError("Only BGT60TR13C recordings are supported")
        if self.adc_bits != 12 or self.sample_rate_Hz > 4e6:
            raise ValueError("Expected BGT60TR13C 12-bit ADC with sampling rate at most 4 MHz")
        # Legacy headers store times as float32. Allow its rounding error at
        # boundaries without permitting a materially shorter acquisition ramp.
        adc_duration = self.num_samples_per_chirp / self.sample_rate_Hz
        if self.ramp_duration_s is not None and (not math.isfinite(self.ramp_duration_s) or self.ramp_duration_s < adc_duration * (1 - 1e-6)):
            raise ValueError("Explicit ramp duration must cover the ADC sample window")
        if self.chirp_repetition_time_s < self.chirp_duration_s * (1 - 1e-6):
            raise ValueError("Chirp repetition time is shorter than the ramp duration")
        if self.frame_repetition_time_s < self.num_chirps_per_frame * self.chirp_repetition_time_s * (1 - 1e-6):
            raise ValueError("Frame repetition time is shorter than its chirp burst")
        if self.adc_format not in ("auto", "counts", "normalized_unsigned", "normalized_signed", "arbitrary"):
            raise ValueError("Unsupported adc_format")
        if not math.isfinite(self.range_min_m) or self.range_min_m < 0:
            raise ValueError("range_min_m must be finite and nonnegative")
        if self.range_max_m is not None and (not math.isfinite(self.range_max_m) or self.range_max_m <= self.range_min_m):
            raise ValueError("range_max_m must exceed range_min_m")

    @property
    def center_frequency_Hz(self):
        return (self.start_frequency_Hz + self.end_frequency_Hz) / 2

    @property
    def bandwidth_Hz(self):
        return self.end_frequency_Hz - self.start_frequency_Hz

    @property
    def chirp_duration_s(self):
        return self.ramp_duration_s or self.num_samples_per_chirp / self.sample_rate_Hz

    @property
    def slope_Hz_s(self):
        return self.bandwidth_Hz / self.chirp_duration_s

    @property
    def wavelength_m(self):
        return C0 / self.center_frequency_Hz

    @property
    def prf_Hz(self):
        return 1 / self.chirp_repetition_time_s

    @property
    def range_resolution_m(self):
        return C0 / (2 * self.bandwidth_Hz)

    def to_dict(self):
        return dataclasses.asdict(self)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload):
        # Capture sidecars contain both effective and requested profiles. Only
        # the effective acquisition profile determines sample interpretation.
        if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
            payload = payload["config"]
        flat = _flatten(payload)
        aliases = {
            "start_frequency_Hz": [("start_frequency_Hz", 1), ("lower_frequency_Hz", 1), ("Lower_RF_Frequency_kHz", 1000)],
            "end_frequency_Hz": [("end_frequency_Hz", 1), ("upper_frequency_Hz", 1), ("Upper_RF_Frequency_kHz", 1000)],
            "sample_rate_Hz": [("sample_rate_Hz", 1), ("Sampling_Frequency_kHz", 1000)],
            "num_samples_per_chirp": [("num_samples_per_chirp", 1), ("num_samples", 1), ("Samples_per_Chirp", 1)],
            "num_chirps_per_frame": [("num_chirps_per_frame", 1), ("num_chirps", 1), ("Chirps_per_Frame", 1)],
            "chirp_repetition_time_s": [("chirp_repetition_time_s", 1), ("Pulse_Repetition_Time_sec", 1)],
            "frame_repetition_time_s": [("frame_repetition_time_s", 1), ("Frame_Period_sec", 1)],
            "rx_mask": [("rx_mask", 1), ("Mask_Rx_Antennas", 1)],
            "tx_mask": [("tx_mask", 1), ("Mask_Tx_Antennas", 1)],
            "adc_bits": [("adc_bits", 1), ("ADC_Resolution_Bits", 1)],
            "data_format": [("data_format", 1), ("Data_Format_Enum", 1)],
            "ramp_duration_s": [("ramp_duration_s", 1), ("chirp_duration_s", 1), ("Chirp_Time_sec", 1)],
        }
        integers = {"num_samples_per_chirp", "num_chirps_per_frame", "rx_mask", "tx_mask", "adc_bits", "data_format", "tx_power_level", "if_gain_dB", "hp_cutoff_Hz", "lp_cutoff_Hz"}
        strings = {"sensor", "adc_format", "array_layout"}
        result = {}
        for f in dataclasses.fields(cls):
            candidates = []
            for name, scale in aliases.get(f.name, [(f.name, 1)]):
                value = flat.get(_key(name))
                if value is not None:
                    candidates.append(str(value) if f.name in strings else float(value) * scale)
            if candidates:
                value = candidates[0]
                if any(v != value for v in candidates[1:]):
                    raise ValueError(f"Conflicting aliases for {f.name}")
                if f.name in integers:
                    if not math.isfinite(value) or int(value) != value:
                        raise ValueError(f"{f.name} must be an integer")
                    value = int(value)
                result[f.name] = value
            elif f.default is dataclasses.MISSING:
                raise ValueError(f"Missing required acquisition parameter: {f.name}")
        return cls(**result)

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        if path.suffix.lower() == ".xml":
            root = ET.parse(path).getroot()
            values = {}
            for element in root.iter():
                key = element.attrib.get("name", element.tag.rsplit("}", 1)[-1])
                value = element.attrib.get("value", element.text)
                if value and value.strip() and len(element) == 0:
                    if key in values and values[key] != value.strip():
                        raise ValueError(f"Ambiguous XML parameter: {key}")
                    values[key] = value.strip()
            return cls.from_dict(values)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))

