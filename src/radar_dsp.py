"""Configuration-driven processing for real-IF, three-receiver BGT60TR13C data.

The feature names and their order form a versioned model input contract.  This is
an explicit adaptation of Oh et al. (2019) to a different radar and four classes;
it does not claim to reproduce that paper's accuracy.  See the research report
for source links, implementation conventions, and the FTMDSE duty-cycle caveat.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import fft, signal
from scipy.interpolate import CubicSpline

C0 = 299_792_458.0
FEATURE_VERSION = "bgt60tr13c-46-v1"
_INTRA_NAMES = (
    "zero_crossings", "normalized_energy", "std", "max_mean_difference",
    "log_energy_entropy", "spectral_energy", "max_spectral_energy",
    "frequency_peak_distance",
)
_INTER_NAMES = (
    "zero_crossing_ratio", "energy_ratio", "spectral_energy_ratio",
    "max_amplitude_ratio", "concatenated_log_energy_entropy",
)
EMD_FEATURE_NAMES = [f"emd_{name}_{k}" for name in _INTRA_NAMES for k in range(1, 4)]
EMD_FEATURE_NAMES += [f"emd_{name}_{k}_{k+1}" for name in _INTER_NAMES for k in range(1, 3)]
PHYSICAL_FEATURE_NAMES = [
    "spec_peak_doppler_hz", "spec_centroid_hz", "spec_bandwidth_hz", "spec_entropy",
    "signal_energy", "doppler_papr_db", "temporal_variation",
    "cadence_periodicity_ratio", "estimated_range_m", "range_cell_snr_db",
    "aoa_azimuth_phase_std_rad", "aoa_elevation_phase_std_rad",
]
FEATURE_NAMES = EMD_FEATURE_NAMES + PHYSICAL_FEATURE_NAMES
_EPS = np.finfo(np.float64).eps


def _validate_burst(raw_burst: np.ndarray, cfg: Any) -> np.ndarray:
    data = np.asarray(raw_burst)
    expected = (3, int(cfg.num_chirps_per_frame), int(cfg.num_samples_per_chirp))
    if data.shape != expected:
        raise ValueError(f"Expected real IF burst {expected}, received {data.shape}")
    if np.iscomplexobj(data):
        raise ValueError("BGT60TR13C ADC data must be real IF, not complex IQ or a processed FFT")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise ValueError("Burst must contain only finite numeric ADC samples")
    if expected[1] < 4 or expected[2] < 8:
        raise ValueError("At least four chirps and eight fast-time samples are required")
    return data.astype(np.float64, copy=False)


def normalize_complex_burst(values: np.ndarray) -> np.ndarray:
    """Oh Eq. (6), independently per receiver; a zero input stays zero."""
    x = np.asarray(values, dtype=np.complex128)
    scale = np.mean(np.abs(x), axis=-1, keepdims=True)
    return np.divide(x, scale, out=np.zeros_like(x), where=scale > _EPS)


def _circular_phase_std(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    product = a * np.conjugate(b)
    magnitude = np.abs(product)
    valid = magnitude > max(float(np.max(magnitude)) * 1e-8, _EPS)
    if np.count_nonzero(valid) < 2:
        return 0.0, False
    resultant = np.abs(np.mean(product[valid] / magnitude[valid]))
    # Circular standard deviation avoids wrap-around at +/-pi.  Unweighted
    # valid samples are used; no angle calibration is implied by this statistic.
    return float(np.sqrt(-2.0 * np.log(np.clip(resultant, _EPS, 1.0)))), True


def pruning_flags(rv_db: np.ndarray, target_bin: int) -> dict[str, Any]:
    """Return diagnostic predicates; do not map a four-class task to non-UAV.

    Oh's Rule 1 prose says 'strongest' yet subsequently says satisfying either
    rule rejects a UAV.  Both the literal and inverse interpretations are
    exposed, so this ambiguity never silently overrides a model prediction.
    Rule 2's fixed rows and threshold are used literally only for 64 chirps.
    """
    db = np.asarray(rv_db, dtype=float)
    low, high = float(np.min(db)), float(np.max(db))
    normalized = (db - low) / (high - low) if high > low else np.zeros_like(db)
    l_count, r_count = normalized.shape
    left, right = max(0, target_bin - 3), min(r_count, target_bin + 4)
    peak_velocity = int(np.argmax(normalized[:, target_bin]))
    is_local_max = bool(normalized[peak_velocity, target_bin] >=
                        np.max(normalized[peak_velocity, left:right]))
    supported = l_count == 64
    edge_low = bool(np.max(normalized[:15, target_bin]) < 0.45 and
                    np.max(normalized[49:64, target_bin]) < 0.45) if supported else False
    return {
        "target_is_local_max": is_local_max,
        "rule1_literal_satisfied": is_local_max,
        "rule1_candidate_rejected": not is_local_max,
        "rule2_low_edge_spread": edge_low,
        "paper_rules_supported": supported,
        "empty_signal": high == low,
        "recommended_prune": bool(not is_local_max or edge_low or high == low),
        "automatic_class_override": False,
    }


def compute_range_doppler_and_target_bin(
    raw_burst: np.ndarray, cfg: Any, apply_mti: bool = True,
    target_bin: int | None = None,
) -> dict[str, Any]:
    """Compute complex range bins, MTI, and noncoherently fused Doppler power.

    Positive slow-time phase rotation gives positive Doppler here; the physical
    approaching/receding sign must be calibrated against the acquired mixer
    convention. Power averaging lowers variance of independent receiver noise,
    but does not guarantee the prompt's asserted 4.77-dB SNR improvement.
    """
    data = _validate_burst(raw_burst, cfg)
    _, l_count, n_samples = data.shape
    centered = data - np.mean(data, axis=-1, keepdims=True)
    window_fast = signal.windows.hann(n_samples, sym=False)
    range_fft = fft.rfft(centered * window_fast, axis=-1)
    range_fft_mti = range_fft - np.mean(range_fft, axis=1, keepdims=True)
    selected = range_fft_mti if apply_mti else range_fft
    window_slow = signal.windows.hann(l_count, sym=False)
    doppler_fft = fft.fftshift(fft.fft(selected * window_slow[None, :, None], axis=1), axes=1)
    rv_power = np.mean(np.abs(doppler_fft) ** 2, axis=0)
    rv_db = 10.0 * np.log10(np.maximum(rv_power, np.finfo(float).tiny))
    frequencies = fft.rfftfreq(n_samples, d=1.0 / float(cfg.sample_rate_Hz))
    range_axis = frequencies * C0 / (2.0 * float(cfg.slope_Hz_s))
    doppler_axis = fft.fftshift(fft.fftfreq(l_count, d=float(cfg.chirp_repetition_time_s)))
    range_power = np.sum(rv_power, axis=0)
    range_min = float(getattr(cfg, "range_min_m", 0.0))
    range_max = getattr(cfg, "range_max_m", None)
    range_gate = (range_axis >= range_min)
    if range_max is not None:
        range_gate &= range_axis <= float(range_max)
    range_gate[0] = False
    candidates = np.flatnonzero(range_gate)
    if candidates.size == 0:
        raise ValueError("Configured range gate contains no positive FFT range bins")
    if target_bin is None:
        # DC is an invalid range estimate after fast-time mean removal. The
        # strongest remaining cell is a candidate, not a CFAR-confirmed target.
        target_bin = int(candidates[np.argmax(range_power[candidates])])
    if not isinstance(target_bin, (int, np.integer)) or not 1 <= target_bin < len(frequencies):
        raise ValueError("target_bin must be a positive rFFT range-bin index")
    if not range_gate[target_bin]:
        raise ValueError("Requested target_bin lies outside the configured range gate")
    target_raw = selected[:, :, target_bin]
    target_signals = normalize_complex_burst(target_raw)
    # AN600 BGT60TR13C Shield: RX1--RX3 is horizontal; RX2--RX3 vertical.
    # RX1--RX2 in the supplied prompt is the diagonal, not pure azimuth.
    az_std, az_valid = _circular_phase_std(target_raw[0], target_raw[2])
    el_std, el_valid = _circular_phase_std(target_raw[1], target_raw[2])
    flags = pruning_flags(rv_db, int(target_bin))
    flags["empty_signal"] = bool(np.max(range_power[range_gate]) <= _EPS)
    flags["recommended_prune"] = bool(flags["recommended_prune"] or flags["empty_signal"])
    return {
        "range_fft": range_fft,
        "range_fft_mti": range_fft_mti,
        "doppler_fft": doppler_fft,
        "rv_power": rv_power,
        "rv_db": rv_db,
        "range_axis_m": range_axis,
        "range_gate_mask": range_gate,
        "doppler_axis_hz": doppler_axis,
        "velocity_axis_mps": doppler_axis * float(cfg.wavelength_m) / 2.0,
        "target_bin": int(target_bin),
        "target_signals": target_signals,
        "target_signals_raw": target_raw,
        "target_range_m": float(range_axis[target_bin]),
        "target_power": float(range_power[target_bin]),
        "pruning": flags,
        "aoa_azimuth_phase_std_rad": az_std,
        "aoa_elevation_phase_std_rad": el_std,
        "aoa_phase_valid": bool(az_valid and el_valid),
        "antenna_pairs": {"azimuth": [1, 3], "elevation": [2, 3]},
        "target_detection_method": "strongest_non_dc_range_power_candidate_no_CFAR",
        "apply_mti": bool(apply_mti),
    }


def interpolate_chirp_idle_time(chirps: np.ndarray, cfg: Any) -> dict[str, Any]:
    """Insert only the missing ADC timestamps, preserving every acquired sample.

    If Q=round((CRI-N/fs)*fs), the last measured sample and the following
    chirp's first sample are Q+1 sampling intervals apart. The inserted weights
    are therefore 1/(Q+1), ..., Q/(Q+1). This corrects the duplicate endpoint
    produced by the n=0 convention in the prompt's transcription of Eq. (19).
    No idle segment is invented after the last chirp, or between frames.
    """
    x = np.asarray(chirps)
    if x.ndim != 3 or min(x.shape) < 1 or not np.issubdtype(x.dtype, np.number):
        raise ValueError("chirps must have receiver, chirp, fast-time axes")
    if not np.isfinite(x).all():
        raise ValueError("chirps must contain only finite values")
    x = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64, copy=False)
    rx_count, l_count, n_samples = x.shape
    fs = float(cfg.sample_rate_Hz)
    period_samples_float = float(cfg.chirp_repetition_time_s) * fs
    period_samples = int(round(period_samples_float))
    # IFRB stores timing as float32; tolerate its representation error only.
    if not np.isclose(period_samples_float, period_samples, rtol=1e-7, atol=1e-6):
        raise ValueError("CRI * ADC sample rate must be integral for uniform-grid interpolation")
    q_count = period_samples - n_samples
    if q_count < 0:
        raise ValueError("Chirp repetition interval is shorter than the sampled chirp")
    total = l_count * n_samples + (l_count - 1) * q_count
    joined = np.empty((rx_count, total), dtype=x.dtype)
    observed = np.zeros(total, dtype=bool)
    for index in range(l_count):
        start = index * period_samples
        joined[:, start:start+n_samples] = x[:, index]
        observed[start:start+n_samples] = True
        if index + 1 < l_count and q_count:
            fraction = np.arange(1, q_count + 1, dtype=float) / (q_count + 1)
            last = x[:, index, -1, None]
            following = x[:, index + 1, 0, None]
            joined[:, start+n_samples:start+period_samples] = last + (following - last) * fraction
    return {"signal": joined, "time_s": np.arange(total) / fs,
            "observed_sample_mask": observed, "idle_samples": q_count,
            "sampling_rate_hz": fs}


def extract_fast_time_interpolated_signal(
    raw_burst: np.ndarray, cfg: Any, target_bin: int | None = None,
    sigma_range_hz: float | None = None, demodulate: bool = False,
) -> dict[str, Any]:
    """Gaussian 2-D RD filtering and timestamp-correct idle interpolation.

    The Doppler filter is all-pass. Conjugate range mirrors are both retained
    for real IF. The default range sigma is three FFT bins, an explicit analysis
    parameter rather than a hardware claim; increase it to include the expected
    rotor spread. Optional demodulation removes the detected range carrier from
    an analytic representation, with the sign consistent with this FFT.

    This implements the requested Eq. (11)--(19) front end, not the paper's
    subsequent complex-EMD/IMF-selection algorithm. Linear interpolation does
    not recover unobserved high-frequency motion, especially at low duty cycle.
    """
    data = _validate_burst(raw_burst, cfg)
    _, _, n_samples = data.shape
    if target_bin is None:
        target_bin = compute_range_doppler_and_target_bin(data, cfg)["target_bin"]
    if not isinstance(target_bin, (int, np.integer)) or not 1 <= target_bin <= n_samples // 2:
        raise ValueError("target_bin lies outside the positive real-IF range spectrum")
    fs = float(cfg.sample_rate_Hz)
    center_hz = float(target_bin) * fs / n_samples
    sigma = 3.0 * fs / n_samples if sigma_range_hz is None else float(sigma_range_hz)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_range_hz must be finite and positive")
    frequency = fft.fftfreq(n_samples, d=1.0/fs)
    # Circular distance also handles filters touching the Nyquist boundary.
    distance_plus = np.abs((frequency - center_hz + fs/2) % fs - fs/2)
    distance_minus = np.abs((frequency + center_hz + fs/2) % fs - fs/2)
    gaussian = np.maximum(np.exp(-0.5 * (distance_plus/sigma)**2),
                          np.exp(-0.5 * (distance_minus/sigma)**2))
    centered = data - np.mean(data, axis=-1, keepdims=True)
    rd = fft.fft2(centered, axes=(-2, -1))
    filtered_complex = fft.ifft2(rd * gaussian[None, None, :], axes=(-2, -1))
    filtered = filtered_complex.real
    chirps_to_join = filtered
    if demodulate:
        analytic = signal.hilbert(filtered, axis=-1)
        fast_time = np.arange(n_samples) / fs
        chirps_to_join = analytic * np.exp(-2j * np.pi * center_hz * fast_time)[None, None, :]
    result = interpolate_chirp_idle_time(chirps_to_join, cfg)
    duty_cycle = n_samples / (fs * float(cfg.chirp_repetition_time_s))
    result.update({
        "filtered_chirps": filtered,
        "range_filter": gaussian,
        "target_bin": int(target_bin),
        "carrier_frequency_hz": center_hz,
        "sigma_range_hz": sigma,
        "doppler_filter": "all_pass",
        "demodulated": bool(demodulate),
        "duty_cycle": float(duty_cycle),
        "unaliased_recovery_guaranteed": False,
        "limitations": [
            "Linear fill is fabricated interpolation, not measured rotor motion.",
            "Low duty cycle and range-filter bandwidth can distort the signature.",
            "No inter-frame gap is filled and no complex-EMD denoising is applied.",
        ],
    })
    return result


def _spline_envelope(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Mirror two extrema at each boundary to reduce spline endpoint drift."""
    points = peaks.astype(float)
    left = -points[:2][::-1]
    right = 2 * (x.size - 1) - points[-2:][::-1]
    knots = np.concatenate((left, points, right))
    values = np.concatenate((x[peaks[:2]][::-1], x[peaks], x[peaks[-2:]][::-1]))
    unique, positions = np.unique(knots, return_index=True)
    return CubicSpline(unique, values[positions], bc_type="natural")(np.arange(x.size))


def emd_decompose(
    values: np.ndarray, max_imfs: int = 3, max_sift_iterations: int = 10,
    return_residual: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Cubic-spline, fixed-iteration sifting; missing oscillatory modes are zero.

    At most ten sifts by default, as in Oh's reported setting. Endpoint mirroring
    and zero padding are reproducible implementation choices. A monotone trend
    is retained in the residual rather than mislabeled as an IMF.
    """
    x = np.asarray(values)
    if x.ndim != 1 or np.iscomplexobj(x) or x.size < 4 or not np.isfinite(x).all():
        raise ValueError("EMD requires a finite real vector of at least four samples")
    if max_imfs < 1 or max_sift_iterations < 1:
        raise ValueError("IMF count and sifting iteration limit must be positive")
    residual = x.astype(float, copy=True)
    imfs = np.zeros((max_imfs, x.size), dtype=float)
    for imf_index in range(max_imfs):
        maxima, minima = signal.find_peaks(residual)[0], signal.find_peaks(-residual)[0]
        if min(len(maxima), len(minima)) < 2:
            break
        candidate = residual.copy()
        extracted = False
        for _ in range(max_sift_iterations):
            maxima, minima = signal.find_peaks(candidate)[0], signal.find_peaks(-candidate)[0]
            if min(len(maxima), len(minima)) < 2:
                break
            mean_envelope = (_spline_envelope(candidate, maxima) +
                             _spline_envelope(candidate, minima)) / 2.0
            candidate -= mean_envelope
            extracted = True
        if not extracted:
            break
        imfs[imf_index] = candidate
        residual -= candidate
    return (imfs, residual) if return_residual else imfs


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    n, d = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    return np.divide(n, d, out=np.zeros(np.broadcast_shapes(n.shape, d.shape)), where=np.abs(d) > _EPS)


def log_energy_entropy(values: np.ndarray, bins: int = 16) -> float:
    """Oh Eq. (12) using an explicit empirical amplitude-PMF convention.

    The paper does not specify PMF estimation. Here equal-width amplitude bins
    assign each sample its bin's empirical mass; zero-mass bins are never logged.
    This is the stated negative sum of squared log2 masses, not Shannon entropy.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.ptp(x) <= _EPS:
        return 0.0
    counts, edges = np.histogram(x, bins=min(bins, x.size))
    indexes = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(counts)-1)
    probability = counts[indexes] / x.size
    return float(-np.sum(np.log2(probability) ** 2))


def features_from_imfs(imfs: np.ndarray) -> np.ndarray:
    """The 13 ordered feature groups (13K-5) from explicitly supplied IMFs.

    Eq. (8)'s sign is binary 1[x>=0], avoiding np.sign's factor-of-two error.
    Group 6 uses integrated FFT power and group 7 its largest bin, resolving the
    source's scalar/vector F notation. Group 8 uses a centered FFT. Standard
    deviation uses ddof=1 (MATLAB default). The robust maximum is percentile 95.
    """
    modes = np.asarray(imfs, dtype=float)
    if modes.ndim != 2 or modes.shape[0] < 1 or modes.shape[1] < 2 or not np.isfinite(modes).all():
        raise ValueError("imfs must be a finite K-by-samples matrix")
    k_count, dimension = modes.shape
    crossings = np.sum(np.abs(np.diff((modes >= 0).astype(float), axis=1)), axis=1)
    energies = np.sum(modes**2, axis=1)
    normalized_energy = _safe_ratio(energies, np.sum(energies))
    deviations = np.std(modes, axis=1, ddof=1)
    robust_maxima = np.percentile(modes, 95, axis=1)
    positive_means = np.array([np.mean(m[m > 0]) if np.any(m > 0) else 0.0 for m in modes])
    differences = _safe_ratio(robust_maxima - positive_means, robust_maxima)
    entropy = np.array([log_energy_entropy(m) for m in modes])
    spectra = np.abs(fft.fftshift(fft.fft(modes, axis=1), axes=1)) ** 2
    spectral_energy = np.sum(spectra, axis=1)
    spectral_maxima = np.max(spectra, axis=1)
    peak_distance = np.abs(np.argmax(spectra, axis=1) - dimension//2) * 2.0 / dimension
    peak_distance[energies <= _EPS] = 0.0
    maxima = np.max(modes, axis=1)
    groups = [crossings, normalized_energy, deviations, differences, entropy,
              spectral_energy, spectral_maxima, peak_distance,
              _safe_ratio(crossings[1:], crossings[:-1]),
              _safe_ratio(energies[1:], energies[:-1]),
              _safe_ratio(spectral_energy[1:], spectral_energy[:-1]),
              _safe_ratio(maxima[1:], maxima[:-1]),
              np.array([log_energy_entropy(np.concatenate((modes[i], modes[i+1])))
                        for i in range(k_count-1)])]
    features = np.concatenate(groups)
    if features.shape != (13 * k_count - 5,) or not np.isfinite(features).all():
        raise FloatingPointError("EMD feature calculation produced invalid values")
    return features


def extract_oh2019_emd_features(complex_signal: np.ndarray) -> np.ndarray:
    """Normalize then expand [magnitude, real, imaginary] and extract 34 values."""
    x = np.asarray(complex_signal, dtype=np.complex128)
    if x.ndim != 1 or x.size < 4 or not np.isfinite(x).all():
        raise ValueError("Expected one finite complex slow-time vector")
    normalized = normalize_complex_burst(x)
    expanded = np.concatenate((np.abs(normalized), normalized.real, normalized.imag))
    return features_from_imfs(emd_decompose(expanded, max_imfs=3, max_sift_iterations=10))


def _resize_matrix(values: np.ndarray, output_size: int = 32) -> np.ndarray:
    """Separable interpolation to a fixed feature grid, with exact output size."""
    rows, columns = values.shape
    old_x = np.linspace(0, 1, columns)
    new_x = np.linspace(0, 1, output_size)
    horizontal = np.vstack([np.interp(new_x, old_x, row) for row in values])
    old_y = np.linspace(0, 1, rows)
    new_y = np.linspace(0, 1, output_size)
    return np.column_stack([np.interp(new_y, old_y, horizontal[:, j]) for j in range(output_size)])


def standardized_spectrogram(
    signals: np.ndarray, sampling_rate_hz: float, output_size: int = 32,
    nperseg: int | None = None, center_frequency_hz: float | None = None,
    half_bandwidth_hz: float | None = None, return_axes: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average receiver STFT powers then map a fixed 80-dB display range to [0,1].

    Each frame is processed independently; frame gaps are never mistaken for
    samples at the chirp repetition rate. Resizing creates a model input image,
    not extra physical time/frequency resolution.
    """
    x = np.asarray(signals)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("Spectrogram input must be receiver-by-time and finite")
    if sampling_rate_hz <= 0 or output_size < 2:
        raise ValueError("Positive sampling rate and output_size >=2 are required")
    segment = min(x.shape[1], 32 if nperseg is None else int(nperseg))
    if segment < 4:
        raise ValueError("STFT segment must have at least four samples")
    nfft = max(64, int(2 ** np.ceil(np.log2(segment))))
    freq, times, stft = signal.stft(x, fs=sampling_rate_hz, window="hann", nperseg=segment,
                               noverlap=segment*3//4, nfft=nfft, return_onesided=False,
                               boundary=None, padded=False, axis=-1)
    power = fft.fftshift(np.mean(np.abs(stft)**2, axis=0), axes=0)
    freq = fft.fftshift(freq)
    if center_frequency_hz is not None and half_bandwidth_hz is not None:
        band = np.abs(freq - center_frequency_hz) <= half_bandwidth_hz
        if np.count_nonzero(band) >= 2:
            power = power[band]
            freq = freq[band]
    frequency_axis = np.linspace(freq[0], freq[-1], output_size)
    time_axis = np.linspace(times[0], times[-1], output_size)
    maximum = float(np.max(power))
    if maximum <= np.finfo(float).tiny:
        result = np.zeros((output_size, output_size), dtype=np.float32)
    else:
        relative_db = 10.0 * np.log10(np.maximum(power / maximum, 1e-8))
        result = _resize_matrix(np.clip((relative_db + 80.0)/80.0, 0.0, 1.0), output_size)
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return (result, frequency_axis, time_axis) if return_axes else result


def _physical_features(dsp: dict[str, Any]) -> np.ndarray:
    index = dsp["target_bin"]
    power = dsp["rv_power"][:, index]
    frequencies = dsp["doppler_axis_hz"]
    total = float(np.sum(power))
    if total > _EPS:
        probability = power / total
        peak = float(frequencies[np.argmax(power)])
        centroid = float(np.sum(frequencies * probability))
        width = float(np.sqrt(np.sum((frequencies-centroid)**2 * probability)))
        nonzero = probability[probability > 0]
        entropy = float(-np.sum(nonzero*np.log(nonzero)) / np.log(len(power)))
        papr = float(10*np.log10(np.max(power)/np.mean(power)))
    else:
        peak = centroid = width = entropy = papr = 0.0
    raw = dsp["target_signals_raw"]
    energy = float(np.mean(np.sum(np.abs(raw)**2, axis=1)))
    envelope = np.mean(np.abs(raw), axis=0)
    mean_amplitude = float(np.mean(envelope))
    variation = float(np.std(envelope)/mean_amplitude) if mean_amplitude > _EPS else 0.0
    envelope_ac = envelope - mean_amplitude
    autocorrelation = signal.correlate(envelope_ac, envelope_ac, mode="full")[len(envelope)-1:]
    cadence = 0.0
    if autocorrelation[0] > _EPS and len(envelope) >= 6:
        cadence = float(np.clip(np.max(autocorrelation[2:len(envelope)//2+1]) /
                                autocorrelation[0], 0.0, 1.0))
    range_power = np.sum(dsp["rv_power"], axis=0)
    background_mask = np.ones(len(range_power), dtype=bool)
    background_mask[max(0,index-3):min(len(range_power),index+4)] = False
    background_mask[0] = False
    noise = float(np.median(range_power[background_mask])) if background_mask.any() else 0.0
    snr = float(10*np.log10((total+_EPS)/(noise+_EPS))) if total > _EPS else 0.0
    return np.array([peak, centroid, width, entropy, energy, papr, variation, cadence,
                     dsp["target_range_m"], snr, dsp["aoa_azimuth_phase_std_rad"],
                     dsp["aoa_elevation_phase_std_rad"]], dtype=np.float64)


def extract_burst_features(
    raw_burst: np.ndarray, cfg: Any, apply_mti: bool = True,
    include_ftmdse: bool = True,
) -> dict[str, Any]:
    """Return 46-D features, two 32x32 images, and audit-friendly DSP results.

    EMD features are computed per receiver then averaged, avoiding cancellation
    from uncalibrated coherent antenna summation. The STFT likewise averages
    powers. The fast-time image is diagnostic and is not input to the default
    CNN trained on the slow-time image.
    """
    dsp = compute_range_doppler_and_target_bin(raw_burst, cfg, apply_mti=apply_mti)
    emd_features = np.mean([extract_oh2019_emd_features(x) for x in dsp["target_signals"]], axis=0)
    features = np.concatenate((emd_features, _physical_features(dsp)))
    if features.shape != (46,) or not np.isfinite(features).all():
        raise FloatingPointError("Burst feature extraction must produce 46 finite numbers")
    spectrogram, frequency_axis, time_axis = standardized_spectrogram(
        dsp["target_signals"], float(cfg.prf_Hz), return_axes=True)
    fast_result = None
    ftmdse_spectrogram = np.zeros((32, 32), dtype=np.float32)
    if include_ftmdse:
        fast_result = extract_fast_time_interpolated_signal(raw_burst, cfg, dsp["target_bin"])
        ftmdse_spectrogram = standardized_spectrogram(
            fast_result["signal"], float(cfg.sample_rate_Hz), nperseg=256,
            center_frequency_hz=fast_result["carrier_frequency_hz"],
            half_bandwidth_hz=3*fast_result["sigma_range_hz"],
        )
    return {"features": features, "spectrogram": spectrogram,
            "spectrogram_frequency_hz": frequency_axis, "spectrogram_time_s": time_axis,
            "ftmdse_spectrogram": ftmdse_spectrogram, "dsp": dsp,
            "ftmdse": fast_result, "feature_names": list(FEATURE_NAMES),
            "feature_version": FEATURE_VERSION}

