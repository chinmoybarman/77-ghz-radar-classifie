# 60 GHz BGT60TR13C drone classification research package

## Scope and evidence status

This project implements an offline, reproducible path from **real-valued raw IF samples** to range-Doppler maps, micro-Doppler representations, engineered features, and class predictions for the Infineon DEMO BGT60TR13C. The classes are `drone`, `bird`, `other`, and `background`. It is a research baseline for a dominant candidate target in a recording, not a validated multi-target surveillance product.

The BGT benchmark uses simulated observations only. Its metrics measure software execution and separation of the simulator's classes; they do not establish drone detection range, bird rejection, real-world accuracy, calibrated probability, or equivalence to the cited experiments. A separate user-provided KTH 77 GHz measured-data experiment is documented in [kth77_experiment.md](kth77_experiment.md), with its own adapter, schema, checkpoints and measurement-grouped split. No BGT60TR13C hardware measurements are asserted. Benchmark outputs retain provenance and session assignments alongside metrics.

## Literature and implementation boundary

Passafiume et al. [1] developed a multi-motor vibration model and compared it with **77 GHz** measurements. Peter and Reddy [2] describe fast-time processing with range-Doppler filtering, interpolation, and further EMD-based denoising. Oh et al. [3] describe a classifier for tracks supplied by an **X-band** surveillance radar, including burst selection, pruning, EMD features, and temporal filtering. Combining these approaches at 60 GHz is an engineering adaptation; their published performance does not transfer to this board or dataset.

| Requested assumption | Implementation decision and scientific consequence |
| --- | --- |
| All three papers directly validate this 60 GHz board | Their methods inform the software; this board requires its own validation campaign. |
| RX1–RX2 measures azimuth | Correct the physical baseline to RX1–RX3. RX2–RX3 is the elevation baseline in the reference board orientation [5]. RX1–RX2 is diagonal. |
| ADC hardware maximum is 2 MSps | The datasheet describes up to 4 MSps [4]. The nominal acquisition here remains 1 MSps. Accepted rates depend on the actual SDK/device configuration. |
| Every IF file uses a 92-byte header | Honor declared legacy header size and format. An official BGT60TR13C example has an 88-byte header [7]. Reject unsupported or ambiguous payloads. |
| Three-channel power averaging adds exactly 4.77 dB SNR | Average power noncoherently; there is no universal 10 log10(3) improvement in mean signal/noise power ratio. Noise independence, target coherence, and detection statistic matter. |
| A complex range FFT removes Doppler ambiguity | It retains phase and can distinguish signs within the sampled Doppler interval. Frequencies outside that interval still alias. |
| Idle-time interpolation guarantees unaliased recovery | Preserve timestamps and label interpolated samples. Missing measurements cannot generally be recovered by straight lines. |
| Every printed feature equation uniquely defines a program | Fix histogram, FFT-centering, endpoint, and zero-denominator conventions explicitly below. These choices prevent a false claim of bit-identical replication. |
| Frame, burst, and surveillance scan are interchangeable | One BGT frame is a burst. Frame-wise temporal classification is an adaptation; a true scan may contain several bursts and require an externally associated target track. |

## Configuration and nominal theory

All processing functions take a configuration object. They do not embed an assumed sampling rate, chirp count, or bandwidth. The default JSON is an explicit nominal acquisition design, not a claim that USB hardware has accepted it.

Using the exact speed of light, `c = 299792458 m/s`, the default design yields:

| Quantity | Nominal value |
| --- | --- |
| RF start / end | 59 / 61 GHz |
| Center frequency / wavelength | 60 GHz / 4.99654 mm |
| Sweep bandwidth / ideal range resolution | 2 GHz / 0.0749481 m |
| ADC sampling rate / samples per chirp | 1 MHz / 128 |
| Nominal sampled chirp duration | 128 microseconds |
| Nominal slope, assuming sweep equals sampled window | 1.5625e13 Hz/s |
| Chirp repetition interval / frequency | 500 microseconds / 2000 Hz |
| Nominal idle samples between chirps | 372 |
| Chirps per frame / nominal burst scheduling interval | 64 / 32 ms |
| Frame repetition interval | 66.67 ms |
| Doppler Nyquist interval | [-1000, 1000) Hz |
| Corresponding signed radial velocity interval | approximately [-2.4983, 2.4983) m/s |
| Doppler FFT-bin spacing | 31.25 Hz |
| Nominal ADC Nyquist range | 4.79668 m |

The nominal range limit follows from `R = c f_beat / (2 slope)` at `f_beat = fs/2`; it is not a measured detection distance. The analog passband, SNR, spectral leakage, and target Doppler impose additional limits. Zero-padding changes grid spacing, not physical range resolution.

The actual synthesizer ramp may include settling and acquisition offsets, so its duration need not equal `N/fs`. Use the acquisition's effective sweep/ramp metadata when available; otherwise record the nominal timing assumption. The first-to-last sample span is `(N-1)/fs`, not `N/fs`. SDK timing quantization must be inspected during acquisition. Frame gaps must never be removed and then treated as uniformly sampled slow time.

## Raw data contract and quality controls

Canonical data have shape `(frames, 3, chirps, samples)`. RX order and any flattening must be established by metadata. The real IF samples can be integer ADC codes or explicitly normalized values. Modern Radar Fusion GUI recordings use `radar.npy` with configuration and sensor metadata; Infineon's format guide describes older binary and text exports [6]. Plain tracking reports are not ADC data.

The loader checks shape/configuration agreement, finite values, inactive receivers, and clipping. It rejects detection-table column names and four-column tracking-style arrays under the requested Rule 8 guard. No loader can prove physical provenance from numbers alone; preserve original acquisition metadata and file identity. A rejected recording must be investigated rather than silently reshaped into valid-looking data.

The four-panel inspection shows raw receiver chirps, a DC-removed chirp with Hilbert envelope, the RX1 chirp/sample matrix, and range profiles. A Hilbert envelope is a signal diagnostic; it does not turn the board into an I/Q receiver. Receiver variance and saturation statistics accompany the plot. Synthetic diagnostics carry `[SYNTHETIC 60 GHz BENCHMARK - NOT REAL HARDWARE DATA]`.

## Processing conventions

### Range and slow-time processing

Subtract each chirp's fast-time DC, apply a Hann window, then use a real FFT along samples. For `N=128` there are 65 complex range coefficients. For each channel and range cell, MTI subtracts the complex mean across the chirps of that frame. Apply a slow-time Hann window, FFT and center the Doppler axis; average the three receiver power maps. MTI suppresses stationary returns, including potentially useful static body information.

The strongest eligible range cell is a **candidate**, not a calibrated detection. A maximum exists even in pure noise. Candidate energy, noise-relative diagnostics, and pruning flags should be evaluated against independently labeled background data before deploying a detection threshold. The original pruning threshold 0.45 is retained as a diagnostic; it is not calibrated for this radar. In particular, a non-UAV decision does not uniquely identify `bird`, `other`, or `background`.

For feature extraction, normalize each complex target slow-time vector by its mean magnitude, with deterministic handling of zero-energy inputs. Phase-difference stability uses the physical receiver baselines. This statistic is not a calibrated angle estimate: absolute angles require known antenna orientation, gain/phase calibration, and an appropriate array response model.

### Fast-time reconstruction experiment

The implementation uses a 2D FFT, Gaussian range passbands around both conjugate target bins for real IF, an all-pass Doppler response, then a 2D inverse FFT. The chosen passband width is exposed; a narrow filter can remove blade modulation, while a wide filter admits other objects. A time-frequency image of the result contains range beat as well as modulation and is not automatically a pure Doppler measurement.

There are `Q = round(fs*CRI) - N` unobserved grid samples between chirps. Interpolate fractions `1/(Q+1)` through `Q/(Q+1)` between the last measured sample and the next measured sample. This preserves sample timestamps without duplicating either measured endpoint. It is a deliberate discrete-time correction to the endpoint convention in the requested transcription. Retain an interpolation mask and timing metadata when interpreting the result. At the default settings, 74.4% of each inter-chirp interval is unobserved. Useful reconstruction depends on signal dynamics and acquisition duty cycle; increasing the output sampling frequency does not create new information.

### EMD and the 46-feature contract

For a length-64 complex vector `s`, concatenate `abs(s)`, `real(s)`, and `imag(s)` into a length-192 real vector. Cubic-spline envelope sifting uses up to ten iterations for each of three modes, with deterministic boundaries and handling when fewer modes exist. This does not reproduce an unspecified MATLAB EMD boundary implementation exactly. The 34 mode features follow the requested grouping; twelve additional spectral/range/phase statistics produce a stable 46-column feature schema.

The code makes these numerical conventions explicit: binary nonnegative/negative zero-crossing indicators; a 95th-percentile robust maximum; sample standard deviation; empirical amplitude probabilities from 16 equal-width histogram bins; centered FFT power; total spectral energy for the scalar energy group and maximum-bin power for its maximum group; guarded ratios for degenerate modes. The probability estimator and interpretation of scalar/vector spectral notation are declared implementation choices, not claims of undocumented author settings. Stable feature names and preprocessing travel with each saved model.

Spectrograms use per-receiver complex slow-time STFT power, average power across receivers, and resize to 32 by 32 with consistent normalization. An enlarged image does not increase the observation duration or true time-frequency resolution. The fast-time branch is saved separately and must be described as an experimental reconstructed IF representation.

## Synthetic data and provenance

Simulated drone signals combine configurable body/rotor scattering and multi-motor displacement components. The blade integral uses the unnormalized convention `sinc(x)=sin(x)/x`; in NumPy this is `np.sinc(x/pi)`. A physical linear chirp has phase `2*pi*(f_start*t + slope*t*t/2)`. Consequently its nominal range beat is `2*slope*R/c`; the coefficient called `mu` in [1]'s `mu*t*t` phase is half that physical slope. A literal factor-four transcription must not be mixed with a factor-two range calibration.

Simulated bird, other, and background classes are controlled surrogates for software checks. They are not validated electromagnetic models of those object populations. Randomized range, velocity, phase, noise, and session parameters reduce trivial reuse, but they do not close the simulation-to-hardware domain gap. Every synthetic recording and registry row is marked `is_synthetic=True`; model and metric provenance retain that fact. A mixed or wholly synthetic training run requires explicit allowance, except for the explicitly named synthetic benchmark command.

## Learning and evaluation protocol

1. Validate the registry and group complete recordings by `session_id` before feature fitting or sample selection. Keep sessions disjoint across train, validation, and test. A new file from the same scene/session is not independent data.
2. Select middle-70% training frames within each training recording. Evaluate held-out frames and separately the center burst convention. Do not split overlapping windows or frames from one session across subsets.
3. Fit scaling and class models using training observations only. Random Forest uses 250 trees, balanced class weights, and seed 42. The CNN uses 32 by 32 inputs, Adam, validation loss for early stopping, and the best saved checkpoint.
4. TERRM uses the reduced polynomial basis and the class-weighted closed-form linear solve. For degree two, the basis contains constant, coordinate, coordinate-squared, sum, sum-squared, and coordinate-times-sum terms [8]. In Eq. (3)'s `tau +/- eta` target convention, `eta=0` makes positive and negative targets identical. The practical default is `tau=0.5, eta=0.5`, giving 0/1 targets; this is disclosed rather than labeled exact author hyperparameter replication.
5. Compute held-out confusion matrices, accuracy, balanced accuracy, and per-class scores. Softmax-transformed TERRM/CNN scores and RF vote fractions are model confidence outputs, not calibrated physical probabilities.
6. Evaluate frame-wise temporal filtering separately. Accumulate predicted-class confidence, abstain during the first three frames, emit the first-four majority at frame four, and use accumulated scores on subsequent label disagreement. Reset for each independent recording/track; never mix unrelated objects.

Report the number of sessions, recordings, and evaluated frames as well as class coverage and abstention count. Frame-level scores have correlated observations; future uncertainty intervals should resample independent sessions rather than pretending every chirp is independent. Hyperparameter selection and threshold tuning must not use the final test sessions. Tiny synthetic splits are smoke tests, not evidence of stable generalization.

## Hardware validation campaign

Before a field study, preserve the GUI/SDK configuration, board identifier, firmware/SDK versions, RX order, raw file, session ID, UTC recording time, and ground-truth label. Verify the range scale using reflectors at measured distances, Doppler sign using known motion, baseline phase with a known direction, and clipping/noise with targets absent. Record gain and filter settings and identify any missing frames.

Collect separate sessions across days, positions, backgrounds, ranges, orientations, and target types. Include motor-off drones, moving clutter, genuine bird observations where feasible, and empty scenes. Hold out complete days/locations or target instances for a stronger external test. Preserve false positives and low-SNR failures rather than deleting difficult examples. Operate only in authorized test areas and follow the equipment and flight requirements applicable to the actual study.

## Reproducibility and limitations

The completed verification run, numerical regression coverage, computed synthetic scores, and fixes are recorded in [verification.md](verification.md). The [pre-board guide](pre_board_guide.md) corrects dataset compatibility and range-planning advice. Configuration checks retain effective SDK settings, reject conflicting recording profiles and ramp slopes, and allow only representation-scale float32 timing tolerance for legacy headers. Diagnostic STFT axes use actual analysis-window centers.

ADC validation rejects complex input and fractional ADC counts, and checks clipping in each frame and receiver so a short failure cannot be diluted by a long recording. Supplied frame timestamps must be finite, increasing, and aligned with the frame count. Exports retain these times; generated nominal times are identified. Temporal smoothing resets after a timestamp gap greater than 1.5 configured frame periods. This heuristic detects interruptions in supplied timing and cannot establish whether nominal SDK timestamps conceal dropped frames. Invalid SDK acquisition readback fails validation rather than silently substituting the requested settings.

Run the unit suite and synthetic benchmark as documented in `README.md`. The pipeline saves actual computed artifacts; this prose does not insert assumed accuracy values. Seeds stabilize the synthetic data and learning, but bitwise identity across numerical libraries and devices is not promised. Dependency pins record the software environment; the optional vendor USB SDK is installed separately from the matching RDK wheel and must be version-recorded when used.

Remaining experimental limits include single-candidate selection, unresolved multiple targets, no independent target association, oscillator/phase noise not validated against this board, interpolation artifacts, uncalibrated confidence, and an unmeasured simulation-to-real gap. Real-data claims require a new, provenance-checked evaluation using real held-out sessions.

## References

[1] M. Passafiume, N. Rojhani, G. Collodi, and A. Cidronali, “Modeling Small UAV Micro-Doppler Signature Using Millimeter-Wave FMCW Radar,” *Electronics*, vol. 10, no. 6, art. 747, 2021. [Publisher and DOI](https://www.mdpi.com/2079-9292/10/6/747).

[2] S. Peter and V. V. Reddy, “Extraction of Unaliased High-Frequency Micro-Doppler Signature using FMCW Radar,” arXiv:2204.09621v1, 2022. [Primary manuscript](https://arxiv.org/html/2204.09621v1).

[3] B.-S. Oh, X. Guo, and Z. Lin, “A UAV classification system based on FMCW radar micro-Doppler signature analysis,” *Expert Systems with Applications*, vol. 132, pp. 239–255, 2019, doi:10.1016/j.eswa.2019.05.007. [Publisher](https://www.sciencedirect.com/science/article/pii/S0957417419303203); [author-uploaded accepted manuscript](https://www.researchgate.net/publication/332996672_A_UAV_Classification_System_based_on_FMCW_Radar_Micro-Doppler_Signature_Analysis).

[4] Infineon Technologies, *XENSIV BGT60TR13C Datasheet*, v2.4.9, Nov. 2023. [Official datasheet](https://www.infineon.com/assets/row/public/documents/24/49/infineon-bgt60tr13c-datasheet-en.pdf?fileId=8ac78c8c7d718a49017d94bac88e5d43).

[5] Infineon Technologies, *BGT60TR13C Shield*, AN600, v2.2, Nov. 2021, sec. 3.2. [Official application note](https://community.infineon.com/gfawx74859/attachments/gfawx74859/Radarsensor/831/1/AN600%20-%20BGT60TR13C%20Shield%20-%20V2.2.pdf).

[6] Infineon Developer Community, “Radar Fusion GUI data recording formats (.npy, .bin, .txt) across versions,” Feb. 2025, updated Dec. 2025. [Official format guide](https://community.infineon.com/t5/Knowledge-Base-Articles/Radar-Fusion-GUI-data-recording-formats-npy-bin-txt-across-versions/ta-p/948400).

[7] Infineon Developer Community, “How to calculate normalized ADC signal ... BGT60TR13C sensor,” Jun. 2023, reported raw header example and moderator response. [Header example](https://community.infineon.com/t5/Radar-sensors/How-to-calculate-normalized-ADC-signal-of-values-and-Magnitude-dBF-S-values-of/td-p/448107). This is an observed export example, not a universal binary format specification.

[8] K.-A. Toh, Q.-L. Tran, and D. Srinivasan, “Benchmarking a reduced multivariate polynomial pattern classifier,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 26, no. 6, pp. 740–755, 2004. [Primary paper, institutional repository](https://eprints.uet.vnu.edu.vn/eprints/id/eprint/3267/1/2004%20Benchmarking%20a%20Reduced%20Multivariate%20Polynomial%20Pattern%20Classifier.pdf).

