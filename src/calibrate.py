#!/usr/bin/env python3
"""GUI calibration tool for the Spoke Detector project.

Purpose
-------
Calibrate the detector whenever the plastic cable tie (or its position) changes.
The operator records a controlled wheel-rotation WAV with a known physical
number of spoke strikes:

    expected_hits = spoke_count * wheel_revolutions

The program deliberately does NOT force the detector to return that count.
It first finds the recording's natural high-prominence transient cluster,
then compares the independently detected count with the known physical count.
It also checks transient separation and interval plausibility before declaring
PASS/FAIL.

Outputs
-------
- calibration_profile.json  : session calibration settings + quality metrics
- calibration_detections.csv: accepted calibration events
- calibration_review.wav    : filtered review audio with obvious 8 kHz markers

Run from the repository root:
    python src/calibrate.py

Dependencies: numpy, scipy, tkinter (normally included with desktop Python).
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Tkinter is required for the calibration GUI. Install a desktop Python "
        "distribution that includes Tk/Tkinter."
    ) from exc

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import detector_v4 as d4  # noqa: E402
import detector_v5 as d5  # noqa: E402

PROFILE_FILENAME = "calibration_profile.json"
CSV_FILENAME = "calibration_detections.csv"
REVIEW_FILENAME = "calibration_review.wav"

DEFAULT_COUNT_TOLERANCE_PCT = 2.0
DEFAULT_MIN_SEPARATION_RATIO = 1.18
DEFAULT_MAX_INTERVAL_OUTLIER_PCT = 15.0


@dataclass
class CalibrationMetrics:
    expected_hits: int
    detected_hits: int
    count_error: int
    count_error_pct: float
    adaptive_prominence_threshold: float
    natural_split_ratio: float
    expected_boundary_ratio: float
    interval_median_s: float | None
    interval_mad_s: float | None
    interval_outlier_pct: float | None
    duration_s: float
    sample_rate: int
    pass_count: bool
    pass_separation: bool
    pass_timing: bool
    passed: bool


@dataclass
class CalibrationHit:
    index: int
    time_seconds: float
    prominence: float
    onset_strength: float
    profile_distance: float
    source: str


def _candidate_prominences(candidates: list[d4.Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=float)
    return np.asarray([c.normalized_prominence for c in candidates], dtype=float)


def _natural_split_ratio(values: np.ndarray, threshold: float) -> float:
    if len(values) < 2:
        return 1.0
    vals = np.sort(values)
    pos = int(np.searchsorted(vals, threshold))
    if pos <= 0 or pos >= len(vals):
        return 1.0
    lower = max(float(vals[pos - 1]), 1e-12)
    upper = float(vals[pos])
    return upper / lower


def _expected_boundary_ratio(values: np.ndarray, expected_hits: int) -> float:
    if expected_hits <= 0 or len(values) <= expected_hits:
        return 1.0
    vals = np.sort(values)[::-1]
    weakest_expected = max(float(vals[expected_hits - 1]), 1e-12)
    strongest_rejected = max(float(vals[expected_hits]), 1e-12)
    return weakest_expected / strongest_rejected


def _interval_metrics(times: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if len(times) < 5:
        return None, None, None

    intervals = np.diff(times)
    median = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - median)))

    bad = 0
    total = 0
    for i, interval in enumerate(intervals):
        lo = max(0, i - 4)
        hi = min(len(intervals), i + 5)
        local = intervals[lo:hi]
        if len(local) < 3:
            continue
        local_med = float(np.median(local))
        if local_med <= 0:
            continue
        ratio = interval / local_med
        bad += int(ratio < 0.55 or ratio > 1.80)
        total += 1

    outlier_pct = (100.0 * bad / total) if total else None
    return median, mad, outlier_pct


def _to_int16(signal: np.ndarray) -> np.ndarray:
    return np.round(np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)


def write_filtered_review(path: Path, audio: d4.Audio, detections: list[d4.Detection]) -> None:
    """Review-only audio: suppress background, preserve clicks, add 8 kHz burst."""
    from scipy.signal import butter, sosfiltfilt

    mono = audio.mono.astype(np.float64)
    nyq = audio.sample_rate * 0.5
    low = 1800.0 / nyq
    high = min(10000.0 / nyq, 0.95)
    sos = butter(5, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, mono).astype(np.float32)

    win = max(1, int(round(0.012 * audio.sample_rate)))
    kernel = np.ones(win, dtype=np.float64) / win
    rms = np.sqrt(np.convolve(filtered.astype(np.float64) ** 2, kernel, mode="same") + 1e-12)
    noise_floor = float(np.quantile(rms, 0.55))
    strong = float(np.quantile(rms, 0.90))
    den = max(strong - noise_floor, 1e-9)
    gate = np.clip((rms - noise_floor) / den, 0.0, 1.0)
    gate = 0.06 + 0.94 * gate ** 1.6
    review = filtered * gate.astype(np.float32)

    peak = float(np.max(np.abs(review))) if review.size else 0.0
    if peak > 0:
        review = 0.42 * review / peak

    duration = 0.010
    n = max(1, int(round(duration * audio.sample_rate)))
    t = np.arange(n, dtype=np.float64) / audio.sample_rate
    attack_n = max(1, int(round(0.0005 * audio.sample_rate)))
    env = np.exp(-t * 380.0)
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    burst = np.sin(2 * np.pi * 8000.0 * t) * env
    burst = (burst / (np.max(np.abs(burst)) + 1e-12)).astype(np.float32)

    for det in detections:
        start = int(round(det.time_seconds * audio.sample_rate))
        stop = min(len(review), start + len(burst))
        if start < len(review):
            review[start:stop] += burst[: stop - start] * 0.82

    peak = float(np.max(np.abs(review))) if review.size else 0.0
    if peak > 0.99:
        review *= 0.99 / peak
    wavfile.write(path, audio.sample_rate, _to_int16(review))


def calibrate(
    wav_path: Path,
    output_dir: Path,
    spoke_count: int,
    revolutions: int,
    highpass_hz: float = d4.DEFAULT_HIGHPASS_HZ,
    min_gap_ms: float = d4.DEFAULT_MIN_GAP_MS,
    count_tolerance_pct: float = DEFAULT_COUNT_TOLERANCE_PCT,
    min_separation_ratio: float = DEFAULT_MIN_SEPARATION_RATIO,
    max_interval_outlier_pct: float = DEFAULT_MAX_INTERVAL_OUTLIER_PCT,
) -> tuple[CalibrationMetrics, list[CalibrationHit], Path, Path, Path]:
    if spoke_count <= 0 or revolutions <= 0:
        raise ValueError("Spoke count and revolutions must both be positive integers.")
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    expected_hits = spoke_count * revolutions
    audio = d4.load_wav(wav_path)
    duration_s = len(audio.mono) / audio.sample_rate
    analysis = d4.highpass_analysis(audio.mono, audio.sample_rate, highpass_hz)

    min_gap_s = min_gap_ms / 1000.0
    times, strengths, prominences = d4.all_onset_peaks(analysis, audio.sample_rate, min_gap_s)
    if len(times) == 0:
        raise RuntimeError("No transient candidates were found in the calibration recording.")

    max_strength = float(np.max(strengths)) + 1e-12
    candidates: list[d4.Candidate] = []
    for t, strength, prominence in zip(times, strengths, prominences):
        candidates.append(
            d4.Candidate(
                time_seconds=float(t),
                distance=0.0,
                confidence=1.0,
                onset_strength=float(strength / max_strength),
                normalized_prominence=float(prominence),
            )
        )

    adaptive_threshold = d5.adaptive_prominence_threshold(candidates, training_floor=4.0)
    base = d5.adaptive_detect(candidates, adaptive_threshold)
    detections = d5.refine_sequence(base)

    values = _candidate_prominences(candidates)
    split_ratio = _natural_split_ratio(values, adaptive_threshold)
    expected_ratio = _expected_boundary_ratio(values, expected_hits)

    detected_hits = len(detections)
    count_error = detected_hits - expected_hits
    count_error_pct = 100.0 * abs(count_error) / expected_hits

    det_times = np.asarray([d.time_seconds for d in detections], dtype=float)
    interval_median, interval_mad, interval_outlier_pct = _interval_metrics(det_times)

    pass_count = count_error_pct <= count_tolerance_pct
    pass_separation = max(split_ratio, expected_ratio) >= min_separation_ratio
    pass_timing = interval_outlier_pct is None or interval_outlier_pct <= max_interval_outlier_pct
    passed = pass_count and pass_separation and pass_timing

    metrics = CalibrationMetrics(
        expected_hits=expected_hits,
        detected_hits=detected_hits,
        count_error=count_error,
        count_error_pct=count_error_pct,
        adaptive_prominence_threshold=float(adaptive_threshold),
        natural_split_ratio=float(split_ratio),
        expected_boundary_ratio=float(expected_ratio),
        interval_median_s=interval_median,
        interval_mad_s=interval_mad,
        interval_outlier_pct=interval_outlier_pct,
        duration_s=float(duration_s),
        sample_rate=int(audio.sample_rate),
        pass_count=pass_count,
        pass_separation=pass_separation,
        pass_timing=pass_timing,
        passed=passed,
    )

    hits = [
        CalibrationHit(
            index=i,
            time_seconds=float(d.time_seconds),
            prominence=float(d.normalized_prominence),
            onset_strength=float(d.onset_strength),
            profile_distance=float(d.distance),
            source=d.source,
        )
        for i, d in enumerate(detections, 1)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILENAME
    review_path = output_dir / REVIEW_FILENAME
    profile_path = output_dir / PROFILE_FILENAME

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hit_index", "timestamp_seconds", "normalized_prominence",
            "onset_strength", "source",
        ])
        for hit in hits:
            writer.writerow([
                hit.index,
                f"{hit.time_seconds:.6f}",
                f"{hit.prominence:.6f}",
                f"{hit.onset_strength:.6f}",
                hit.source,
            ])

    write_filtered_review(review_path, audio, detections)

    feature_vectors: list[np.ndarray] = []
    for det in detections:
        feature_vectors.append(d4.feature_vector(analysis, audio.sample_rate, det.time_seconds))

    if feature_vectors:
        feature_matrix = np.stack(feature_vectors)
        feature_centroid = feature_matrix.mean(axis=0)
        feature_scale = feature_matrix.std(axis=0) + 0.15
    else:
        feature_centroid = np.zeros(24, dtype=float)
        feature_scale = np.ones(24, dtype=float)

    profile: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "source_wav": str(wav_path.resolve()),
        "spoke_count": spoke_count,
        "wheel_revolutions": revolutions,
        "expected_hits": expected_hits,
        "analysis": {
            "highpass_hz": float(highpass_hz),
            "min_gap_ms": float(min_gap_ms),
            "adaptive_prominence_threshold": float(adaptive_threshold),
            "feature_centroid": feature_centroid.tolist(),
            "feature_scale": feature_scale.tolist(),
        },
        "acceptance_criteria": {
            "count_tolerance_pct": float(count_tolerance_pct),
            "min_separation_ratio": float(min_separation_ratio),
            "max_interval_outlier_pct": float(max_interval_outlier_pct),
        },
        "metrics": asdict(metrics),
        "notes": (
            "A PASS means the independently selected transient cluster agrees "
            "with the known physical spoke-strike count and passes separation/"
            "timing checks. Participant recordings should preserve this file "
            "with their session metadata."
        ),
    }
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return metrics, hits, profile_path, csv_path, review_path


class CalibrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector — Cable Tie Calibration")
        self.geometry("860x760")
        self.minsize(760, 650)

        self.wav_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.spokes_var = tk.StringVar(value="32")
        self.revs_var = tk.StringVar(value="5")
        self.highpass_var = tk.StringVar(value=str(int(d4.DEFAULT_HIGHPASS_HZ)))
        self.min_gap_var = tk.StringVar(value=str(int(d4.DEFAULT_MIN_GAP_MS)))
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Spoke Detector Calibration",
            font=("TkDefaultFont", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Use this whenever the plastic cable tie is replaced, moved, or sounds noticeably different.",
        ).pack(anchor="w", pady=(2, 12))

        instructions = (
            "CALIBRATION PROCEDURE\n"
            "1. Fit the cable tie at the normal operating position. Keep its protruding length as consistent as possible.\n"
            "2. Lift the wheel so it can rotate freely. Mark the tyre/valve so complete revolutions are easy to count.\n"
            "3. Start the audio recording. Rotate the wheel for the exact number of revolutions entered below.\n"
            "4. Include some slow, medium and faster rotation. Gradual speed changes are useful.\n"
            "5. Stop recording only after the final complete revolution. Do not deliberately add extra clicks.\n"
            "6. Choose the WAV below, enter the wheel spoke count and completed revolutions, then click RUN CALIBRATION.\n\n"
            "The software does NOT force the expected count. It independently detects the strong transient cluster and "
            "then checks whether that count agrees with the physical count. A failed calibration should be repeated before participant recording."
        )
        box = tk.Text(outer, height=12, wrap="word")
        box.insert("1.0", instructions)
        box.configure(state="disabled")
        box.pack(fill="x", pady=(0, 14))

        form = ttk.Frame(outer)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Calibration WAV:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.wav_var).grid(row=0, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(form, text="Choose WAV…", command=self.choose_wav).grid(row=0, column=2, pady=5)

        ttk.Label(form, text="Output folder:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(form, text="Choose Folder…", command=self.choose_output).grid(row=1, column=2, pady=5)

        ttk.Label(form, text="Spokes on wheel:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.spokes_var, width=12).grid(row=2, column=1, sticky="w", padx=8, pady=5)

        ttk.Label(form, text="Complete revolutions recorded:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.revs_var, width=12).grid(row=3, column=1, sticky="w", padx=8, pady=5)

        ttk.Label(form, text="High-pass analysis (Hz):").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.highpass_var, width=12).grid(row=4, column=1, sticky="w", padx=8, pady=5)

        ttk.Label(form, text="Minimum onset gap (ms):").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.min_gap_var, width=12).grid(row=5, column=1, sticky="w", padx=8, pady=5)

        self.run_button = ttk.Button(outer, text="RUN CALIBRATION", command=self.run_calibration)
        self.run_button.pack(fill="x", pady=(16, 8))

        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))

        ttk.Label(outer, textvariable=self.status_var, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        self.result = tk.Text(outer, height=15, wrap="word")
        self.result.pack(fill="both", expand=True, pady=(8, 0))
        self.result.insert("1.0", "No calibration has been run yet.\n")
        self.result.configure(state="disabled")

    def choose_wav(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose calibration WAV",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if filename:
            self.wav_var.set(filename)
            if not self.output_var.get().strip():
                wav = Path(filename)
                self.output_var.set(str(wav.parent / "calibration_output"))

    def choose_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose calibration output folder")
        if folder:
            self.output_var.set(folder)

    def _set_result(self, text: str) -> None:
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.result.configure(state="disabled")

    def run_calibration(self) -> None:
        try:
            wav_path = Path(self.wav_var.get().strip())
            output_dir = Path(self.output_var.get().strip())
            spoke_count = int(self.spokes_var.get().strip())
            revolutions = int(self.revs_var.get().strip())
            highpass_hz = float(self.highpass_var.get().strip())
            min_gap_ms = float(self.min_gap_var.get().strip())
        except Exception:
            messagebox.showerror("Invalid settings", "Check all numeric fields and choose a WAV/output folder.")
            return

        if not wav_path.exists():
            messagebox.showerror("Missing WAV", "Choose an existing calibration WAV file.")
            return
        if not self.output_var.get().strip():
            messagebox.showerror("Missing output folder", "Choose an output folder.")
            return

        expected = spoke_count * revolutions
        if expected < 20:
            if not messagebox.askyesno(
                "Short calibration",
                f"This calibration contains only {expected} expected strikes. "
                "For robust calibration, 100+ strikes is preferable. Continue anyway?",
            ):
                return

        self.run_button.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set("Analysing calibration recording…")
        self.update_idletasks()

        try:
            metrics, hits, profile_path, csv_path, review_path = calibrate(
                wav_path=wav_path,
                output_dir=output_dir,
                spoke_count=spoke_count,
                revolutions=revolutions,
                highpass_hz=highpass_hz,
                min_gap_ms=min_gap_ms,
            )
        except Exception as exc:
            self.progress.stop()
            self.run_button.configure(state="normal")
            self.status_var.set("Calibration failed to run")
            self._set_result(traceback.format_exc())
            messagebox.showerror("Calibration error", str(exc))
            return

        self.progress.stop()
        self.run_button.configure(state="normal")

        status = "PASS" if metrics.passed else "FAIL"
        self.status_var.set(f"CALIBRATION {status}")

        timing_text = (
            "n/a" if metrics.interval_outlier_pct is None
            else f"{metrics.interval_outlier_pct:.1f}% local interval outliers"
        )
        result_text = (
            f"CALIBRATION {status}\n\n"
            f"Expected physical strikes: {metrics.expected_hits}\n"
            f"Detected strikes:          {metrics.detected_hits}\n"
            f"Count error:               {metrics.count_error:+d} ({metrics.count_error_pct:.2f}%)\n\n"
            f"Adaptive threshold:        {metrics.adaptive_prominence_threshold:.2f} x MAD\n"
            f"Natural split ratio:       {metrics.natural_split_ratio:.3f}\n"
            f"Expected-count boundary:   {metrics.expected_boundary_ratio:.3f}\n"
            f"Timing check:              {timing_text}\n\n"
            f"Count check:      {'PASS' if metrics.pass_count else 'FAIL'}\n"
            f"Separation check: {'PASS' if metrics.pass_separation else 'FAIL'}\n"
            f"Timing check:     {'PASS' if metrics.pass_timing else 'FAIL'}\n\n"
            f"Profile: {profile_path}\n"
            f"CSV:     {csv_path}\n"
            f"Review:  {review_path}\n"
        )

        if not metrics.passed:
            result_text += (
                "\nDO NOT use this calibration for participant recording yet. "
                "Check the cable-tie position, recording level, and whether the exact "
                "number of revolutions was completed, then record another calibration.\n"
            )
        else:
            result_text += (
                "\nCalibration is suitable for a short real-world pavement validation. "
                "Preserve calibration_profile.json with the participant session data.\n"
            )

        self._set_result(result_text)
        if metrics.passed:
            messagebox.showinfo("Calibration PASS", "Calibration passed. Review the marker WAV, then run a short pavement validation.")
        else:
            messagebox.showwarning("Calibration FAIL", "Calibration did not meet the quality criteria. See the results panel and repeat calibration.")


def main() -> None:
    app = CalibrationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
