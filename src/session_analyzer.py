#!/usr/bin/env python3
"""GUI session analyser for the Spoke Detector project.

Calibrated mode now treats the calibration prominence threshold as a hard lower
bound. A session may adapt upward when its own evidence supports a stricter
threshold, but it may not silently become more permissive than calibration.

Outputs:
- session_detections.csv
- session_reference.wav    filtered review with 8 kHz marker bursts
- session_isolated.wav     short review windows around accepted detections
- session_summary.json     provenance, thresholds and quality diagnostics

Run from the repository root:
    python src/session_analyzer.py
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Tkinter is required for the session analyser GUI. Install a desktop "
        "Python distribution that includes Tk/Tkinter."
    ) from exc

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import detector_v4 as d4  # noqa: E402
import detector_v5 as d5  # noqa: E402
from calibrate import write_filtered_review  # noqa: E402

CSV_FILENAME = "session_detections.csv"
REVIEW_FILENAME = "session_reference.wav"
ISOLATED_FILENAME = "session_isolated.wav"
SUMMARY_FILENAME = "session_summary.json"


@dataclass
class SessionResult:
    detected_hits: int
    duration_s: float
    sample_rate: int
    recording_adaptive_threshold: float
    effective_prominence_threshold: float
    calibration_prominence_threshold: float | None
    threshold_ratio_to_calibration: float | None
    threshold_source: str
    median_interval_s: float | None
    interval_mad_s: float | None
    spectral_distance_median: float | None
    spectral_distance_p95: float | None
    plateau_change_pct: float | None
    below_floor_pressure_pct: float | None
    quality_status: str
    calibration_status: str
    profile_kind: str
    analysis_mode: str


def load_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported calibration profile schema.")
    if data.get("status") != "PASS":
        raise ValueError(
            "This calibration profile is not marked PASS. Run calibration again "
            "before using it for participant/session analysis."
        )
    analysis = data.get("analysis") or {}
    if "adaptive_prominence_threshold" not in analysis:
        raise ValueError("Calibration profile is missing its prominence threshold.")
    return data


def interval_stats(times: np.ndarray) -> tuple[float | None, float | None]:
    if len(times) < 3:
        return None, None
    intervals = np.diff(times)
    med = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - med)))
    return med, mad


def profile_distance(
    analysis_signal: np.ndarray,
    sample_rate: int,
    times: list[float],
    profile: dict[str, Any] | None,
) -> list[float | None]:
    if profile is None:
        return [None] * len(times)

    cfg = profile.get("analysis") or {}
    centroid = np.asarray(cfg.get("feature_centroid", []), dtype=float)
    scale = np.asarray(cfg.get("feature_scale", []), dtype=float)
    if centroid.size == 0 or scale.size != centroid.size:
        return [None] * len(times)

    distances: list[float | None] = []
    for t in times:
        vec = d4.feature_vector(analysis_signal, sample_rate, t)
        if vec.size != centroid.size:
            distances.append(None)
            continue
        dist = float(np.sqrt(np.mean(((vec - centroid) / np.maximum(scale, 1e-9)) ** 2)))
        distances.append(dist)
    return distances


def detections_at_threshold(
    candidates: list[d4.Candidate], threshold: float
) -> list[d4.Detection]:
    base = d5.adaptive_detect(candidates, float(threshold))
    return d5.refine_sequence(base)


def threshold_diagnostics(
    candidates: list[d4.Candidate], threshold: float
) -> tuple[dict[str, int], float | None, float | None]:
    """Measure whether detections are stable around the chosen threshold.

    A good threshold normally sits on a plateau: moving it upward by 20% should
    not radically change the accepted count. We also inspect 80% of the chosen
    threshold to estimate how much near-threshold transient pressure exists.
    """
    levels = {
        "0.80x": threshold * 0.80,
        "1.00x": threshold,
        "1.10x": threshold * 1.10,
        "1.20x": threshold * 1.20,
    }
    counts = {name: len(detections_at_threshold(candidates, value)) for name, value in levels.items()}

    center = counts["1.00x"]
    if center <= 0:
        return counts, None, None

    plateau_change = 100.0 * abs(counts["1.20x"] - center) / center
    below_pressure = 100.0 * max(0, counts["0.80x"] - center) / center
    return counts, float(plateau_change), float(below_pressure)


def quality_label(
    plateau_change_pct: float | None,
    below_floor_pressure_pct: float | None,
    profile_kind: str,
) -> str:
    warnings: list[str] = []
    if plateau_change_pct is not None and plateau_change_pct > 10.0:
        warnings.append("threshold not on a stable plateau")
    if below_floor_pressure_pct is not None and below_floor_pressure_pct > 25.0:
        warnings.append("many near-threshold background transients")
    if profile_kind == "PROVISIONAL_TEST_ONLY":
        warnings.append("provisional test calibration")
    return "REVIEW: " + "; ".join(warnings) if warnings else "GOOD"


def _to_int16(signal: np.ndarray) -> np.ndarray:
    return np.round(np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)


def write_isolated_review(
    path: Path, audio: d4.Audio, detections: list[d4.Detection]
) -> None:
    """Mute most of the recording and retain only short windows around hits."""
    mono = audio.mono.astype(np.float64)
    nyq = audio.sample_rate * 0.5
    sos = butter(
        5,
        [1800.0 / nyq, min(10000.0 / nyq, 0.95)],
        btype="bandpass",
        output="sos",
    )
    filtered = sosfiltfilt(sos, mono).astype(np.float32)

    peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    if peak > 0:
        filtered = 0.38 * filtered / peak

    isolated = np.zeros_like(filtered)
    pre = int(round(0.030 * audio.sample_rate))
    post = int(round(0.060 * audio.sample_rate))
    fade_n = max(2, int(round(0.005 * audio.sample_rate)))

    for det in detections:
        center = int(round(det.time_seconds * audio.sample_rate))
        a = max(0, center - pre)
        b = min(len(filtered), center + post)
        if b <= a:
            continue
        seg = filtered[a:b].copy()
        f = min(fade_n, len(seg) // 2)
        if f > 1:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            seg[:f] *= ramp
            seg[-f:] *= ramp[::-1]
        isolated[a:b] += seg

    # Same distinct 8 kHz review marker used by the filtered reference WAV.
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
        stop = min(len(isolated), start + len(burst))
        if start < len(isolated):
            isolated[start:stop] += burst[: stop - start] * 0.88

    peak = float(np.max(np.abs(isolated))) if isolated.size else 0.0
    if peak > 0.99:
        isolated *= 0.99 / peak
    wavfile.write(path, audio.sample_rate, _to_int16(isolated))


def analyse_session(
    wav_path: Path,
    output_dir: Path,
    profile_path: Path | None = None,
    highpass_override: float | None = None,
    min_gap_override_ms: float | None = None,
) -> tuple[SessionResult, Path, Path, Path]:
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    profile = load_profile(profile_path)
    profile_analysis = (profile or {}).get("analysis", {})
    profile_kind = str((profile or {}).get("profile_kind", "FORMAL_CALIBRATION" if profile else "NONE"))

    highpass_hz = float(
        highpass_override
        if highpass_override is not None
        else profile_analysis.get("highpass_hz", d4.DEFAULT_HIGHPASS_HZ)
    )
    min_gap_ms = float(
        min_gap_override_ms
        if min_gap_override_ms is not None
        else profile_analysis.get("min_gap_ms", d4.DEFAULT_MIN_GAP_MS)
    )

    audio = d4.load_wav(wav_path)
    duration_s = len(audio.mono) / audio.sample_rate
    analysis_signal = d4.highpass_analysis(audio.mono, audio.sample_rate, highpass_hz)

    times, strengths, prominences = d4.all_onset_peaks(
        analysis_signal, audio.sample_rate, min_gap_ms / 1000.0
    )
    if len(times) == 0:
        raise RuntimeError("No transient candidates were found in this recording.")

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

    # Always estimate what the recording itself would choose, for diagnostics.
    recording_adaptive = d5.adaptive_prominence_threshold(candidates, 4.0)

    calibration_threshold: float | None = None
    if profile is not None:
        calibration_threshold = float(profile_analysis["adaptive_prominence_threshold"])

    if calibration_threshold is None:
        effective_threshold = float(recording_adaptive)
        threshold_source = "recording_adaptive"
    else:
        # Critical calibrated-mode rule: never become more permissive than the
        # calibration. The session may adapt upward only.
        effective_threshold = float(max(recording_adaptive, calibration_threshold))
        threshold_source = (
            "recording_adaptive_above_calibration"
            if recording_adaptive > calibration_threshold
            else "calibration_floor"
        )

    detections = detections_at_threshold(candidates, effective_threshold)
    sweep_counts, plateau_change, below_pressure = threshold_diagnostics(
        candidates, effective_threshold
    )

    det_times = [float(d.time_seconds) for d in detections]
    distances = profile_distance(analysis_signal, audio.sample_rate, det_times, profile)
    numeric_distances = np.asarray([x for x in distances if x is not None], dtype=float)
    dist_med = float(np.median(numeric_distances)) if numeric_distances.size else None
    dist_p95 = float(np.quantile(numeric_distances, 0.95)) if numeric_distances.size else None

    median_interval, interval_mad = interval_stats(np.asarray(det_times, dtype=float))
    threshold_ratio = (
        float(effective_threshold / calibration_threshold)
        if calibration_threshold not in (None, 0.0)
        else None
    )
    quality_status = quality_label(plateau_change, below_pressure, profile_kind)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILENAME
    review_path = output_dir / REVIEW_FILENAME
    isolated_path = output_dir / ISOLATED_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hit_index",
            "timestamp_seconds",
            "timestamp_hms",
            "normalized_prominence",
            "effective_prominence_threshold",
            "prominence_ratio",
            "onset_strength",
            "calibration_feature_distance",
            "source",
        ])
        for i, (det, dist) in enumerate(zip(detections, distances), 1):
            writer.writerow([
                i,
                f"{det.time_seconds:.6f}",
                d4.timestamp_hms(det.time_seconds),
                f"{det.normalized_prominence:.6f}",
                f"{effective_threshold:.6f}",
                f"{det.normalized_prominence / max(effective_threshold, 1e-12):.6f}",
                f"{det.onset_strength:.6f}",
                "" if dist is None else f"{dist:.6f}",
                det.source,
            ])

    write_filtered_review(review_path, audio, detections)
    write_isolated_review(isolated_path, audio, detections)

    result = SessionResult(
        detected_hits=len(detections),
        duration_s=float(duration_s),
        sample_rate=int(audio.sample_rate),
        recording_adaptive_threshold=float(recording_adaptive),
        effective_prominence_threshold=float(effective_threshold),
        calibration_prominence_threshold=calibration_threshold,
        threshold_ratio_to_calibration=threshold_ratio,
        threshold_source=threshold_source,
        median_interval_s=median_interval,
        interval_mad_s=interval_mad,
        spectral_distance_median=dist_med,
        spectral_distance_p95=dist_p95,
        plateau_change_pct=plateau_change,
        below_floor_pressure_pct=below_pressure,
        quality_status=quality_status,
        calibration_status=(profile or {}).get("status", "LEGACY / NO PROFILE"),
        profile_kind=profile_kind,
        analysis_mode="calibrated" if profile is not None else "legacy_adaptive_v5",
    )

    summary = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_wav": str(wav_path.resolve()),
        "calibration_profile": str(profile_path.resolve()) if profile_path else None,
        "analysis": {
            "mode": result.analysis_mode,
            "highpass_hz": highpass_hz,
            "min_gap_ms": min_gap_ms,
            "recording_adaptive_threshold": recording_adaptive,
            "calibration_prominence_threshold": calibration_threshold,
            "effective_prominence_threshold": effective_threshold,
            "threshold_source": threshold_source,
            "threshold_sweep_refined_counts": sweep_counts,
        },
        "result": result.__dict__,
        "outputs": {
            "detections_csv": str(csv_path.resolve()),
            "reference_wav": str(review_path.resolve()),
            "isolated_wav": str(isolated_path.resolve()),
        },
        "notes": (
            "In calibrated mode the calibration prominence threshold is a hard "
            "lower bound. Recording adaptation may only raise the effective threshold. "
            "Spectral feature distance remains diagnostic rather than a hard gate."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Preserve the existing four-value API used by scripts/workflows.
    return result, csv_path, review_path, summary_path


class SessionAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector — Session Analyzer")
        self.geometry("920x790")
        self.minsize(800, 650)

        self.wav_var = tk.StringVar()
        self.profile_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()
        self._suggest_default_profile()

    def _suggest_default_profile(self) -> None:
        default_profile = THIS_DIR.parent / "calibration" / "calibration_profile.json"
        if default_profile.exists():
            self.profile_var.set(str(default_profile))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Spoke Detector — Session Analyzer",
            font=("TkDefaultFont", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Analyse a session WAV using a calibrated lower detection boundary.",
        ).pack(anchor="w", pady=(2, 12))

        instructions = (
            "WORKFLOW\n"
            "1. Choose the participant/session WAV.\n"
            "2. Use the calibration_profile.json created for the current cable-tie setup.\n"
            "3. Choose an output folder and click ANALYSE SESSION.\n\n"
            "CALIBRATED MODE\n"
            "The session may automatically choose a stricter threshold, but it cannot "
            "drop below the calibrated threshold. This prevents noisy sessions from "
            "suddenly accepting large numbers of weak background transients.\n\n"
            "Outputs include a filtered reference WAV, an isolated-hit review WAV, CSV, "
            "and a JSON quality report with threshold-stability diagnostics."
        )
        box = tk.Text(outer, height=13, wrap="word")
        box.insert("1.0", instructions)
        box.configure(state="disabled")
        box.pack(fill="x", pady=(0, 14))

        form = ttk.Frame(outer)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Session WAV:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.wav_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(form, text="Choose…", command=self._choose_wav).grid(row=0, column=2)

        ttk.Label(form, text="Calibration profile:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.profile_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(form, text="Choose…", command=self._choose_profile).grid(row=1, column=2)

        ttk.Label(form, text="Output folder:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(form, text="Choose…", command=self._choose_output).grid(row=2, column=2)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(18, 10))
        self.run_button = ttk.Button(buttons, text="ANALYSE SESSION", command=self._run)
        self.run_button.pack(side="left")
        ttk.Button(
            buttons,
            text="Clear profile (legacy mode)",
            command=lambda: self.profile_var.set(""),
        ).pack(side="left", padx=8)

        self.result = tk.Text(outer, height=16, wrap="word")
        self.result.pack(fill="both", expand=True, pady=(4, 8))
        self.result.configure(state="disabled")
        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w")

    def _choose_wav(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose session WAV",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.wav_var.set(path)
            if not self.output_var.get():
                p = Path(path)
                self.output_var.set(str(p.parent / f"{p.stem}_analysis"))

    def _choose_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose PASS calibration profile",
            filetypes=[("Calibration profile", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.profile_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_var.set(path)

    def _set_result(self, text: str) -> None:
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.result.configure(state="disabled")

    def _run(self) -> None:
        try:
            wav_text = self.wav_var.get().strip()
            out_text = self.output_var.get().strip()
            profile_text = self.profile_var.get().strip()
            if not wav_text:
                raise ValueError("Choose a session WAV first.")
            if not out_text:
                raise ValueError("Choose an output folder.")

            wav_path = Path(wav_text)
            output_dir = Path(out_text)
            profile_path = Path(profile_text) if profile_text else None

            self.run_button.configure(state="disabled")
            self.status_var.set("Analysing…")
            self.update_idletasks()

            result, csv_path, review_path, summary_path = analyse_session(
                wav_path=wav_path,
                output_dir=output_dir,
                profile_path=profile_path,
            )
            isolated_path = output_dir / ISOLATED_FILENAME

            lines = [
                "ANALYSIS COMPLETE",
                "",
                f"Mode: {result.analysis_mode}",
                f"Calibration status: {result.calibration_status}",
                f"Profile kind: {result.profile_kind}",
                f"Quality: {result.quality_status}",
                "",
                f"Detected hits: {result.detected_hits}",
                f"Recording duration: {result.duration_s:.2f} s",
                f"Recording-only adaptive threshold: {result.recording_adaptive_threshold:.3f}",
                f"Effective threshold used: {result.effective_prominence_threshold:.3f}",
                f"Threshold source: {result.threshold_source}",
            ]
            if result.calibration_prominence_threshold is not None:
                lines.append(
                    f"Calibration threshold: {result.calibration_prominence_threshold:.3f}"
                )
            if result.plateau_change_pct is not None:
                lines.append(
                    f"Count change at +20% threshold: {result.plateau_change_pct:.1f}%"
                )
            if result.below_floor_pressure_pct is not None:
                lines.append(
                    f"Extra detections at -20% threshold: {result.below_floor_pressure_pct:.1f}%"
                )

            lines.extend([
                "",
                f"CSV: {csv_path}",
                f"Filtered review: {review_path}",
                f"Isolated review: {isolated_path}",
                f"Summary: {summary_path}",
            ])
            if result.profile_kind == "PROVISIONAL_TEST_ONLY":
                lines.extend([
                    "",
                    "WARNING: This is the provisional Bike Test 2 profile, not a formal known-revolution calibration.",
                ])

            self._set_result("\n".join(lines))
            self.status_var.set("Complete")
        except Exception as exc:
            self.status_var.set("Failed")
            self._set_result(traceback.format_exc())
            messagebox.showerror("Session analysis failed", str(exc))
        finally:
            self.run_button.configure(state="normal")


def main() -> None:
    app = SessionAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
