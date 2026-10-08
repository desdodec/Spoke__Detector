#!/usr/bin/env python3
"""GUI session analyser for the Spoke Detector project.

Loads a participant/session WAV and (optionally) a PASS calibration profile,
detects spoke clicks, and exports:

- session_detections.csv
- session_reference.wav   (background-suppressed review with 8 kHz markers)
- session_summary.json    (provenance and diagnostics)

The calibration profile is used as a session reference, not as an absolute
spectral gate. The target recording still receives its own adaptive prominence
threshold so differences in traffic/background level do not invalidate an
otherwise good cable-tie calibration.

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
SUMMARY_FILENAME = "session_summary.json"


@dataclass
class SessionResult:
    detected_hits: int
    duration_s: float
    sample_rate: int
    adaptive_prominence_threshold: float
    calibration_prominence_threshold: float | None
    threshold_ratio_to_calibration: float | None
    median_interval_s: float | None
    interval_mad_s: float | None
    spectral_distance_median: float | None
    spectral_distance_p95: float | None
    calibration_status: str
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

    calibration_threshold = None
    if profile is not None:
        calibration_threshold = float(profile_analysis["adaptive_prominence_threshold"])

    training_floor = 4.0
    if calibration_threshold is not None:
        training_floor = max(4.0, calibration_threshold * 0.20)

    adaptive_threshold = d5.adaptive_prominence_threshold(candidates, training_floor)
    base = d5.adaptive_detect(candidates, adaptive_threshold)
    detections = d5.refine_sequence(base)

    det_times = [float(d.time_seconds) for d in detections]
    distances = profile_distance(analysis_signal, audio.sample_rate, det_times, profile)

    numeric_distances = np.asarray([x for x in distances if x is not None], dtype=float)
    dist_med = float(np.median(numeric_distances)) if numeric_distances.size else None
    dist_p95 = float(np.quantile(numeric_distances, 0.95)) if numeric_distances.size else None

    median_interval, interval_mad = interval_stats(np.asarray(det_times, dtype=float))
    threshold_ratio = (
        float(adaptive_threshold / calibration_threshold)
        if calibration_threshold not in (None, 0.0)
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILENAME
    review_path = output_dir / REVIEW_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hit_index",
            "timestamp_seconds",
            "timestamp_hms",
            "normalized_prominence",
            "adaptive_prominence_threshold",
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
                f"{adaptive_threshold:.6f}",
                f"{det.normalized_prominence / max(adaptive_threshold, 1e-12):.6f}",
                f"{det.onset_strength:.6f}",
                "" if dist is None else f"{dist:.6f}",
                det.source,
            ])

    write_filtered_review(review_path, audio, detections)

    result = SessionResult(
        detected_hits=len(detections),
        duration_s=float(duration_s),
        sample_rate=int(audio.sample_rate),
        adaptive_prominence_threshold=float(adaptive_threshold),
        calibration_prominence_threshold=calibration_threshold,
        threshold_ratio_to_calibration=threshold_ratio,
        median_interval_s=median_interval,
        interval_mad_s=interval_mad,
        spectral_distance_median=dist_med,
        spectral_distance_p95=dist_p95,
        calibration_status=(profile or {}).get("status", "LEGACY / NO PROFILE"),
        analysis_mode="calibrated" if profile is not None else "legacy_adaptive_v5",
    )

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_wav": str(wav_path.resolve()),
        "calibration_profile": str(profile_path.resolve()) if profile_path else None,
        "analysis": {
            "mode": result.analysis_mode,
            "highpass_hz": highpass_hz,
            "min_gap_ms": min_gap_ms,
            "adaptive_prominence_threshold": adaptive_threshold,
            "calibration_prominence_threshold": calibration_threshold,
        },
        "result": result.__dict__,
        "outputs": {
            "detections_csv": str(csv_path.resolve()),
            "reference_wav": str(review_path.resolve()),
        },
        "notes": (
            "The calibration spectral feature distance is diagnostic only and is "
            "not used as a hard acceptance gate, because earlier tests showed that "
            "absolute spectral shape can shift substantially between recordings."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return result, csv_path, review_path, summary_path


class SessionAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spoke Detector — Session Analyzer")
        self.geometry("900x720")
        self.minsize(800, 620)

        self.wav_var = tk.StringVar()
        self.profile_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()

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
            text="Analyse a participant/session WAV and export a reference WAV plus detection CSV.",
        ).pack(anchor="w", pady=(2, 12))

        instructions = (
            "WORKFLOW\n"
            "1. Run calibrate.py whenever the cable tie is replaced or moved.\n"
            "2. For participant/session analysis, choose the recorded WAV below.\n"
            "3. Choose the PASS calibration_profile.json created for that cable-tie setup.\n"
            "4. Choose an output folder.\n"
            "5. Click ANALYSE SESSION.\n\n"
            "Outputs:\n"
            "• session_reference.wav — background-suppressed review audio with a short 8 kHz marker at every accepted hit.\n"
            "• session_detections.csv — timestamp and diagnostic values for every accepted hit.\n"
            "• session_summary.json — provenance, profile path and analysis diagnostics.\n\n"
            "Legacy mode: you may leave the calibration profile blank. The app will then use the current v5 recording-adaptive method, "
            "but calibrated mode is preferred for participant data."
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
        ttk.Button(buttons, text="Clear profile (legacy mode)", command=lambda: self.profile_var.set("")).pack(side="left", padx=8)

        self.result = tk.Text(outer, height=14, wrap="word")
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

            lines = [
                "ANALYSIS COMPLETE",
                "",
                f"Mode: {result.analysis_mode}",
                f"Calibration status: {result.calibration_status}",
                f"Detected hits: {result.detected_hits}",
                f"Recording duration: {result.duration_s:.2f} s",
                f"Adaptive prominence threshold: {result.adaptive_prominence_threshold:.3f} × MAD",
            ]
            if result.calibration_prominence_threshold is not None:
                lines.append(
                    f"Calibration prominence threshold: {result.calibration_prominence_threshold:.3f} × MAD"
                )
                lines.append(
                    f"Session/calibration threshold ratio: {result.threshold_ratio_to_calibration:.3f}"
                )
            if result.median_interval_s is not None:
                lines.append(f"Median inter-click interval: {result.median_interval_s*1000:.1f} ms")
            if result.spectral_distance_median is not None:
                lines.append(f"Median calibration feature distance: {result.spectral_distance_median:.3f}")
                lines.append(f"95th percentile feature distance: {result.spectral_distance_p95:.3f}")

            lines += [
                "",
                f"CSV: {csv_path}",
                f"Reference WAV: {review_path}",
                f"Summary: {summary_path}",
                "",
                "Listen to the reference WAV before treating the CSV as final participant data.",
            ]
            self._set_result("\n".join(lines))
            self.status_var.set("Complete")
        except Exception as exc:
            self.status_var.set("Failed")
            self._set_result(f"ERROR\n\n{exc}\n\n{traceback.format_exc()}")
            messagebox.showerror("Session analysis failed", str(exc))
        finally:
            self.run_button.configure(state="normal")


def main() -> None:
    app = SessionAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
