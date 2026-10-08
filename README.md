# Spoke Detector

Audio analysis project for detecting bicycle-spoke impacts created by a plastic cable tie striking the spokes while a bicycle is pushed.

The detector is designed to tolerate changing walking speed, acceleration/deceleration, variable click pitch, and real outdoor noise. Version 5 uses a recording-adaptive transient-prominence split instead of requiring every future recording to match the exact spectral character of the original cable tie.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
```

Tkinter is also required for the calibration GUI. It is normally included with desktop Python distributions.

## Cable-tie calibration GUI

Run calibration whenever the plastic cable tie is replaced, moved, or sounds noticeably different:

```bash
python src/calibrate.py
```

The GUI provides file/folder choosers and the calibration procedure.

Recommended procedure:

1. Fit the cable tie in its normal operating position and keep its protruding length reasonably consistent.
2. Lift the wheel so it can rotate freely.
3. Use the valve or a visible tyre mark to count exact complete wheel revolutions.
4. Record a calibration WAV containing a known number of complete revolutions. Include slow, medium and faster rotation.
5. In the GUI, choose the WAV and output folder, enter the wheel's spoke count and number of completed revolutions, then click **RUN CALIBRATION**.
6. The known physical count is `spokes × revolutions`.

The calibration detector does **not** force itself to return the expected number of hits. It independently finds the recording's natural high-prominence transient cluster, then checks whether that detected count agrees with the physical count. It also checks signal separation and local interval plausibility.

A successful calibration produces:

- `calibration_profile.json` — session calibration settings, learned feature statistics and quality metrics
- `calibration_detections.csv` — accepted calibration events
- `calibration_review.wav` — background-suppressed review audio with a distinct 8 kHz marker at every accepted hit

The GUI reports **CALIBRATION PASS** or **CALIBRATION FAIL**. A failed calibration should be repeated before participant recording.

For robust calibration, prefer at least 100 physical spoke strikes. For example, a 32-spoke wheel rotated five complete revolutions gives 160 expected impacts.

## Current detector

Version 5:

```bash
python src/detector_v5.py --input "audio/Bike test 2.wav" --output-dir output_v5
```

Version 5 keeps the high-pass/onset machinery from v4 but adapts its prominence threshold to each target recording. Spectral distance is retained as diagnostic metadata rather than used as a hard cross-recording rejection rule.

The detector does not assume a fixed walking speed or a fixed spoke period.

## Validation workflow

A calibration PASS should be followed by a short real-world pavement validation before participant recording. That validation should include normal bike handling and realistic environmental noise such as traffic. Calibration and validation should remain separate from participant data so detector quality is not judged on the same recording used to tune it.

## Research interpretation

The spoke-click event train measures bicycle wheel motion and is therefore a proxy for participant forward-movement rhythm. It should not be described as direct heel-strike/toe-off gait measurement unless that relationship is independently validated.
