# Spoke Detector

Audio analysis project for detecting bicycle-spoke impacts created by a cable tie striking the spokes while the bicycle is pushed.

The detector learns a profile from a short WAV containing a known number of genuine spoke hits, then applies that profile to a longer recording. It deliberately does **not** depend on a fixed walking speed or a fixed click pitch.

## Current inputs

- `short_sample/7_clicks.wav` — training sample containing exactly 7 genuine spoke hits
- `audio/Bike_test.wav` — longer recording to analyse

Both WAV files should use the same sample rate.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Run

From the repository root:

```bash
python src/detector.py
```

The default run assumes 7 training hits. To use a future training file with a different known count:

```bash
python src/detector.py --training short_sample/20_clicks.wav --training-hits 20
```

## Outputs

The detector writes these files to `output/`:

- `detections.csv` — one row per accepted hit, including timestamp, confidence, profile distance, and onset strength
- `review_clicks.wav` — the original recording with an obvious synthetic marker click mixed at every detected spoke hit
- `detected_hits.wav` — all accepted hit excerpts concatenated with short gaps for rapid listening

Generated output files are ignored by Git.

## How version 1 works

1. Spectral-flux onset detection automatically locates the known number of hits in the training WAV.
2. Each training hit is represented by amplitude-normalised spectral-band energy, short-time envelope shape, zero-crossing rate, spectral centroid, and spectral bandwidth.
3. The long recording is scanned for transient candidates.
4. Each candidate is compared with the training hits using a normalised feature-space distance.
5. The acceptance threshold is learned from leave-one-out variation inside the training sample and deliberately widened for high recall.

Timing between successive hits is not required for classification, so changes in walking speed, acceleration, deceleration, and uneven pushing do not directly break the detector.

## Current baseline

Using the supplied 7-hit training file, the automatic profiler recovers all seven training onsets. On the supplied 78.2-second `Bike_test.wav`, the current high-recall baseline produces 626 candidate detections.

That number is **not yet ground truth**. The next validation step is to listen to `review_clicks.wav`, identify false positives and missed genuine hits, and use those errors to improve the classifier. Difficult negative examples will be especially useful for version 2.
