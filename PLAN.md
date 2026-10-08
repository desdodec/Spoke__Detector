# Spoke Detector — build and test plan

## Goal
Detect each cable-tie/spoke impact in a longer bicycle recording and produce:

1. `detections.csv` — one row per detected spoke hit with timestamp and score.
2. `review_clicks.wav` — copy of the original audio with a synthetic review click mixed onto every detection.
3. `detected_hits.wav` — a compact audio file containing the detected hit snippets for fast listening/review.
4. Optional diagnostics: plots, candidate scores and false-positive/false-negative review data.

## Initial observations from `7_clicks.wav`

The supplied training clip is about 0.95 s at 44.1 kHz and contains seven clearly separated transient impacts. Their spacing is very consistent, roughly 143–150 ms apart.

The useful invariant is **not a single pitch**. Across the seven impacts, the strongest spectral component varies substantially (approximately 450 Hz to 2.3 kHz in a simple short-window analysis). The impacts are short broadband/mechanical transients, with most useful energy below roughly 4 kHz in this sample.

Therefore the first detector should learn/match the **transient envelope and multi-band spectral shape**, not look for one fixed frequency.

## Proposed architecture

### Stage 1 — normalisation
- Decode all sources to mono 44.1 kHz floating-point audio for analysis.
- Preserve the original source audio separately.
- Remove DC offset and optionally apply a gentle 300–500 Hz high-pass filter to reduce pavement/handling rumble.

### Stage 2 — candidate generation
Generate generous candidate transients first, prioritising recall:
- band-pass/high-pass filtered onset envelope
- short-time energy increase
- spectral flux / high-frequency change
- peak-picking with a small refractory period

This stage should intentionally return more candidates than real hits.

### Stage 3 — hit profiler
For each labelled spoke hit, extract a short window around the onset (for example 10 ms before to 40–60 ms after) and calculate several complementary features:
- normalised waveform/envelope correlation
- log-mel or small filter-bank spectrum
- spectral centroid / bandwidth / roll-off
- transient duration / decay
- band-energy ratios
- optional MFCCs

Rather than averaging the seven examples into one rigid template, retain an ensemble of examples and score a candidate by similarity to its nearest/most compatible known hit. This accommodates pitch changes caused by wheel speed and spoke/cable-tie mechanics.

### Stage 4 — classifier / scoring
Start simple and interpretable:
- robust feature scaling
- nearest-neighbour/template similarity or one-class model
- threshold chosen from labelled validation data

Do not start with a neural network. The training set is far too small and a simpler detector will be easier to debug acoustically.

### Stage 5 — temporal sanity checks
Wheel motion gives useful context, but it must be a secondary cue rather than a hard rule:
- suppress duplicate detections a few milliseconds apart
- estimate the recent median inter-hit interval
- allow the interval to change smoothly as the bike accelerates/slows
- use timing regularity to adjust confidence, not to invent missing hits automatically

### Stage 6 — outputs
`detections.csv` fields should initially be:

```text
hit_index,timestamp_seconds,timestamp_hhmmss,score,peak_amplitude,estimated_interval_seconds
```

`review_clicks.wav` should preserve the source and mix in an unmistakable short synthetic click at each detection.

`detected_hits.wav` should concatenate short windows around each detection with a small silence gap between snippets. A companion CSV should retain the original source timestamp for each snippet.

## Training-data recommendation

Seven positive examples are enough to build the first prototype and learn what the signal looks like, but they are **not enough to establish a reliable production threshold**.

The critical missing data is not only more positive hits, but **negative examples**: pavement knocks, handling noise, voices, freewheel/chain sounds, bumps and other impulsive events that could resemble a spoke hit.

Recommended next dataset:
- 30–50 labelled spoke hits across slow, medium and fast wheel speeds
- several different passages/background conditions
- at least 50–100 hard negative transients from the same recordings

A particularly useful workflow is active learning: run the first detector on long audio, review its mistakes, then add false positives and missed true hits to the training set.

## Testing strategy

Use time-based train/validation splits so nearly identical neighbouring hits do not leak into both sets.

Primary metrics:
- precision: what fraction of reported hits are real?
- recall: what fraction of real spoke hits were found?
- F1 score
- timestamp error, with a tolerance such as ±20 ms

For the art workflow, also report total hit count and make manual review fast with the overlaid-click and extracted-hit audio files.

## Development sequence

1. Commit the source recordings and establish the canonical paths.
2. Write an audio inspection script that confirms sample rate/channels and plots onset envelope + spectrogram.
3. Automatically locate the seven known hits in `short_sample/7_clicks.wav` and export their feature profiles.
4. Implement a high-recall candidate detector on the long bike recording.
5. Add template/feature similarity scoring.
6. Export CSV, click-overlay WAV and extracted-hit WAV.
7. Manually label one long passage and calculate precision/recall.
8. Add false positives / missed hits to the training corpus and retune.
9. Only if the simple profiler plateaus, evaluate a lightweight supervised classifier on the accumulated labelled snippets.
