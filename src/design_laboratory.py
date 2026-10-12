#!/usr/bin/env python3
"""Art-directed design laboratory for Spoke Detector movement portraits.

The laboratory deliberately keeps the measurement layer fixed. Every candidate
uses the same locked central interval slice produced by the study workflow.
The default artwork source is the first/outward leg. The same central slice of
the return leg is retained for paired movement-rhythm analysis.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import study_visualizer as visual

LAB_STYLES = (
    "Radial fingerprint",
    "Spoke-wheel barcode",
    "Concentric rhythm rings",
    "36-spoke aggregate glyph",
    "Woven stripes",
    "Recurrence matrix",
    "Grid weave",
)

ART_DIRECTIONS: dict[str, dict[str, Any]] = {
    "Quiet Luxury": {
        "description": "Restrained, spacious and premium. The data form is isolated and allowed to breathe.",
        "palettes": ("Gold on Black", "Ivory & Navy", "Silver Graphite"),
        "theme": "Dark garment",
        "density": "Low",
        "smoothing": "Medium",
        "styles": ("Radial fingerprint", "36-spoke aggregate glyph", "Concentric rhythm rings"),
    },
    "Textile Studio": {
        "description": "Layered, tactile and pattern-led. The movement sequence behaves like a woven material.",
        "palettes": ("Clay & Sand", "Moss & Stone", "Ochre & Slate"),
        "theme": "Light garment",
        "density": "High",
        "smoothing": "Low",
        "styles": ("Woven stripes", "Grid weave", "Recurrence matrix"),
    },
    "Electric Pulse": {
        "description": "High-energy, luminous and contemporary. Colour makes rhythm and recurrence feel alive.",
        "palettes": ("Neon Pulse", "Electric Violet", "Acid Blue"),
        "theme": "Dark garment",
        "density": "High",
        "smoothing": "Low",
        "styles": ("Spoke-wheel barcode", "Recurrence matrix", "Grid weave"),
    },
    "Modernist Poster": {
        "description": "Bold geometry, hard hierarchy and deliberate negative space. Designed to read at distance.",
        "palettes": ("Blue Gold", "Ivory & Navy", "Clay & Sand"),
        "theme": "Light garment",
        "density": "Medium",
        "smoothing": "Medium",
        "styles": ("Radial fingerprint", "Spoke-wheel barcode", "36-spoke aggregate glyph"),
    },
    "Monochrome Etching": {
        "description": "One-colour, line-led and screen-print friendly. Fine structure rewards close viewing.",
        "palettes": ("Ink", "Chalk", "Graphite"),
        "theme": "Transparent",
        "density": "High",
        "smoothing": "Low",
        "styles": ("Radial fingerprint", "Concentric rhythm rings", "Woven stripes"),
    },
    "Chromatic Field": {
        "description": "Immersive colour fields for matrix-based data. Best for large back prints and poster-scale work.",
        "palettes": ("Viridis", "Plasma", "Blue Gold"),
        "theme": "Dark garment",
        "density": "High",
        "smoothing": "Medium",
        "styles": ("Recurrence matrix", "Grid weave", "Woven stripes"),
    },
}

RUBRIC = (
    "Impact",
    "Identity",
    "Data necessity",
    "Concept",
    "Humanity",
    "Integrity",
    "Composition",
    "Colour",
    "Materiality",
    "Detail",
    "Restraint",
    "Memorability",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    art_direction: str
    style: str
    palette: str
    theme: str
    density: str
    smoothing: str
    source: str = "Baseline / first leg"


def build_candidates(limit: int = 24) -> list[Candidate]:
    """Build a deterministic, curated spread rather than random preset spam."""
    candidates: list[Candidate] = []
    for direction, spec in ART_DIRECTIONS.items():
        styles = spec["styles"]
        palettes = spec["palettes"]
        # Rotate palette assignment so adjacent candidates differ visibly.
        for i, style in enumerate(styles):
            palette = palettes[i % len(palettes)]
            candidates.append(
                Candidate(
                    name=f"{direction} · {style}",
                    art_direction=direction,
                    style=style,
                    palette=palette,
                    theme=spec["theme"],
                    density=spec["density"],
                    smoothing=spec["smoothing"],
                )
            )
            # Add a second colourway for selected visually strong families.
            if style in {"Radial fingerprint", "Spoke-wheel barcode", "Recurrence matrix", "Grid weave", "Woven stripes"}:
                candidates.append(
                    Candidate(
                        name=f"{direction} · {style} · Alt",
                        art_direction=direction,
                        style=style,
                        palette=palettes[(i + 1) % len(palettes)],
                        theme=spec["theme"],
                        density=spec["density"],
                        smoothing=spec["smoothing"],
                    )
                )
    # Preserve coverage across all directions before truncating.
    return candidates[:max(1, int(limit))]


def paired_leg_analysis(data: visual.VisualData) -> dict[str, Any]:
    """Compare the same locked central slice from outward and return legs.

    This is a wheel-motion / forward-movement rhythm comparison, not direct
    heel-strike, toe-off, or clinical gait-event measurement.
    """
    out = np.asarray(data.outward_intervals, dtype=float)
    ret = np.asarray(data.return_intervals, dtype=float)
    n = min(len(out), len(ret))
    out = out[:n]
    ret = ret[:n]

    out_med = float(np.median(out))
    ret_med = float(np.median(ret))
    out_rate = 1.0 / out
    ret_rate = 1.0 / ret

    out_norm = (out - out_med) / max(out_med, 1e-12)
    ret_norm = (ret - ret_med) / max(ret_med, 1e-12)
    corr = float(np.corrcoef(out_norm, ret_norm)[0, 1]) if n > 2 else float("nan")
    paired_abs = np.abs(out_norm - ret_norm)

    return {
        "intervals_per_leg": n,
        "equivalent_revolutions_per_leg": n / max(1, data.spoke_count),
        "outward_median_click_rate_hz": float(np.median(out_rate)),
        "return_median_click_rate_hz": float(np.median(ret_rate)),
        "return_vs_outward_rate_change_percent": (float(np.median(ret_rate)) / max(float(np.median(out_rate)), 1e-12) - 1.0) * 100.0,
        "outward_robust_cv": data.outward_cv,
        "return_robust_cv": data.return_cv,
        "paired_normalised_difference_median": float(np.median(paired_abs)),
        "paired_normalised_difference_p90": float(np.quantile(paired_abs, 0.90)),
        "sequence_correlation": corr,
        "interpretation": (
            "Both legs use the same central fixed-count slice, so the comparison covers the same wheel-travel equivalent. "
            "These values describe wheel-motion / forward-movement rhythm and should be treated as a gait proxy rather than direct foot-event timing."
        ),
    }


def analysis_text(data: visual.VisualData) -> str:
    a = paired_leg_analysis(data)
    out_cv = "n/a" if a["outward_robust_cv"] is None else f"{100*a['outward_robust_cv']:.1f}%"
    ret_cv = "n/a" if a["return_robust_cv"] is None else f"{100*a['return_robust_cv']:.1f}%"
    corr = a["sequence_correlation"]
    corr_text = "n/a" if not np.isfinite(corr) else f"{corr:+.2f}"
    return "\n".join([
        "PAIRED MOVEMENT-RHYTHM ANALYSIS",
        f"Central slice: {a['intervals_per_leg']} intervals per leg = {a['equivalent_revolutions_per_leg']:.1f} wheel revolutions per leg",
        f"Outward median rhythm: {a['outward_median_click_rate_hz']:.2f} clicks/s",
        f"Return median rhythm: {a['return_median_click_rate_hz']:.2f} clicks/s",
        f"Return vs outward rhythm change: {a['return_vs_outward_rate_change_percent']:+.1f}%",
        f"Robust variability: outward {out_cv} / return {ret_cv}",
        f"Median paired pattern difference: {100*a['paired_normalised_difference_median']:.1f}% of local median interval",
        f"90th percentile paired difference: {100*a['paired_normalised_difference_p90']:.1f}%",
        f"Sequence-shape correlation: {corr_text}",
        "",
        a["interpretation"],
    ])


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_caption_panel(candidate: Candidate, width: int, height: int) -> Image.Image:
    panel = Image.new("RGBA", (width, height), (18, 18, 18, 255))
    d = ImageDraw.Draw(panel)
    d.text((20, 14), candidate.art_direction.upper(), fill=(245, 245, 245, 255), font=_font(22))
    d.text((20, 48), candidate.style, fill=(220, 220, 220, 255), font=_font(18))
    d.text((20, 78), candidate.palette, fill=(155, 155, 155, 255), font=_font(14))
    return panel


def generate_laboratory(output_dir: Path, candidate_limit: int = 24) -> dict[str, Any]:
    """Generate a design-lab folder with PNG/SVG candidates, contact sheet and metadata."""
    output_dir = Path(output_dir)
    data = visual.load_visual_data(output_dir)
    candidates = build_candidates(candidate_limit)
    lab_dir = output_dir / "design_lab"
    lab_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    thumbs: list[tuple[Candidate, Image.Image]] = []
    for idx, c in enumerate(candidates, 1):
        stem = f"{idx:02d}_{c.art_direction}_{c.style}_{c.palette}".lower()
        stem = stem.replace(" ", "_").replace("/", "_").replace("&", "and").replace("·", "_").replace("-", "_")
        png = lab_dir / f"{stem}.png"
        svg = lab_dir / f"{stem}.svg"
        visual.export_png(
            png, data, c.style, theme=c.theme, density=c.density, smoothing=c.smoothing,
            annotated=False, transparent=False, source=c.source, palette_name=c.palette,
            width=1200, height=1200,
        )
        visual.export_svg(
            svg, data, c.style, theme=c.theme, density=c.density, smoothing=c.smoothing,
            annotated=False, transparent=False, source=c.source, palette_name=c.palette,
            width=1600, height=1600,
        )
        im = Image.open(png).convert("RGBA")
        im.thumbnail((360, 360))
        thumbs.append((c, im.copy()))
        records.append({
            "index": idx,
            "name": c.name,
            "art_direction": c.art_direction,
            "art_direction_meaning": ART_DIRECTIONS[c.art_direction]["description"],
            "style": c.style,
            "style_meaning": visual.STYLE_HELP.get(c.style, ""),
            "palette": c.palette,
            "theme": c.theme,
            "density": c.density,
            "smoothing": c.smoothing,
            "source": c.source,
            "png": png.name,
            "svg": svg.name,
            "rubric": {criterion: None for criterion in RUBRIC},
        })

    tile_w, art_h, caption_h = 400, 380, 120
    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (tile_w * cols, (art_h + caption_h) * rows), (238, 236, 230, 255))
    for i, (candidate, im) in enumerate(thumbs):
        col, row = i % cols, i // cols
        x = col * tile_w + (tile_w - im.width) // 2
        y = row * (art_h + caption_h) + (art_h - im.height) // 2
        sheet.alpha_composite(im, (x, y))
        caption = _render_caption_panel(candidate, tile_w - 20, caption_h - 10)
        sheet.alpha_composite(caption, (col * tile_w + 10, row * (art_h + caption_h) + art_h))
    sheet_path = lab_dir / "design_laboratory_contact_sheet.png"
    sheet.convert("RGB").save(sheet_path, quality=94)

    paired = paired_leg_analysis(data)
    manifest = {
        "participant_id": data.participant_id,
        "trial_id": data.trial_id,
        "design_principle": "One fixed central baseline slice creates the artwork; the same-count central return slice is retained for paired movement-rhythm analysis.",
        "locked_intervals_per_leg": data.intervals_per_leg,
        "spoke_count": data.spoke_count,
        "equivalent_revolutions_per_leg": data.intervals_per_leg / max(1, data.spoke_count),
        "candidate_count": len(records),
        "candidates": records,
        "paired_movement_analysis": paired,
        "rubric": list(RUBRIC),
    }
    manifest_path = lab_dir / "design_laboratory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    analysis_path = lab_dir / "paired_movement_analysis.txt"
    analysis_path.write_text(analysis_text(data), encoding="utf-8")

    return {
        "lab_dir": lab_dir,
        "contact_sheet": sheet_path,
        "manifest": manifest_path,
        "paired_analysis": analysis_path,
        "candidate_count": len(records),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate an art-directed design laboratory from a completed study output.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    result = generate_laboratory(args.output_dir, args.count)
    print(f"Generated {result['candidate_count']} candidates")
    print(result["contact_sheet"])
