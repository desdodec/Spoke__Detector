#!/usr/bin/env python3
"""Art-direction variation helpers for the Spoke Detector design studio.

This module deliberately changes presentation only.  It extends the palette set
and creates stronger, reproducible parameter variations while leaving the locked
180-interval participant data untouched.
"""
from __future__ import annotations

import random
from dataclasses import asdict

import design_studio as studio

EXTRA_PALETTES: dict[str, tuple[str, str, str, str]] = {
    "Bauhaus Signal": ("#e6392f", "#f2c230", "#1f5aa6", "#f5efe0"),
    "Cobalt Orange": ("#1455d9", "#ff6b19", "#7ec8ff", "#fff0dc"),
    "Acid Orchid": ("#d9ff00", "#f02eff", "#00e6ff", "#fff9d8"),
    "Hot Cold": ("#ff3355", "#00c8ff", "#ffd64a", "#f8ffff"),
    "Pink Fern": ("#ff4f9a", "#1f8f5f", "#ffd7e8", "#f1ffe9"),
    "Purple Citrus": ("#6f32d4", "#ffd51f", "#ff7a1a", "#fbf4ff"),
    "Cherry Cream": ("#c5152f", "#fff0d7", "#2b171c", "#f7b9a8"),
    "Ink Vermilion": ("#101318", "#ff3b21", "#f0d9a8", "#ffffff"),
    "Jewel Box": ("#0d6b63", "#8f2bb3", "#d8a117", "#f9e7ff"),
    "Peacock": ("#0a6975", "#1fc7a5", "#7b2cbf", "#f6d85d"),
    "Night Poppy": ("#17122b", "#ff315f", "#ff9f1c", "#fff4e6"),
    "Ice Candy": ("#68d8ff", "#ff9bd5", "#a98cff", "#f5ffff"),
    "Pastel Voltage": ("#ffb3c8", "#9ee493", "#7dc4ff", "#fff09e"),
    "Desert Modern": ("#8b3a2b", "#e3a54b", "#315b54", "#f6ead7"),
    "Moss Rose": ("#36543d", "#d27b83", "#c2a862", "#f0eadc"),
    "Copper Teal": ("#b85f36", "#1c8a8a", "#e3b76b", "#f2f1e8"),
    "Royal Sport": ("#112f91", "#f2d133", "#ef3340", "#ffffff"),
    "Black Gold Red": ("#111111", "#d7ad3a", "#d6262b", "#f7f0df"),
    "Emerald Coral": ("#007f68", "#ff6f61", "#f4c95d", "#f7fff6"),
    "Laser Lemon": ("#00f5d4", "#fee440", "#f15bb5", "#f9f9ff"),
}


def install() -> None:
    """Extend the studio palette registry in-place."""
    studio.PALETTES.update(EXTRA_PALETTES)


def _copy(p: studio.DesignParams) -> studio.DesignParams:
    return studio.DesignParams(**asdict(p)).validated()


def _seeded(seed: int | None) -> tuple[int, random.Random]:
    value = random.SystemRandom().randrange(1, 2**31 - 1) if seed is None else int(seed)
    return value, random.Random(value)


def _stroke_from_zone(rng: random.Random, wild: bool = False) -> float:
    """Choose from deliberate art-direction zones rather than a bland uniform range."""
    if wild:
        zone = rng.choices(("fine", "graphic", "heavy"), weights=(15, 30, 55), k=1)[0]
    else:
        zone = rng.choices(("fine", "graphic", "heavy"), weights=(22, 43, 35), k=1)[0]
    if zone == "fine":
        return round(rng.uniform(0.35, 1.25), 2)
    if zone == "graphic":
        return round(rng.uniform(1.35, 3.0), 2)
    return round(rng.uniform(3.1, 5.0), 2)


def randomise_design(current: studio.DesignParams, *, keep_family: bool = True, seed: int | None = None) -> studio.DesignParams:
    """Full curated variation: form treatment, colour and mark weight."""
    install()
    seed, rng = _seeded(seed)
    p = _copy(current)
    if not keep_family:
        p.family = rng.choice(tuple(studio.FAMILIES))
    p.palette = rng.choice(tuple(studio.PALETTES))
    p.stroke_width = _stroke_from_zone(rng)
    p.detail = rng.choice((rng.randint(36, 75), rng.randint(80, 145), rng.randint(150, 235)))
    p.jitter = round(rng.choice((rng.uniform(0.01, 0.12), rng.uniform(0.13, 0.38), rng.uniform(0.40, 0.78))), 3)
    p.symmetry = rng.choice((rng.randint(2, 5), rng.randint(6, 10), rng.randint(11, 16)))
    p.scale = round(rng.uniform(0.68, 1.17), 3)
    p.rotation_deg = round(rng.uniform(0, 360), 1)
    p.opacity = round(rng.uniform(0.55, 1.0), 3)
    p.seed = seed
    return p.validated()


def randomise_form(current: studio.DesignParams, *, seed: int | None = None) -> studio.DesignParams:
    """Change geometry/treatment while keeping family, palette and background."""
    seed, rng = _seeded(seed)
    p = _copy(current)
    palette, background, family = p.palette, p.background, p.family
    p.stroke_width = _stroke_from_zone(rng)
    p.detail = rng.randint(35, 235)
    p.jitter = round(rng.uniform(0.0, 0.72), 3)
    p.symmetry = rng.randint(2, 16)
    p.scale = round(rng.uniform(0.65, 1.18), 3)
    p.rotation_deg = round(rng.uniform(0, 360), 1)
    p.opacity = round(rng.uniform(0.5, 1.0), 3)
    p.seed = seed
    p.palette, p.background, p.family = palette, background, family
    return p.validated()


def randomise_colour(current: studio.DesignParams, *, seed: int | None = None) -> studio.DesignParams:
    """Change palette only, preserving the current form and seed geometry."""
    install()
    seed, rng = _seeded(seed)
    p = _copy(current)
    choices = [name for name in studio.PALETTES if name != p.palette]
    if choices:
        p.palette = rng.choice(choices)
    return p.validated()


def go_wild(current: studio.DesignParams, *, keep_family: bool = False, seed: int | None = None) -> studio.DesignParams:
    """Deliberately favour extreme, poster-like outcomes."""
    install()
    seed, rng = _seeded(seed)
    p = _copy(current)
    if not keep_family:
        p.family = rng.choice(tuple(studio.FAMILIES))
    p.palette = rng.choice(tuple(studio.PALETTES))
    p.stroke_width = _stroke_from_zone(rng, wild=True)
    p.detail = rng.choice((rng.randint(24, 55), rng.randint(185, 240)))
    p.jitter = round(rng.choice((rng.uniform(0.0, 0.08), rng.uniform(0.55, 1.0))), 3)
    p.symmetry = rng.choice((1, 2, 3, 4, 12, 14, 16))
    p.scale = round(rng.choice((rng.uniform(0.58, 0.74), rng.uniform(1.06, 1.2))), 3)
    p.rotation_deg = round(rng.uniform(0, 360), 1)
    p.opacity = round(rng.choice((rng.uniform(0.35, 0.58), rng.uniform(0.9, 1.0))), 3)
    p.seed = seed
    return p.validated()


install()
