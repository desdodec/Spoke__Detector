#!/usr/bin/env python3
"""Scrub-enabled Click Review UI.

This deliberately subclasses the accepted click-review app so the existing UI and
workflow remain intact while adding transport-style scrubbing and a persistent
playback cursor.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from transient_lab_app import App as BaseApp


class App(BaseApp):
    def __init__(self) -> None:
        # BaseApp creates the Tk root before it dispatches to our _build().
        # Tk variables therefore have to be created in _build(), not before
        # BaseApp.__init__(), otherwise Python 3.13 raises "Too early to create
        # variable: no default root window".
        self._scrubbing = False
        self._resume_after_scrub = False
        super().__init__()

    def _build(self) -> None:
        # At this point the Tk root exists, so these variables are safe to make.
        self.scrub_time = tk.DoubleVar(master=self, value=0.0)
        self.timecode = tk.StringVar(master=self, value="0.000 / 0.000 s")

        super()._build()

        # Keep the accepted layout intact. The scrubber sits directly beneath the
        # existing horizontal view scrollbar and above the transport controls.
        stage = self.scroll.master
        scrub = ttk.Frame(stage, style="Card.TFrame", padding=(8, 4, 8, 6))
        scrub.grid(row=2, column=0, sticky="ew")
        scrub.columnconfigure(1, weight=1)

        ttk.Label(scrub, text="SCRUB", style="CardMuted.TLabel").grid(row=0, column=0, padx=(0, 8))
        self.scrub_scale = ttk.Scale(
            scrub,
            from_=0.0,
            to=1.0,
            variable=self.scrub_time,
            orient="horizontal",
            style="Clean.Horizontal.TScale",
            command=self._scrub_drag,
        )
        self.scrub_scale.grid(row=0, column=1, sticky="ew")
        self.scrub_scale.bind("<ButtonPress-1>", self._scrub_start, add="+")
        self.scrub_scale.bind("<ButtonRelease-1>", self._scrub_end, add="+")
        ttk.Label(scrub, textvariable=self.timecode, style="Card.TLabel").grid(row=0, column=2, padx=(10, 0))

        # A click in the ruler area seeks the playhead without disturbing the
        # drag-to-select interaction in the waveform itself.
        self.canvas.bind("<Button-1>", self._canvas_seek_or_select, add="+")

        # Keep keyboard transport familiar.
        self.bind("<Home>", lambda _e: self.seek_to(0.0))
        self.bind("<End>", lambda _e: self.seek_to(self.audio.duration_s if self.audio else 0.0))

    def load_path(self, path) -> None:
        super().load_path(path)
        if self.audio is None:
            return
        self.scrub_scale.configure(to=max(0.001, self.audio.duration_s))
        self.playhead_t = 0.0
        self.scrub_time.set(0.0)
        self._sync_timecode()
        self.draw()

    def _canvas_seek_or_select(self, event) -> None:
        # Top ruler strip behaves as a transport ruler: click to place cursor.
        # Everywhere else keeps the existing selection behaviour.
        if self.audio is not None and event.y <= 30:
            self.dragging = False
            self.sel_start = self.sel_end = None
            self.seek_to(self._x_to_time(event.x))

    def _scrub_start(self, _event=None) -> None:
        if self.audio is None:
            return
        self._scrubbing = True
        self._resume_after_scrub = self.playing
        if self.playing:
            self.playhead_t = self._current_play_time()
            self._stop_backend()
            self.playing = False
            self.paused = True

    def _scrub_drag(self, value) -> None:
        if self.audio is None:
            return
        t = max(0.0, min(self.audio.duration_s, float(value)))
        self.playhead_t = t
        self.scrub_time.set(t)
        self._keep_playhead_visible(t)
        self._sync_timecode()
        self.draw()

    def _scrub_end(self, _event=None) -> None:
        if self.audio is None:
            return
        t = max(0.0, min(self.audio.duration_s, float(self.scrub_time.get())))
        self._scrubbing = False
        should_resume = self._resume_after_scrub
        self._resume_after_scrub = False
        self.seek_to(t, resume=should_resume)

    def seek_to(self, t: float, resume: bool | None = None) -> None:
        if self.audio is None:
            return
        t = max(0.0, min(self.audio.duration_s, float(t)))
        was_playing = self.playing if resume is None else bool(resume)

        self._stop_backend()
        self.playing = False
        self.paused = True
        self.playhead_t = t
        self.play_anchor_t = t

        # Seeking should not unexpectedly shorten playback because an old
        # selection happened to exist. Continue to the current playback end when
        # valid, otherwise to the end of the file.
        if self.play_end_t <= t or self.play_end_t > self.audio.duration_s:
            self.play_end_t = self.audio.duration_s
        self.play_start_t = t

        self.scrub_time.set(t)
        self._keep_playhead_visible(t)
        self._sync_timecode()
        self.status.set(f"Cursor {t:.3f} s")
        self.draw()

        if was_playing and self.play_end_t > t and self._start_backend(t, self.play_end_t):
            self.play_anchor_t = t
            self.play_wall = time.monotonic()
            self.playing = True
            self.paused = False
            self.status.set(f"Playing from {t:.3f} s")
            self._schedule_play_tick()

    def _keep_playhead_visible(self, t: float) -> None:
        if self.audio is None:
            return
        span = self.view_end - self.view_start
        if span >= self.audio.duration_s:
            return
        if self.view_start <= t <= self.view_end:
            return
        start = t - 0.5 * span
        start = min(max(0.0, start), max(0.0, self.audio.duration_s - span))
        self.view_start, self.view_end = start, start + span
        self._update_scrollbar()

    def _sync_timecode(self) -> None:
        total = self.audio.duration_s if self.audio is not None else 0.0
        current = self.playhead_t if self.playhead_t is not None else 0.0
        self.timecode.set(f"{current:.3f} / {total:.3f} s")

    def play(self) -> None:
        # If a cursor has been deliberately positioned, Play starts there unless
        # the user has an active selection. This makes the scrubber feel like a
        # conventional transport rather than just a display.
        if self.audio is None:
            return
        if not self.paused and self.playhead_t is not None and self.selection() is None:
            t = max(0.0, min(self.audio.duration_s, self.playhead_t))
            if t < self.audio.duration_s:
                self.play_start_t = t
                self.play_end_t = self.audio.duration_s
                self.paused = True
        super().play()
        self._sync_timecode()

    def pause(self) -> None:
        super().pause()
        if self.playhead_t is not None:
            self.scrub_time.set(self.playhead_t)
        self._sync_timecode()

    def stop(self) -> None:
        super().stop()
        if hasattr(self, "scrub_time"):
            t = self.playhead_t if self.playhead_t is not None else 0.0
            self.scrub_time.set(t)
            self._sync_timecode()

    def _play_tick(self) -> None:
        super()._play_tick()
        if self.playhead_t is not None and not self._scrubbing:
            self.scrub_time.set(self.playhead_t)
            self._sync_timecode()

    def draw(self) -> None:
        super().draw()
        if self.audio is None or self.playhead_t is None:
            return

        # Reinforce the playback cursor so it remains visually dominant over
        # selection fills, event markers and hover guides.
        if self.view_start <= self.playhead_t <= self.view_end:
            x = self._time_to_x(self.playhead_t)
            h = max(10, self.canvas.winfo_height())
            self.canvas.create_line(x, 0, x, h, fill="#1575c7", width=3)
            self.canvas.create_polygon(
                x - 6, 0,
                x + 6, 0,
                x, 8,
                fill="#1575c7",
                outline="",
            )


if __name__ == "__main__":
    App().mainloop()
