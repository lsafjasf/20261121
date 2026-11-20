"""Desktop narrowband noise repair toolkit.

Modules:
    dsp      -- STFT/ISTFT with end padding and bypass error metrics.
    strokes  -- freehand stroke -> soft time-frequency mask rasterization.
    synth    -- synthetic test signals (crossed chirps, transients, hum).
    verify   -- headless numerical verification scenarios.
    app      -- PySide6 desktop GUI with synced A/B playback.
"""

__version__ = "0.1.0"
