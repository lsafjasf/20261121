"""User-facing errors raised by the handoff tool."""


class HandoffError(Exception):
    """Base class for expected, explainable failures."""


class ConfigError(HandoffError):
    """The version configuration is malformed or inconsistent."""


class WavError(HandoffError):
    """The source or room-tone WAV cannot be processed safely."""


class ManifestError(HandoffError):
    """A replacement manifest cannot be verified."""
