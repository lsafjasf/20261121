"""Exceptions used by the local renderer."""


class LocalRenderError(Exception):
    """Base class for renderer errors."""


class MissingHandleError(LocalRenderError):
    """Raised when a transition needs media frames the source cannot provide.

    ``details`` contains one human readable string per failed requirement so
    the CLI can print every problem in one pass.
    """

    def __init__(self, details):
        self.details = list(details)
        super().__init__("insufficient media handle(s) for transition(s)")
