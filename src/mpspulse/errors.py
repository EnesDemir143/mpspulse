"""mpspulse error types (torch-free)."""


class MpsError(Exception):
    """Base error for mpspulse."""


class MpsUnavailableError(MpsError):
    """Raised when Apple Silicon MPS is required but unavailable.

    Never fall back to CPU silently: CPU timing would corrupt the
    GPU-event / wall-clock separation that every report depends on.
    """
