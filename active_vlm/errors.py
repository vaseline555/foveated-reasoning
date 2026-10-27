"""Failures that must stop the shared worker rather than one model trajectory."""


class IntegrityError(RuntimeError):
    """Frozen inputs, a cached receipt or exact resume replay no longer match."""


class HarnessError(RuntimeError):
    """Unexpected implementation failure; invalidate the entire strict study."""
