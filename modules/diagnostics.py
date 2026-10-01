"""One switch for the load and inference diagnostics this package prints.

Every ``[vv...]`` line, the ``Load diagnostics:`` summary and the
``GGUF forward diagnostics:`` counters exist to make a slow load or a slow
autoregressive step explainable. None of them are part of producing audio, so
they are silent by default and recoverable with one environment variable:

    VIBEVOICE_DIAGNOSTICS=1     all of them
    VIBEVOICE_RAM_CENSUS=1      only the byte census and the RSS sampler
    VIBEVOICE_VBAR_OBSERVER=1   only the vbar residency observer

Production leaves all three unset. The formatting helpers stay importable and
unconditional -- only the ``logger.info`` that publishes them is gated -- so a
caller that wants the string (a probe, a test) can still ask for it directly.
"""

import os

_FALSEY_ENV = frozenset({"0", "false", "no", "off"})

DIAGNOSTICS_ENV = "VIBEVOICE_DIAGNOSTICS"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _FALSEY_ENV and bool(
        os.environ.get(name, "").strip()
    )


def diagnostics_enabled() -> bool:
    """True when ``VIBEVOICE_DIAGNOSTICS`` is set to a non-falsey value."""
    return _truthy(DIAGNOSTICS_ENV)


def census_enabled() -> bool:
    """True when the byte census and RSS sampler may log.

    On by default under ``VIBEVOICE_DIAGNOSTICS``, independently switchable
    with ``VIBEVOICE_RAM_CENSUS``.
    """
    return diagnostics_enabled() or _truthy("VIBEVOICE_RAM_CENSUS")


def vbar_observer_enabled() -> bool:
    """True when the vbar residency observer may install its patch."""
    return diagnostics_enabled() or _truthy("VIBEVOICE_VBAR_OBSERVER")
