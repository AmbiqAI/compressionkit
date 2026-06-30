"""Compression playbook — a browsable catalog of approaches across tradeoff lanes.

The playbook is a thin layer over the existing building blocks
(``evaluation``, ``dsp``, ``models``). It offers:

* **Lanes** — coarse operating-point archetypes (faithful, clean,
  compact-learned, long-context).
* **A catalog** — declarative :class:`MethodCard` entries, seeded from
  classical codecs and hooked into the golden registry.
* **A run engine** — score any runnable method on a user-supplied signal.

CLI entry point: ``compressionkit playbook ...``.
"""

from __future__ import annotations

from compressionkit.playbook import methods as _methods
from compressionkit.playbook.catalog import (
    Faithfulness,
    MethodCard,
    Status,
    Tier,
    encoder_decoder_params,
    get_method,
    list_methods,
    register_method,
)
from compressionkit.playbook.lanes import LANE_INFO, Lane, LaneInfo
from compressionkit.playbook.run import RunResult, load_signal, run_method_on_signal

__all__ = [
    "LANE_INFO",
    "Faithfulness",
    "Lane",
    "LaneInfo",
    "MethodCard",
    "RunResult",
    "Status",
    "Tier",
    "encoder_decoder_params",
    "get_method",
    "list_methods",
    "load_signal",
    "register_method",
    "run_method_on_signal",
]
