"""The tool layer: schema reflection, the tool contract, and the registry.

Submodules are not imported eagerly so that importing :mod:`cagent.tools` stays
cheap and so the concrete tool modules can import :mod:`cagent.tools.base`
without a cycle back through this package.
"""

from __future__ import annotations

__all__: list[str] = []
