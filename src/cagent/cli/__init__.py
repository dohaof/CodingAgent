"""The terminal interface: argument parsing, rendering, and the REPL.

Intentionally free of eager imports. Pulling ``app`` in here would both slow
startup and make ``python -m cagent.cli.app`` re-import a module already in
``sys.modules``, which Python warns about.
"""

from __future__ import annotations
