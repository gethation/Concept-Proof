"""Put scripts/ on sys.path so `import lib.x` works from any subdirectory.

Running `python scripts/features/spread.py` puts scripts/features on sys.path,
not scripts/, so a subdirectory entry point cannot see the lib package without
help. Every entry point starts with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

which is the same trick the flat layout already used, just pointed one level up.
This module is what that trick buys: once scripts/ is importable, everything
else is an ordinary package import.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def ensure_on_path() -> Path:
    """Idempotently add scripts/ to sys.path. Returns the directory added."""
    entry = str(SCRIPTS_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return SCRIPTS_DIR
