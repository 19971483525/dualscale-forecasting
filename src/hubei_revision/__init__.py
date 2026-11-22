"""Audited experiments for next-day daily electricity consumption forecasting."""

from pathlib import Path
import sys

# Optional offline wheel contents are local-only and excluded from Git. Public
# installations use requirements.txt and do not need the offline directory.
_offline = Path(__file__).resolve().parents[2] / ".local_deps"
if _offline.exists():
    sys.path.insert(0, str(_offline))

__version__ = "0.1.0"
