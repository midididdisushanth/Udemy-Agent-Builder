"""
api/config.py — Application configuration (top-level entry for main.py).
Delegates to core/config.py — kept here for import convenience.
"""
from api.core.config import get_settings, Settings

__all__ = ["get_settings", "Settings"]
