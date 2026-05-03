"""Public paradigm extension API for DESBench."""

from __future__ import annotations

from .v2.authority_modes import (
    AuthorityMode,
    get_authority_mode,
    list_authority_modes,
    register_authority_mode,
)

__all__ = [
    "AuthorityMode",
    "get_authority_mode",
    "list_authority_modes",
    "register_authority_mode",
]
