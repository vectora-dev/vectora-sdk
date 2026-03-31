from __future__ import annotations

from datetime import datetime, timezone
from secrets import token_hex
import re

TRACE_ID_PATTERN = re.compile(r"^vct_\d{8}_[a-z0-9]{4,}$", re.IGNORECASE)


def generate_trace_id() -> str:
    date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"vct_{date_prefix}_{token_hex(2)}"


def is_valid_trace_id(value: str) -> bool:
    return bool(TRACE_ID_PATTERN.fullmatch(value))


def isValidTraceId(value: str) -> bool:
    return is_valid_trace_id(value)
