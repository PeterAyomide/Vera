"""Authentication & security middleware for AgencyOS.

Provides:
  • X-API-Key header validation using hmac.compare_digest (timing-safe).
  • Webhook signature verification (HMAC-SHA256).
  • Input sanitization helpers.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

load_dotenv()
logger = logging.getLogger(__name__)

# ── API Key ──────────────────────────────────────────────────────────────────

API_KEY: str = os.environ.get("AGENCYOS_API_KEY", "")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: Optional[str] = Security(_api_key_header),
) -> str:
    """FastAPI dependency – rejects requests without a valid API key.

    Uses ``hmac.compare_digest`` to prevent timing attacks.
    """
    if not API_KEY:
        # If no key is configured, allow all requests (dev mode)
        logger.warning("AGENCYOS_API_KEY not set – auth is DISABLED (dev mode)")
        return "dev-mode"

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    if not hmac.compare_digest(api_key.encode(), API_KEY.encode()):
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API key.")

    return api_key


# ── Webhook Signature Verification ───────────────────────────────────────────

WEBHOOK_SECRET: str = os.environ.get("WEBHOOK_SECRET", "")


async def verify_webhook_signature(request: Request) -> bytes:
    """Verify the HMAC-SHA256 signature on incoming webhook payloads.

    Expects headers:
      • X-Webhook-Signature: hex-encoded HMAC-SHA256 of the raw body.

    Returns the raw body bytes so the caller can parse them.
    """
    body = await request.body()

    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not set – signature verification DISABLED")
        return body

    signature = request.headers.get("X-Webhook-Signature", "")
    if not signature:
        raise HTTPException(status_code=401, detail="Missing X-Webhook-Signature header.")

    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        logger.warning("Webhook signature mismatch")
        raise HTTPException(status_code=403, detail="Invalid webhook signature.")

    return body


# ── Input Sanitization ───────────────────────────────────────────────────────

# Pattern that matches common SQL injection characters/sequences
_SQL_INJECTION_PATTERN = re.compile(
    r"(--|;|'|\"|\b(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC|UNION|SELECT)\b)",
    re.IGNORECASE,
)


def sanitize_search_input(value: str | None) -> str | None:
    """Sanitize search input to prevent SQL injection via ilike patterns.

    - Strips dangerous SQL fragments.
    - Escapes Postgres LIKE wildcards (%, _) that the user didn't intend.
    - Limits length to 200 characters.
    """
    if not value:
        return value

    # Truncate
    value = value[:200].strip()

    # Remove dangerous sequences
    value = _SQL_INJECTION_PATTERN.sub("", value)

    # Escape Postgres LIKE special chars so they're treated as literals
    value = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    return value.strip()
