"""Vera — Chat session service.

Handles persistent chat session storage in Supabase.
Sessions and messages survive server restarts.
"""

from __future__ import annotations
import logging
import uuid
from typing import List, Optional

from services.db import supabase

logger = logging.getLogger(__name__)


def create_session(title: str = "New conversation") -> dict:
    """Create a new chat session. Returns the session row."""
    result = (
        supabase.table("chat_sessions")
        .insert({"id": str(uuid.uuid4()), "title": title})
        .execute()
    )
    return result.data[0] if result.data else {}


def list_sessions(limit: int = 30) -> List[dict]:
    """Return recent sessions ordered by last activity."""
    result = (
        supabase.table("chat_sessions")
        .select("*")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages (CASCADE handles messages)."""
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()


def update_session_title(session_id: str, title: str) -> None:
    """Update the title of a session (use first user message, truncated)."""
    try:
        supabase.table("chat_sessions").update(
            {"title": title[:80], "updated_at": "now()"}
        ).eq("id", session_id).execute()
    except Exception as e:
        logger.warning("Could not update session title: %s", e)


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[str]] = None,
) -> None:
    """Persist a single chat message."""
    try:
        supabase.table("chat_messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "sources": sources or [],
        }).execute()
        # Bump session updated_at so it floats to the top of the list
        supabase.table("chat_sessions").update(
            {"updated_at": "now()"}
        ).eq("id", session_id).execute()
    except Exception as e:
        logger.warning("Could not save message: %s", e)


def load_messages(session_id: str) -> List[dict]:
    """Load all messages for a session in chronological order."""
    result = (
        supabase.table("chat_messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return result.data or []
