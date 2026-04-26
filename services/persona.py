from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PersonaConfig:
    key: str
    assistant_name: str
    org_noun: str
    app_description: str
    logo_subtitle: str
    ask_label: str
    chat_empty_subtitle: str
    chat_input_placeholder: str
    command_placeholder: str
    settings_org_label: str
    settings_org_placeholder: str
    default_org_name: str
    draft_instruction: str
    normalize_from_terms: List[str]
    chips: List[Dict[str, str]]


PERSONA_MAP: Dict[str, PersonaConfig] = {
    "law_firm": PersonaConfig(
        key="law_firm",
        assistant_name="Leo",
        org_noun="firm",
        app_description="AI operating system for law firms.",
        logo_subtitle="Legal Intelligence",
        ask_label="Ask Leo",
        chat_empty_subtitle="Ask about matters, billing, precedents, or firm operations. Leo uses uploaded documents when relevant.",
        chat_input_placeholder="Ask anything about your firm...",
        command_placeholder="Ask Leo...",
        settings_org_label="Firm Name",
        settings_org_placeholder="Adewale & Partners LLP",
        default_org_name="Your Firm",
        draft_instruction="Use relevant law-firm knowledge where helpful.",
        normalize_from_terms=["this agency", "our agency", "this law firm", "our law firm"],
        chips=[
            {"label": "Practice areas?", "query": "What practice areas do we cover?"},
            {"label": "Billing models?", "query": "What billing models do we use for clients?"},
            {"label": "Matter intake?", "query": "What is our standard matter intake process?"},
            {"label": "Response timelines?", "query": "What response timeline do we promise new clients?"},
        ],
    ),
    "agency": PersonaConfig(
        key="agency",
        assistant_name="Aria",
        org_noun="agency",
        app_description="AI operating system for digital agencies.",
        logo_subtitle="Agency Intelligence",
        ask_label="Ask Aria",
        chat_empty_subtitle="Ask naturally. Aria uses uploaded documents when needed, and handles general questions smoothly.",
        chat_input_placeholder="Ask anything about your agency...",
        command_placeholder="Ask Aria...",
        settings_org_label="Agency Name",
        settings_org_placeholder="Pinnacle Creative Studio",
        default_org_name="Your Agency",
        draft_instruction="Use relevant agency knowledge where helpful.",
        normalize_from_terms=["this agency", "our agency"],
        chips=[
            {"label": "Retainer rates?", "query": "What are our standard retainer rates?"},
            {"label": "Last brand project?", "query": "What did we charge for our last brand project?"},
            {"label": "Website timeline?", "query": "What is our typical website delivery timeline?"},
            {"label": "Our services?", "query": "What services do we offer?"},
        ],
    ),
}

_persona_override: ContextVar[Optional[str]] = ContextVar("persona_override", default=None)


def get_persona_key() -> str:
    override = _persona_override.get()
    if override and override in PERSONA_MAP:
        return override

    raw = (os.environ.get("PERSONA") or "law_firm").strip().lower()
    return raw if raw in PERSONA_MAP else "law_firm"


def set_persona_override(persona_key: Optional[str]) -> Token:
    normalized = (persona_key or "").strip().lower()
    value = normalized if normalized in PERSONA_MAP else None
    return _persona_override.set(value)


def reset_persona_override(token: Token) -> None:
    _persona_override.reset(token)


def get_persona_config() -> PersonaConfig:
    key = get_persona_key()
    config = PERSONA_MAP[key]

    # Allow assistant-name override without changing persona internals.
    override_name = (os.environ.get("ASSISTANT_NAME") or "").strip()
    if override_name:
        return PersonaConfig(
            key=config.key,
            assistant_name=override_name,
            org_noun=config.org_noun,
            app_description=config.app_description,
            logo_subtitle=config.logo_subtitle,
            ask_label=f"Ask {override_name}",
            chat_empty_subtitle=config.chat_empty_subtitle.replace(config.assistant_name, override_name),
            chat_input_placeholder=config.chat_input_placeholder,
            command_placeholder=f"Ask {override_name}...",
            settings_org_label=config.settings_org_label,
            settings_org_placeholder=config.settings_org_placeholder,
            default_org_name=config.default_org_name,
            draft_instruction=config.draft_instruction,
            normalize_from_terms=config.normalize_from_terms,
            chips=config.chips,
        )

    return config


def get_ui_persona_payload() -> dict:
    config = get_persona_config()
    return {
        "persona": config.key,
        "assistant_name": config.assistant_name,
        "org_noun": config.org_noun,
        "logo_subtitle": config.logo_subtitle,
        "ask_label": config.ask_label,
        "chat_empty_subtitle": config.chat_empty_subtitle,
        "chat_input_placeholder": config.chat_input_placeholder,
        "command_placeholder": config.command_placeholder,
        "settings_org_label": config.settings_org_label,
        "settings_org_placeholder": config.settings_org_placeholder,
        "default_org_name": config.default_org_name,
        "draft_instruction": config.draft_instruction,
        "normalize_from_terms": config.normalize_from_terms,
        "chips": config.chips,
    }
