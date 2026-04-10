"""Unified LLM client — routes through Groq → OpenAI → Anthropic.

All hooks should call ``call_llm()`` instead of directly importing anthropic/openai.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# Groq settings
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.3-70b-versatile"

# OpenAI settings
_OPENAI_MODEL = "gpt-4o"

# Anthropic fallback
_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Degradation tracking — set by call_llm, read by orchestrator
last_fallback_used: Optional[str] = None  # 'openai' | 'anthropic' | 'none_available' | None


def call_llm(
    prompt: str,
    max_tokens: int = 800,
    system: Optional[str] = None,
    expect_json: bool = False,
) -> Optional[str]:
    """Send a prompt to the best available LLM and return the text response.

    Resolution order:
      1. GROQ_API_KEY   → OpenAI-compatible endpoint (Llama 3.3 70B)
      2. LLM_API_KEY    → OpenAI API (GPT-4o)
      3. ANTHROPIC_API_KEY → Anthropic Messages API (Claude Sonnet)
      4. None set → return None

    Parameters
    ----------
    prompt : str
        User message content.
    max_tokens : int
        Maximum tokens in the response.
    system : str | None
        Optional system message.
    expect_json : bool
        If True, requests structured JSON output (``response_format``).

    Returns
    -------
    str | None
        The model's text reply, or None on failure / no key available.
    """
    global last_fallback_used
    last_fallback_used = None

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    openai_key = os.environ.get("LLM_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    # ── Try Groq first ────────────────────────────────────────────────────
    if groq_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=groq_key, base_url=_GROQ_BASE_URL, timeout=60.0)
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict = dict(
                model=_GROQ_MODEL,
                max_tokens=max_tokens,
                messages=messages,
            )
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}

            log.info("call_llm: sending to Groq (%s), prompt=%d chars", _GROQ_MODEL, len(prompt))
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            if text:
                return text.strip()
            return None
        except Exception as exc:
            log.warning("Groq call failed (%s), trying OpenAI fallback", exc)
            last_fallback_used = "openai"

    # ── OpenAI fallback ───────────────────────────────────────────────────
    if openai_key:
        try:
            from openai import OpenAI

            client_o = OpenAI(api_key=openai_key, timeout=60.0)
            messages_o: list[dict] = []
            if system:
                messages_o.append({"role": "system", "content": system})
            messages_o.append({"role": "user", "content": prompt})

            kwargs_o: dict = dict(
                model=_OPENAI_MODEL,
                max_tokens=max_tokens,
                messages=messages_o,
            )
            if expect_json:
                kwargs_o["response_format"] = {"type": "json_object"}

            resp_o = client_o.chat.completions.create(**kwargs_o)
            text_o = resp_o.choices[0].message.content
            if text_o:
                return text_o.strip()
            return None
        except Exception as exc:
            log.warning("OpenAI call failed (%s), trying Anthropic fallback", exc)
            last_fallback_used = "anthropic"

    # ── Anthropic fallback ────────────────────────────────────────────────
    if anthropic_key:
        try:
            import anthropic

            client_a = anthropic.Anthropic(api_key=anthropic_key, timeout=60.0)
            kwargs_a: dict = dict(
                model=_ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs_a["system"] = system

            resp_a = client_a.messages.create(**kwargs_a)
            text_a = resp_a.content[0].text
            if text_a:
                return text_a.strip()
            return None
        except Exception as exc:
            log.warning("Anthropic call failed: %s", exc)

    log.info("No LLM API key available (GROQ_API_KEY / LLM_API_KEY / ANTHROPIC_API_KEY)")
    last_fallback_used = "none_available"
    return None
