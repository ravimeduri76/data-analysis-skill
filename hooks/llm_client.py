"""Unified LLM client — routes through Groq (primary) or Anthropic (fallback).

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

# Anthropic fallback
_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def call_llm(
    prompt: str,
    max_tokens: int = 800,
    system: Optional[str] = None,
    expect_json: bool = False,
) -> Optional[str]:
    """Send a prompt to the best available LLM and return the text response.

    Resolution order:
      1. GROQ_API_KEY  → OpenAI-compatible endpoint (Llama 3.3 70B)
      2. ANTHROPIC_API_KEY → Anthropic Messages API (Claude Sonnet)
      3. Neither set → return None

    Parameters
    ----------
    prompt : str
        User message content.
    max_tokens : int
        Maximum tokens in the response.
    system : str | None
        Optional system message (used by execution_graph).
    expect_json : bool
        If True, requests structured JSON output (Groq ``response_format``).

    Returns
    -------
    str | None
        The model's text reply, or None on failure / no key available.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    # ── Try Groq first ────────────────────────────────────────────────────
    if groq_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=groq_key, base_url=_GROQ_BASE_URL)
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

            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            if text:
                return text.strip()
            return None
        except Exception as exc:
            log.warning("Groq call failed (%s), trying Anthropic fallback", exc)

    # ── Anthropic fallback ────────────────────────────────────────────────
    if anthropic_key:
        try:
            import anthropic

            client_a = anthropic.Anthropic(api_key=anthropic_key)
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

    log.info("No LLM API key available (GROQ_API_KEY / ANTHROPIC_API_KEY)")
    return None
