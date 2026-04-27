"""Unified LLM client — routes through Groq → OpenAI → Anthropic.

All hooks should call ``call_llm()`` instead of directly importing anthropic/openai.
Includes token-cost tracking (OB1 pattern: budget-aware LLM calls).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-cost tracking (WI-0A)
# ---------------------------------------------------------------------------
# Cost per 1K tokens (input, output) in USD — approximate as of 2026-04
_COST_PER_1K: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.00059, 0.00079),   # Groq
    "gpt-4o":                  (0.0025, 0.01),        # OpenAI
    "claude-sonnet-4-6":       (0.003, 0.015),        # Anthropic
}

_token_ledger: dict = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd": 0.0,
    "calls": 0,
    "by_provider": {},  # provider -> {input_tokens, output_tokens, cost_usd, calls}
}


def _record_usage(provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Record token usage from a successful LLM call."""
    cost_in, cost_out = _COST_PER_1K.get(model, (0.003, 0.015))
    cost = (input_tokens * cost_in + output_tokens * cost_out) / 1000.0

    _token_ledger["total_input_tokens"] += input_tokens
    _token_ledger["total_output_tokens"] += output_tokens
    _token_ledger["total_cost_usd"] += cost
    _token_ledger["calls"] += 1

    if provider not in _token_ledger["by_provider"]:
        _token_ledger["by_provider"][provider] = {
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0,
        }
    bucket = _token_ledger["by_provider"][provider]
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["cost_usd"] += cost
    bucket["calls"] += 1


def get_token_usage() -> dict:
    """Return a copy of the current token usage ledger."""
    return {k: (v.copy() if isinstance(v, dict) else v) for k, v in _token_ledger.items()}


def check_budget(hard_cap_usd: float) -> bool:
    """Return True if usage is still under the budget cap, False if exceeded."""
    return _token_ledger["total_cost_usd"] < hard_cap_usd


def reset_token_usage() -> None:
    """Reset the token ledger (useful for test isolation or per-run tracking)."""
    _token_ledger["total_input_tokens"] = 0
    _token_ledger["total_output_tokens"] = 0
    _token_ledger["total_cost_usd"] = 0.0
    _token_ledger["calls"] = 0
    _token_ledger["by_provider"] = {}

# Retry config — rides through transient DNS / network blips
_MAX_RETRIES = 4          # total attempts per provider
_INITIAL_BACKOFF_S = 2.0  # first wait (doubles each retry: 2, 4, 8, 16)
_BACKOFF_FACTOR = 2.0
_RETRIABLE_ERRORS = ("ConnectionError", "ConnectError", "getaddrinfo", "Connection error",
                      "RemoteDisconnected", "ConnectionResetError", "TimeoutError")


def _is_retriable(exc: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying."""
    exc_str = str(exc)
    exc_type = type(exc).__name__
    return any(tag in exc_str or tag in exc_type for tag in _RETRIABLE_ERRORS)


def _retry_call(func, provider_name: str):
    """Call func() with exponential backoff on transient network errors."""
    backoff = _INITIAL_BACKOFF_S
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return func()
        except Exception as exc:
            if attempt < _MAX_RETRIES and _is_retriable(exc):
                log.warning(
                    "%s attempt %d/%d failed (%s), retrying in %.1fs",
                    provider_name, attempt, _MAX_RETRIES, exc, backoff,
                )
                time.sleep(backoff)
                backoff *= _BACKOFF_FACTOR
            else:
                raise

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

            def _groq_call():
                resp = client.chat.completions.create(**kwargs)
                if hasattr(resp, "usage") and resp.usage:
                    _record_usage("groq", _GROQ_MODEL,
                                  resp.usage.prompt_tokens or 0,
                                  resp.usage.completion_tokens or 0)
                return resp.choices[0].message.content

            text = _retry_call(_groq_call, "Groq")
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

            def _openai_call():
                resp_o = client_o.chat.completions.create(**kwargs_o)
                if hasattr(resp_o, "usage") and resp_o.usage:
                    _record_usage("openai", _OPENAI_MODEL,
                                  resp_o.usage.prompt_tokens or 0,
                                  resp_o.usage.completion_tokens or 0)
                return resp_o.choices[0].message.content

            text_o = _retry_call(_openai_call, "OpenAI")
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

            def _anthropic_call():
                resp_a = client_a.messages.create(**kwargs_a)
                if hasattr(resp_a, "usage") and resp_a.usage:
                    _record_usage("anthropic", _ANTHROPIC_MODEL,
                                  getattr(resp_a.usage, "input_tokens", 0),
                                  getattr(resp_a.usage, "output_tokens", 0))
                return resp_a.content[0].text

            text_a = _retry_call(_anthropic_call, "Anthropic")
            if text_a:
                return text_a.strip()
            return None
        except Exception as exc:
            log.warning("Anthropic call failed: %s", exc)

    log.info("No LLM API key available (GROQ_API_KEY / LLM_API_KEY / ANTHROPIC_API_KEY)")
    last_fallback_used = "none_available"
    return None
