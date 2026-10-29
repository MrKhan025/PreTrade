"""
Configuration and API key resolution for PreTrade.

Resolves the Finnhub API key (and optional supplemental provider keys) across:
1. Explicit runtime/sidebar override
2. Streamlit secrets (`.streamlit/secrets.toml`)
3. Environment variables (`FINNHUB_API_KEY`)
4. Project `.env` file
5. Local JSON config (`config.local.json`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
LOCAL_JSON_FILE = PROJECT_ROOT / "config.local.json"


def _parse_dotenv(dotenv_path: Path = ENV_FILE) -> dict[str, str]:
    """Parse key=value pairs from a local .env file without external dependencies."""
    values: dict[str, str] = {}
    if not dotenv_path.is_file():
        return values
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
                val = val[1:-1]
            if key:
                values[key] = val
    except OSError:
        pass
    return values


def _read_local_json(json_path: Path = LOCAL_JSON_FILE) -> dict[str, Any]:
    """Read key-value configuration from config.local.json if present."""
    if not json_path.is_file():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_streamlit_secret(key_name: str = "FINNHUB_API_KEY", section: str = "finnhub") -> str:
    """Attempt to read a secret from Streamlit's st.secrets without failing outside Streamlit."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return ""
        if key_name in secrets and secrets[key_name]:
            return str(secrets[key_name]).strip()
        if section in secrets and isinstance(secrets[section], dict):
            nested = secrets[section].get("api_key") or secrets[section].get(key_name)
            if nested:
                return str(nested).strip()
    except Exception:
        return ""
    return ""


def resolve_api_key(
    override: str | None = None,
    env_var: str = "FINNHUB_API_KEY",
    secret_section: str = "finnhub",
    dotenv_path: Path = ENV_FILE,
    json_path: Path = LOCAL_JSON_FILE,
) -> tuple[str, str]:
    """Return `(resolved_key, source_label)` following the priority chain."""
    if override and str(override).strip():
        return str(override).strip(), "sidebar"

    st_secret = _read_streamlit_secret(env_var, secret_section)
    if st_secret:
        return st_secret, "streamlit_secrets"

    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val, "env"

    dotenv_vals = _parse_dotenv(dotenv_path)
    if dotenv_vals.get(env_var, "").strip():
        return dotenv_vals[env_var].strip(), ".env"

    json_vals = _read_local_json(json_path)
    raw_json = json_vals.get(env_var) or (
        json_vals.get(secret_section, {}).get("api_key")
        if isinstance(json_vals.get(secret_section), dict)
        else ""
    )
    if raw_json and str(raw_json).strip():
        return str(raw_json).strip(), "config.local.json"

    return "", "none"


def get_finnhub_api_key(override: str | None = None) -> str:
    """Return the resolved Finnhub API key string (or empty string if unconfigured)."""
    key, _ = resolve_api_key(override=override, env_var="FINNHUB_API_KEY", secret_section="finnhub")
    return key


def finnhub_key_source(override: str | None = None) -> str:
    """Return the human-readable source name where the Finnhub API key was found."""
    _, source = resolve_api_key(override=override, env_var="FINNHUB_API_KEY", secret_section="finnhub")
    return source


def mask_key(key: str | None) -> str:
    """Mask an API key for safe UI display."""
    if not key or not str(key).strip():
        return "Not configured"
    clean = str(key).strip()
    if len(clean) <= 6:
        return "***" + clean[-2:]
    return f"{clean[:3]}…{clean[-3:]}"
