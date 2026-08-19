"""Utilities for generating HMAC signatures for HTTP and WebSocket requests.

This module mirrors the JS sample logic:
- Signature: HMAC-SHA256 over the stringToSign, URL-safe base64 without padding
- HTTP stringToSign: method + path + sortedQueryParams + timestamp + nonce
- WS stringToSign:   GET    + path + sortedQueryParams + timestamp + nonce

Sorted query params are joined as "key=value" concatenations with no separators,
keys sorted lexicographically. Parameters used for signing exclude any keys
starting with "x-" (for WebSocket flow), matching the sample.
"""

from __future__ import annotations

import base64
import hmac
import hashlib
import time
import uuid
from typing import Dict, Mapping, Optional
from urllib.parse import urlsplit, parse_qsl


def _urlsafe_b64_nopad(data: bytes) -> str:
    """Return URL-safe base64 string without padding (=)."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def generate_signature(message: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature and return URL-safe base64 (no padding)."""
    mac = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
    return _urlsafe_b64_nopad(mac.digest())


def _sorted_query_concat(params: Optional[Mapping[str, str]]) -> str:
    """Concatenate sorted query params as key=value with no separators.

    Example: {b:2, a:1} -> "a=1b=2"
    """
    if not params:
        return ""
    return "".join(f"{key}={params[key]}" for key in sorted(params.keys()))


def generate_signed_headers(method: str, url: str, params: Optional[Mapping[str, str]], secret_key: str) -> Dict[str, str]:
    """Create X-* signature headers for HTTP requests.

    stringToSign = method + path + sortedQueryParams + timestamp + nonce
    """
    parsed = urlsplit(url)
    path = parsed.path or "/"
    timestamp = str(int(time.time()))
    nonce = str(uuid.uuid4())

    query_concat = _sorted_query_concat(params)
    string_to_sign = f"{method.upper()}{path}{query_concat}{timestamp}{nonce}"

    signature = generate_signature(string_to_sign, secret_key)
    # Use lowercase header names to match reference implementation
    return {
        "x-timestamp": timestamp,
        "x-nonce": nonce,
        "x-signature": signature,
    }


def generate_signed_websocket_params(url: str, secret_key: str) -> Dict[str, str]:
    """Create x-* URL params for a WebSocket GET request based on the given URL.

    - Exclude any params that start with 'x-' from the string to sign.
    - Use the path component of the URL.
    """
    parsed = urlsplit(url)
    path = parsed.path or "/"

    # Extract query params and omit any x-* keys for signing
    params_for_signing: Dict[str, str] = {}
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if not key.lower().startswith("x-"):
            params_for_signing[key] = value

    timestamp = str(int(time.time()))
    nonce = str(uuid.uuid4())

    query_concat = _sorted_query_concat(params_for_signing)
    string_to_sign = f"GET{path}{query_concat}{timestamp}{nonce}"

    signature = generate_signature(string_to_sign, secret_key)
    return {
        "x-timestamp": timestamp,
        "x-nonce": nonce,
        "x-signature": signature,
    }


def get_secret_key() -> Optional[str]:
    """Return the signing secret from environment if available.

    Checked env vars (in order): NINJA_API_SECRET_KEY, API_SECRET_KEY
    """
    #return os.environ.get("NINJA_API_SECRET_KEY") or os.environ.get("API_SECRET_KEY")
    return '4f0465bfea7761a510dda451ff86a935bf0c8ed6fb37f80441509c64328788c8'