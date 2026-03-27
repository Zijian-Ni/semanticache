"""Security layer for SemantiCache — input sanitization, hashing, encryption, rate limiting."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
)

DEFAULT_MAX_PROMPT_LENGTH = 32_000
"""Default maximum prompt length in characters."""


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------


def strip_control_chars(text: str) -> str:
    """Remove ASCII/Latin-1 control characters from *text*.

    Preserves newlines (``\\n``), carriage returns (``\\r``), and tabs (``\\t``).
    """
    return _CONTROL_CHAR_RE.sub("", text)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize *text* to the given Unicode normal form.

    Args:
        text: Input string.
        form: Unicode normal form (``NFC``, ``NFD``, ``NFKC``, ``NFKD``).

    Returns:
        Normalized string.
    """
    return unicodedata.normalize(form, text)


def sanitize_input(
    text: str,
    *,
    max_length: int = DEFAULT_MAX_PROMPT_LENGTH,
    unicode_form: str = "NFC",
) -> str:
    """Sanitize user input for safe cache operations.

    Steps:
        1. Normalize Unicode to *unicode_form*.
        2. Strip control characters.
        3. Strip leading/trailing whitespace.
        4. Truncate to *max_length*.

    Args:
        text: Raw user input.
        max_length: Maximum allowed length.  Use ``0`` or negative to skip truncation.
        unicode_form: Unicode normalization form.

    Returns:
        Sanitized string.

    Raises:
        ValueError: If *text* exceeds *max_length* after sanitization and
            truncation is disabled (``max_length <= 0`` is no-op; the error is
            never raised by this helper — callers can validate separately).
    """
    text = normalize_unicode(text, form=unicode_form)
    text = strip_control_chars(text)
    text = text.strip()
    if max_length > 0 and len(text) > max_length:
        logger.warning(
            "Prompt truncated from %d to %d characters",
            len(text),
            max_length,
        )
        text = text[:max_length]
    return text


def validate_prompt_length(text: str, max_length: int = DEFAULT_MAX_PROMPT_LENGTH) -> None:
    """Raise ``ValueError`` if *text* exceeds *max_length*.

    Args:
        text: The prompt text to validate.
        max_length: Maximum allowed length.

    Raises:
        ValueError: When the prompt is too long.
    """
    if len(text) > max_length:
        raise ValueError(f"Prompt length {len(text)} exceeds maximum allowed length {max_length}")


# ---------------------------------------------------------------------------
# Cache key hashing
# ---------------------------------------------------------------------------


def hash_cache_key(prompt: str, namespace: str = "default", salt: str = "") -> str:
    """Produce a SHA-256 hex digest for a cache key.

    Hashing prevents cache poisoning by ensuring keys are fixed-length and
    opaque.

    Args:
        prompt: The prompt text.
        namespace: Namespace partition.
        salt: Optional salt for additional entropy.

    Returns:
        64-character hex string.
    """
    payload = f"{salt}:{namespace}:{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# AES-256 encryption (optional — requires ``cryptography``)
# ---------------------------------------------------------------------------


class CacheEncryptor:
    """AES-256-GCM encryption for cached responses at rest.

    Requires the ``cryptography`` package (install via ``pip install semanticache[security]``).

    Args:
        key: A 32-byte encryption key.  Use :func:`generate_key` to create one.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("Encryption key must be exactly 32 bytes (AES-256)")
        self._key = key
        self._backend: Any = None
        self._aesgcm: Any = None
        self._init_crypto()

    def _init_crypto(self) -> None:
        """Lazily import and initialise the cryptography backend."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise ImportError(
                "The 'cryptography' package is required for encryption. "
                "Install it with: pip install semanticache[security]"
            ) from exc
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt *plaintext* and return ``nonce + ciphertext``.

        Args:
            plaintext: The string to encrypt.

        Returns:
            Bytes containing 12-byte nonce followed by ciphertext + tag.
        """
        import os

        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return nonce + ciphertext

    def decrypt(self, data: bytes) -> str:
        """Decrypt data produced by :meth:`encrypt`.

        Args:
            data: The nonce + ciphertext bytes.

        Returns:
            Decrypted plaintext string.

        Raises:
            ValueError: If decryption fails (e.g. wrong key or tampered data).
        """
        if len(data) < 12:
            raise ValueError("Encrypted data too short — expected at least 12-byte nonce")
        nonce = data[:12]
        ciphertext = data[12:]
        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise ValueError("Decryption failed — wrong key or corrupted data") from exc
        return plaintext.decode("utf-8")

    @staticmethod
    def generate_key() -> bytes:
        """Generate a random 32-byte encryption key.

        Returns:
            32 random bytes suitable for AES-256.
        """
        import os

        return os.urandom(32)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple in-memory sliding-window rate limiter.

    Tracks request timestamps per client key and rejects requests that exceed
    the configured rate.

    Args:
        max_requests: Maximum requests allowed in *window_seconds*.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, client_key: str) -> bool:
        """Check whether *client_key* may make another request.

        Args:
            client_key: Identifier for the client (e.g. IP address).

        Returns:
            ``True`` if the request is allowed, ``False`` if rate-limited.
        """
        now = time.monotonic()
        timestamps = self._requests.setdefault(client_key, [])

        # Prune old timestamps
        cutoff = now - self._window
        self._requests[client_key] = timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= self._max_requests:
            return False

        timestamps.append(now)
        return True

    def reset(self, client_key: str | None = None) -> None:
        """Reset rate limit state.

        Args:
            client_key: Reset only this key. If ``None``, reset all.
        """
        if client_key is not None:
            self._requests.pop(client_key, None)
        else:
            self._requests.clear()

    @property
    def max_requests(self) -> int:
        """Maximum requests per window."""
        return self._max_requests

    @property
    def window_seconds(self) -> float:
        """Window length in seconds."""
        return self._window
