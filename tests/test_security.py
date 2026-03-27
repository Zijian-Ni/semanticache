"""Tests for the security module."""

from __future__ import annotations


import pytest

from semanticache.security import (
    CacheEncryptor,
    RateLimiter,
    hash_cache_key,
    normalize_unicode,
    sanitize_input,
    strip_control_chars,
    validate_prompt_length,
)


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------


class TestStripControlChars:
    def test_removes_null_byte(self) -> None:
        assert strip_control_chars("hello\x00world") == "helloworld"

    def test_preserves_newline(self) -> None:
        assert strip_control_chars("hello\nworld") == "hello\nworld"

    def test_preserves_tab(self) -> None:
        assert strip_control_chars("hello\tworld") == "hello\tworld"

    def test_removes_escape_sequences(self) -> None:
        assert strip_control_chars("abc\x1b[31mred\x1b[0m") == "abc[31mred[0m"

    def test_empty_string(self) -> None:
        assert strip_control_chars("") == ""

    def test_only_control_chars(self) -> None:
        assert strip_control_chars("\x00\x01\x02\x03") == ""


class TestNormalizeUnicode:
    def test_nfc_normalization(self) -> None:
        # e + combining acute = é in NFC
        composed = "\u00e9"
        decomposed = "e\u0301"
        assert normalize_unicode(decomposed, "NFC") == composed

    def test_nfkc_normalization(self) -> None:
        # Full-width A -> A in NFKC
        assert normalize_unicode("\uff21", "NFKC") == "A"


class TestSanitizeInput:
    def test_strips_whitespace(self) -> None:
        assert sanitize_input("  hello  ") == "hello"

    def test_truncates_long_input(self) -> None:
        result = sanitize_input("a" * 50000, max_length=100)
        assert len(result) == 100

    def test_removes_control_chars(self) -> None:
        result = sanitize_input("hello\x00\x01world")
        assert result == "helloworld"

    def test_normalizes_unicode(self) -> None:
        result = sanitize_input("e\u0301")  # e + combining acute
        assert result == "\u00e9"

    def test_zero_max_length_no_truncation(self) -> None:
        text = "a" * 100
        assert sanitize_input(text, max_length=0) == text


class TestValidatePromptLength:
    def test_valid_length(self) -> None:
        validate_prompt_length("short prompt")  # Should not raise

    def test_exceeds_length(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_prompt_length("a" * 100, max_length=50)


# ---------------------------------------------------------------------------
# Cache key hashing
# ---------------------------------------------------------------------------


class TestHashCacheKey:
    def test_returns_hex_string(self) -> None:
        result = hash_cache_key("test prompt")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic(self) -> None:
        a = hash_cache_key("hello", namespace="ns", salt="salt")
        b = hash_cache_key("hello", namespace="ns", salt="salt")
        assert a == b

    def test_different_prompts_different_hashes(self) -> None:
        a = hash_cache_key("prompt A")
        b = hash_cache_key("prompt B")
        assert a != b

    def test_different_namespaces_different_hashes(self) -> None:
        a = hash_cache_key("prompt", namespace="ns1")
        b = hash_cache_key("prompt", namespace="ns2")
        assert a != b

    def test_salt_changes_hash(self) -> None:
        a = hash_cache_key("prompt", salt="salt1")
        b = hash_cache_key("prompt", salt="salt2")
        assert a != b


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


class TestCacheEncryptor:
    def test_generate_key_length(self) -> None:
        key = CacheEncryptor.generate_key()
        assert len(key) == 32

    def test_encrypt_decrypt_roundtrip(self) -> None:
        key = CacheEncryptor.generate_key()
        enc = CacheEncryptor(key)
        plaintext = "Hello, World! This is a secret response."
        ciphertext = enc.encrypt(plaintext)
        decrypted = enc.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_different_ciphertexts_each_call(self) -> None:
        key = CacheEncryptor.generate_key()
        enc = CacheEncryptor(key)
        a = enc.encrypt("same text")
        b = enc.encrypt("same text")
        assert a != b  # Different nonce each time

    def test_wrong_key_fails(self) -> None:
        key1 = CacheEncryptor.generate_key()
        key2 = CacheEncryptor.generate_key()
        enc1 = CacheEncryptor(key1)
        enc2 = CacheEncryptor(key2)
        ciphertext = enc1.encrypt("secret")
        with pytest.raises(ValueError, match="Decryption failed"):
            enc2.decrypt(ciphertext)

    def test_invalid_key_length(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            CacheEncryptor(b"short")

    def test_tampered_data_fails(self) -> None:
        key = CacheEncryptor.generate_key()
        enc = CacheEncryptor(key)
        ciphertext = enc.encrypt("secret")
        tampered = ciphertext[:-1] + bytes([ciphertext[-1] ^ 0xFF])
        with pytest.raises(ValueError, match="Decryption failed"):
            enc.decrypt(tampered)

    def test_too_short_data_fails(self) -> None:
        key = CacheEncryptor.generate_key()
        enc = CacheEncryptor(key)
        with pytest.raises(ValueError, match="too short"):
            enc.decrypt(b"short")

    def test_unicode_roundtrip(self) -> None:
        key = CacheEncryptor.generate_key()
        enc = CacheEncryptor(key)
        text = "日本語テスト 🚀"
        assert enc.decrypt(enc.encrypt(text)) == text


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_within_limit(self) -> None:
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("client1")

    def test_blocks_over_limit(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")

    def test_different_clients_independent(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        assert limiter.is_allowed("client2")  # Different client

    def test_reset_single_client(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        limiter.reset("client1")
        assert limiter.is_allowed("client1")

    def test_reset_all(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")
        limiter.reset()
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client2")

    def test_properties(self) -> None:
        limiter = RateLimiter(max_requests=10, window_seconds=30)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 30.0
