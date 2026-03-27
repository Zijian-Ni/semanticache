# Security Model

## Overview

SemantiCache includes a security layer (`semanticache.security`) that provides defense-in-depth for cached LLM responses.

## Input Sanitization

All user prompts should be sanitized before caching to prevent cache poisoning and injection attacks.

```python
from semanticache.security import sanitize_input

clean = sanitize_input(user_prompt, max_length=32000)
```

**Steps performed:**
1. Unicode normalization (NFC by default)
2. Control character stripping (preserves `\n`, `\r`, `\t`)
3. Whitespace trimming
4. Length truncation

## Cache Key Hashing

Cache keys are SHA-256 hashed to prevent information leakage and ensure fixed-length keys:

```python
from semanticache.security import hash_cache_key

key = hash_cache_key(prompt, namespace="prod", salt="my-secret-salt")
```

**Properties:**
- Deterministic: same input always produces the same hash
- Collision-resistant: SHA-256 provides 128-bit collision resistance
- Opaque: original prompt cannot be recovered from the hash

## Encryption at Rest

Cached responses can be encrypted with AES-256-GCM:

```python
from semanticache.security import CacheEncryptor

key = CacheEncryptor.generate_key()  # Store this securely!
encryptor = CacheEncryptor(key)

# Encrypt before storing
ciphertext = encryptor.encrypt("LLM response text")

# Decrypt when retrieving
plaintext = encryptor.decrypt(ciphertext)
```

**Properties:**
- AES-256-GCM: authenticated encryption with associated data
- Unique nonce per encryption: prevents ciphertext analysis
- Tamper detection: GCM tag verifies data integrity
- Requires `pip install semanticache[security]` (cryptography package)

## Rate Limiting

The built-in rate limiter protects dashboard API endpoints:

```python
from semanticache.security import RateLimiter

limiter = RateLimiter(max_requests=60, window_seconds=60)

if limiter.is_allowed(client_ip):
    # Process request
else:
    # Return 429 Too Many Requests
```

## Prompt Length Validation

Prevent abuse by limiting prompt length:

```python
from semanticache.security import validate_prompt_length

validate_prompt_length(prompt, max_length=32000)  # Raises ValueError if too long
```

## Threat Model

| Threat | Mitigation |
|---|---|
| Cache poisoning via control chars | Input sanitization strips control characters |
| Unicode homograph attacks | NFC normalization |
| Cache key enumeration | SHA-256 hashing makes keys opaque |
| Data exposure at rest | Optional AES-256-GCM encryption |
| API abuse / DoS | Rate limiting on dashboard endpoints |
| Prompt injection via length | Configurable max prompt length |

## Recommendations

1. Always sanitize user input before caching
2. Use encryption for sensitive cached responses
3. Enable rate limiting on any public-facing endpoints
4. Set appropriate max prompt lengths for your use case
5. Rotate encryption keys periodically
6. Store encryption keys in a secrets manager, not in code
