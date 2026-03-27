# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

We take the security of SemantiCache seriously. If you discover a security vulnerability, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. Email: **security@semanticache.dev**
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgement**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix release**: Within 30 days for critical issues

### What to Expect

- We will acknowledge receipt of your report
- We will investigate and validate the issue
- We will work on a fix and coordinate disclosure
- We will credit you in the release notes (unless you prefer anonymity)

### Scope

The following are in scope:
- Cache poisoning attacks
- Encryption/decryption vulnerabilities
- Authentication/authorization bypasses
- Injection attacks via cached data
- Information leakage through cache keys

### Out of Scope

- Denial of service via resource exhaustion (use rate limiting)
- Issues in third-party dependencies (report to those projects directly)
- Social engineering

## Security Features

SemantiCache includes built-in security features:

- **Input sanitization**: Unicode normalization, control character stripping
- **Cache key hashing**: SHA-256 prevents cache enumeration
- **Encryption at rest**: Optional AES-256-GCM for cached responses
- **Rate limiting**: Configurable sliding-window rate limiter
- **Prompt length limits**: Prevent abuse via oversized inputs

See [docs/security.md](docs/security.md) for details.
