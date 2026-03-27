# Contributing to SemantiCache

Thank you for your interest in contributing to SemantiCache! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/semanticache.git
   cd semanticache
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

- Run `ruff check .` before committing — zero lint errors allowed
- Run `ruff format .` to auto-format code
- Type hints on all function signatures
- Docstrings on all public methods (Google style)
- Line length: 100 characters

### Testing

- Write tests for all new features
- Maintain >80% test coverage
- Run the test suite:
  ```bash
  pytest
  pytest --cov=semanticache  # with coverage
  ```

### Commit Messages

Follow conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `test:` — test additions/changes
- `refactor:` — code refactoring
- `chore:` — maintenance

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Update CHANGELOG.md
4. Submit a PR against the `main` branch
5. Wait for review

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] `ruff check .` passes with zero errors
- [ ] `ruff format .` applied
- [ ] CHANGELOG.md updated

## Contributor License Agreement (CLA)

By contributing to SemantiCache, you agree that your contributions will be licensed under the same license as the project (PolyForm Shield License 1.0.0).

You represent that:
- You have the right to submit the contribution
- Your contribution is your original work
- You grant the project maintainers the right to use your contribution

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- For security vulnerabilities, see [SECURITY.md](SECURITY.md)
- Include reproduction steps for bug reports
- Search existing issues before creating a new one

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

Open a GitHub Discussion or reach out to the maintainers.
