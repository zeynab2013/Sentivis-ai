# ADR-006: Dependency Injection Strategy

**Status:** Accepted  
**Date:** 2026-07-30  
**Version:** 1.0

## Decision

Use a **lightweight constructor-injection container** (`app/container.py`) built in `app/bootstrap.py`. No third-party DI framework. Interfaces defined as `typing.Protocol` or ABC in feature `interfaces/` packages. Tests override bindings at container build time.

## Context

Engineering contract requires dependency injection, testability, and no hidden dependencies. Full DI frameworks (dependency-injector, inject) add complexity and learning overhead for a focused desktop app.

## Alternatives

1. **dependency-injector library** — feature-rich, external dep.
2. **Manual wiring in main.py** — scales poorly.
3. **Service locator singleton** — hidden dependencies, untestable.
4. **Spring-style XML/TOML wiring** — over-engineered.

## Advantages

- Zero extra DI dependency.
- Explicit wiring visible in bootstrap.
- Easy test fakes via container override.
- Matches Python 3.11 typing ecosystem.

## Disadvantages

- Manual registration grows with modules (manageable at project scale).
- No auto-wiring or lifecycle scopes beyond explicit design.

## Justification

A typed container with bootstrap registration satisfies SOLID and testability without violating YAGNI. All wiring stays in one composition root per Clean Architecture rules.
