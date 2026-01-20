---
layout: default
title: Examples
nav_order: 9
parent: null
---

# Examples
{: .no_toc }

Real-world examples and sample outputs from YOLO Developer.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Example 1: REST API Development

### Input Requirements

**requirements.md:**
```markdown
# User Management API

## Overview
Build a REST API for managing users with authentication.

## Functional Requirements
- FR1: Users can register with email and password
- FR2: Users can login and receive JWT token
- FR3: Users can view their profile
- FR4: Users can update their profile
- FR5: Admins can list all users
- FR6: Admins can deactivate user accounts

## Non-Functional Requirements
- NFR1: API response time < 200ms
- NFR2: Support 1000 concurrent users
- NFR3: 99.9% uptime SLA

## Constraints
- Must use PostgreSQL database
- Must use Python with FastAPI
- Must include OpenAPI documentation
```

### Seed Output

```
$ yolo seed requirements.md

Parsing requirements document...
  ✓ Parsed 6 functional requirements
  ✓ Parsed 3 non-functional requirements
  ✓ Parsed 3 constraints

Running validation...
  ✓ No ambiguities detected
  ✓ No contradictions found
  ✓ SOP constraints validated

Seed Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Requirements: 12 total
  Categories:
    - Functional: 6
    - Non-Functional: 3
    - Constraints: 3
  Quality Score: 0.95

Seed ID: seed_7f3a8b2c-1d4e-5f6a-8b9c-0d1e2f3a4b5c

Ready to run: yolo run
```

### Sprint Output

```
$ yolo run

Starting autonomous development sprint...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ANALYST] Analyzing requirements...
  → Crystallizing FR1: "Users can register with email and password"
    ✓ Crystallized to 3 specifications:
      - REQ-001: Email format validation (RFC 5322)
      - REQ-002: Password minimum 8 chars, 1 upper, 1 number
      - REQ-003: Unique email constraint
  → Crystallizing FR2: "Users can login and receive JWT token"
    ✓ Crystallized to 2 specifications:
      - REQ-004: Credential verification
      - REQ-005: JWT generation with 1h expiry
  ... (remaining requirements)
  ✓ Analysis complete: 12 requirements → 18 specifications (2m 34s)

[PM] Generating user stories...
  → Creating story US-001: "User Registration"
    Priority: Must-Have
    Story Points: 5
    Acceptance Criteria: 4
  → Creating story US-002: "User Authentication"
    Priority: Must-Have
    Story Points: 3
    Acceptance Criteria: 3
  → Creating story US-003: "Profile View"
    Priority: Must-Have
    Story Points: 2
    Acceptance Criteria: 2
  → Creating story US-004: "Profile Update"
    Priority: Should-Have
    Story Points: 3
    Acceptance Criteria: 3
  → Creating story US-005: "Admin User List"
    Priority: Should-Have
    Story Points: 3
    Acceptance Criteria: 3
  → Creating story US-006: "Admin User Deactivation"
    Priority: Could-Have
    Story Points: 2
    Acceptance Criteria: 2
  ✓ Generated 6 user stories (1m 12s)

[ARCHITECT] Designing system architecture...
  → Validating 12-Factor App compliance
    ✓ I. Codebase: Single repo
    ✓ II. Dependencies: requirements.txt
    ✓ III. Config: Environment variables
    ✓ IV. Backing services: PostgreSQL as attached resource
    ✓ V. Build, release, run: Separate stages
    ... (remaining factors)
    Result: 12/12 factors compliant

  → Generating ADR-001: "Authentication Strategy"
    Decision: JWT with refresh tokens
    Rationale: Stateless, scalable, standard
    Trade-offs: Cannot revoke before expiry

  → Generating ADR-002: "Database Schema Design"
    Decision: Single users table with soft deletes
    Rationale: Simple for current requirements
    Trade-offs: May need migration for roles

  → Generating ADR-003: "API Versioning Strategy"
    Decision: URL path versioning (/api/v1/)
    Rationale: Clear, explicit, easy to maintain
    Trade-offs: URL changes between versions

  → Evaluating quality attributes
    Performance: 0.88 (Async handlers, connection pooling)
    Security: 0.92 (JWT, bcrypt, HTTPS, rate limiting)
    Scalability: 0.85 (Stateless, horizontal scaling ready)
    Maintainability: 0.90 (Clean architecture, typed)
    Reliability: 0.87 (Health checks, graceful shutdown)

  → Identifying technical risks
    RISK-001: Token revocation (Medium probability, Low impact)
      Mitigation: Implement token blacklist for critical ops

  ✓ Architecture complete: 3 ADRs, 1 risk (3m 45s)

[DEV] Implementing stories...
  → US-001: User Registration
    Creating: src/app/models/user.py
    Creating: src/app/schemas/user.py
    Creating: src/app/api/v1/endpoints/auth.py
    Creating: src/app/core/security.py
    Creating: tests/api/test_auth.py
    ✓ Completed (4m 22s, 89% coverage)

  → US-002: User Authentication
    Updating: src/app/api/v1/endpoints/auth.py
    Creating: src/app/core/jwt.py
    Creating: tests/api/test_login.py
    ✓ Completed (3m 15s, 91% coverage)

  → US-003: Profile View
    Creating: src/app/api/v1/endpoints/users.py
    Creating: tests/api/test_profile.py
    ✓ Completed (2m 08s, 88% coverage)

  → US-004: Profile Update
    Updating: src/app/api/v1/endpoints/users.py
    Updating: tests/api/test_profile.py
    ✓ Completed (2m 45s, 90% coverage)

  → US-005: Admin User List
    Creating: src/app/api/v1/endpoints/admin.py
    Creating: src/app/core/permissions.py
    Creating: tests/api/test_admin.py
    ✓ Completed (3m 30s, 87% coverage)

  → US-006: Admin User Deactivation
    Updating: src/app/api/v1/endpoints/admin.py
    Updating: tests/api/test_admin.py
    ✓ Completed (2m 18s, 89% coverage)

  ✓ Implementation complete: 6 stories (18m 18s)

[TEA] Validating test coverage...
  → Running test suite
    tests/api/test_auth.py ............ [12 passed]
    tests/api/test_login.py .......... [10 passed]
    tests/api/test_profile.py ........ [8 passed]
    tests/api/test_admin.py .......... [9 passed]
    tests/unit/test_security.py ...... [6 passed]
    tests/unit/test_jwt.py ........... [5 passed]

  → Coverage analysis
    Overall: 89%
    By module:
      app/models: 95%
      app/schemas: 92%
      app/api: 88%
      app/core: 85%

  → Risk assessment
    High Risk: None
    Medium Risk: app/core/security.py (edge cases)
    Low Risk: None

  → Testability audit
    Score: 0.91
    All code is testable

  → Deployment decision
    ✓ APPROVED (confidence: 0.93)

  ✓ Validation complete (2m 10s)

[SM] Sprint complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sprint Summary:
  Duration: 27m 59s
  Stories: 6/6 completed
  Test Coverage: 89%
  Quality Gates: 4/4 passing

Artifacts Generated:
  Architecture:
    - 3 Architecture Decision Records
    - 12-Factor compliance report
    - Quality attribute evaluation
    - Technical risk assessment

  Code:
    - 12 source files
    - 50 test cases
    - OpenAPI specification
    - Database migrations

  Documentation:
    - API documentation
    - README updates
    - CHANGELOG entry

Token Usage: 145,230 tokens ($1.42)

Files created/modified:
  src/
  ├── app/
  │   ├── __init__.py
  │   ├── main.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   └── user.py
  │   ├── schemas/
  │   │   ├── __init__.py
  │   │   └── user.py
  │   ├── api/
  │   │   └── v1/
  │   │       ├── __init__.py
  │   │       ├── endpoints/
  │   │       │   ├── auth.py
  │   │       │   ├── users.py
  │   │       │   └── admin.py
  │   │       └── router.py
  │   └── core/
  │       ├── __init__.py
  │       ├── config.py
  │       ├── security.py
  │       ├── jwt.py
  │       └── permissions.py
  tests/
  ├── __init__.py
  ├── conftest.py
  ├── api/
  │   ├── test_auth.py
  │   ├── test_login.py
  │   ├── test_profile.py
  │   └── test_admin.py
  └── unit/
      ├── test_security.py
      └── test_jwt.py
```

### Generated Code Samples

**src/app/models/user.py:**
```python
"""User model for database operations."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from app.core.database import Base


class User(Base):
    """User model representing registered users."""

    __tablename__ = "users"

    id: UUID = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email: str = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password: str = Column(String(255), nullable=False)
    full_name: str = Column(String(255), nullable=True)
    is_active: bool = Column(Boolean, default=True)
    is_admin: bool = Column(Boolean, default=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<User {self.email}>"
```

**src/app/api/v1/endpoints/auth.py:**
```python
"""Authentication endpoints for user registration and login."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_password_hash, verify_password
from app.core.jwt import create_access_token, create_refresh_token
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, Token, LoginRequest

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Register a new user.

    - **email**: Valid email address (must be unique)
    - **password**: Minimum 8 characters, 1 uppercase, 1 number
    - **full_name**: Optional display name
    """
    # Check if user exists
    existing = await db.execute(
        select(User).where(User.email == user_in.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user
    user = User(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.post("/login", response_model=Token)
async def login(
    credentials: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Authenticate user and return JWT tokens.

    - **email**: Registered email address
    - **password**: Account password
    """
    # Find user
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return {
        "access_token": create_access_token(user.id),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
    }
```

**tests/api/test_auth.py:**
```python
"""Tests for authentication endpoints."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models.user import User


@pytest.fixture
def user_data() -> dict:
    """Sample user registration data."""
    return {
        "email": "test@example.com",
        "password": "SecurePass123",
        "full_name": "Test User",
    }


class TestUserRegistration:
    """Tests for POST /api/v1/auth/register."""

    @pytest.mark.asyncio
    async def test_register_success(
        self,
        client: AsyncClient,
        user_data: dict,
    ) -> None:
        """Test successful user registration."""
        response = await client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["full_name"] == user_data["full_name"]
        assert "id" in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(
        self,
        client: AsyncClient,
        user_data: dict,
        db: AsyncSession,
    ) -> None:
        """Test registration fails with duplicate email."""
        # Create existing user
        user = User(
            email=user_data["email"],
            hashed_password="existing_hash",
        )
        db.add(user)
        await db.commit()

        response = await client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_invalid_email(
        self,
        client: AsyncClient,
    ) -> None:
        """Test registration fails with invalid email format."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "invalid-email", "password": "SecurePass123"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_weak_password(
        self,
        client: AsyncClient,
    ) -> None:
        """Test registration fails with weak password."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "test@example.com", "password": "weak"},
        )

        assert response.status_code == 422


class TestUserLogin:
    """Tests for POST /api/v1/auth/login."""

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        client: AsyncClient,
        registered_user: User,
    ) -> None:
        """Test successful login returns tokens."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": registered_user.email, "password": "SecurePass123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(
        self,
        client: AsyncClient,
        registered_user: User,
    ) -> None:
        """Test login fails with wrong password."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": registered_user.email, "password": "WrongPassword"},
        )

        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_deactivated_user(
        self,
        client: AsyncClient,
        deactivated_user: User,
    ) -> None:
        """Test login fails for deactivated user."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": deactivated_user.email, "password": "SecurePass123"},
        )

        assert response.status_code == 403
        assert "deactivated" in response.json()["detail"]
```

---

## Example 2: Ambiguity Detection

### Input with Ambiguities

**vague-requirements.md:**
```markdown
# E-commerce Platform

## Requirements
- The system should be fast
- Users should be able to checkout easily
- Support many products
- Handle high traffic
- Secure payment processing
```

### Seed Output with Clarifications

```
$ yolo seed vague-requirements.md

Parsing requirements document...
  ✓ Parsed 5 requirements

Running validation...
  ⚠ 4 ambiguities detected:

  1. "The system should be fast"
     Question: What response time threshold defines 'fast'?
     Suggestions:
       a) < 100ms (real-time applications)
       b) < 200ms (standard web applications)
       c) < 500ms (acceptable for complex operations)

  2. "Users should be able to checkout easily"
     Question: What defines an 'easy' checkout?
     Suggestions:
       a) Single-page checkout (all fields on one page)
       b) 3-step checkout (cart → shipping → payment)
       c) One-click checkout (saved payment methods)

  3. "Support many products"
     Question: How many products should the system support?
     Suggestions:
       a) 1,000 products (small catalog)
       b) 100,000 products (medium catalog)
       c) 1,000,000+ products (large marketplace)

  4. "Handle high traffic"
     Question: What traffic volume constitutes 'high'?
     Suggestions:
       a) 1,000 concurrent users
       b) 10,000 concurrent users
       c) 100,000+ concurrent users

  ✓ No contradictions found

Clarification Required:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please address the ambiguities above by:
  1. Updating requirements.md with specific values
  2. Or use --skip-validation to proceed anyway (not recommended)

Quality Score: 0.45 (below threshold 0.70)
```

---

## Example 3: Status Monitoring

### Status During Sprint

```
$ yolo status --agents --gates

Sprint Status: IN_PROGRESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Progress: ████████████████░░░░ 75%

Stories:
  ✓ US-001: User Registration        [DONE]
  ✓ US-002: User Authentication      [DONE]
  ✓ US-003: Profile View             [DONE]
  ✓ US-004: Profile Update           [DONE]
  → US-005: Admin User List          [IN PROGRESS]
  ○ US-006: Admin User Deactivation  [PENDING]

Agent Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────┬───────────┬──────────┬────────────────┐
│ Agent   │ Status    │ Duration │ Tokens         │
├─────────┼───────────┼──────────┼────────────────┤
│ ANALYST │ COMPLETED │ 2m 34s   │ 12,340         │
│ PM      │ COMPLETED │ 1m 12s   │ 8,920          │
│ ARCH    │ COMPLETED │ 3m 45s   │ 15,670         │
│ DEV     │ ACTIVE    │ 12m 18s  │ 42,890         │
│ TEA     │ PENDING   │ -        │ -              │
│ SM      │ MONITOR   │ 19m 49s  │ 2,340          │
└─────────┴───────────┴──────────┴────────────────┘

Quality Gates:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────────┬───────┬───────────┬────────┐
│ Gate                   │ Score │ Threshold │ Status │
├────────────────────────┼───────┼───────────┼────────┤
│ Testability            │ 0.89  │ 0.70      │ ✓ PASS │
│ AC Measurability       │ 0.92  │ 0.70      │ ✓ PASS │
│ Architecture Valid.    │ 0.85  │ 0.70      │ ✓ PASS │
│ Definition of Done     │ 0.88  │ 0.70      │ ✓ PASS │
└────────────────────────┴───────┴───────────┴────────┘

Token Usage: 82,160 tokens ($0.81)
Elapsed: 19m 49s
```

---

## Example 4: Audit Trail Export

### JSON Export

```
$ yolo logs --export audit.json --format json
```

**audit.json:**
```json
{
  "sprint_id": "sprint_abc123",
  "seed_id": "seed_7f3a8b2c",
  "started_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:27:59Z",
  "entries": [
    {
      "timestamp": "2024-01-15T10:00:05Z",
      "agent": "SM",
      "level": "INFO",
      "decision": "Sprint started",
      "context": {
        "seed_id": "seed_7f3a8b2c",
        "requirements_count": 12
      },
      "tokens_used": 0
    },
    {
      "timestamp": "2024-01-15T10:00:10Z",
      "agent": "ANALYST",
      "level": "INFO",
      "decision": "Starting requirement analysis",
      "context": {},
      "tokens_used": 150
    },
    {
      "timestamp": "2024-01-15T10:02:44Z",
      "agent": "ANALYST",
      "level": "INFO",
      "decision": "Crystallized requirement",
      "context": {
        "original": "Users can register with email and password",
        "crystallized_count": 3,
        "confidence": 0.94
      },
      "tokens_used": 2340
    },
    {
      "timestamp": "2024-01-15T10:05:30Z",
      "agent": "ARCHITECT",
      "level": "INFO",
      "decision": "Generated ADR",
      "context": {
        "adr_id": "ADR-001",
        "title": "Authentication Strategy",
        "decision": "JWT with refresh tokens"
      },
      "tokens_used": 3200
    },
    {
      "timestamp": "2024-01-15T10:12:45Z",
      "agent": "DEV",
      "level": "INFO",
      "decision": "Story implementation complete",
      "context": {
        "story_id": "US-001",
        "files_created": 5,
        "test_coverage": 0.89
      },
      "tokens_used": 8900
    },
    {
      "timestamp": "2024-01-15T10:25:55Z",
      "agent": "TEA",
      "level": "INFO",
      "decision": "Deployment approved",
      "context": {
        "coverage": 0.89,
        "confidence": 0.93,
        "risks": []
      },
      "tokens_used": 1200
    }
  ],
  "summary": {
    "total_entries": 145,
    "total_tokens": 145230,
    "total_cost_usd": 1.42,
    "agents": {
      "ANALYST": { "entries": 24, "tokens": 12340 },
      "PM": { "entries": 18, "tokens": 8920 },
      "ARCHITECT": { "entries": 28, "tokens": 15670 },
      "DEV": { "entries": 52, "tokens": 102560 },
      "TEA": { "entries": 12, "tokens": 3400 },
      "SM": { "entries": 11, "tokens": 2340 }
    }
  }
}
```

---

## Next Steps

- [CLI Reference](/yolo-developer/cli/) - All command options
- [Python SDK](/yolo-developer/sdk/) - Programmatic usage
- [Architecture](/yolo-developer/architecture/) - How it works
