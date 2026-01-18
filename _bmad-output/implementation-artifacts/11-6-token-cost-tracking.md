# Story 11.6: Token/Cost Tracking

## Story

**As a** developer,
**I want** token usage and costs tracked,
**So that** I can monitor and optimize spending.

## Status

- **Epic:** 11 - Audit Trail & Observability
- **Status:** done
- **Priority:** P2
- **Story Points:** 5

## Acceptance Criteria

### AC1: Tokens Per Call Are Recorded
**Given** LLM calls being made
**When** tracking is active
**Then** tokens per call are recorded (prompt_tokens, completion_tokens, total_tokens)

### AC2: Costs Are Calculated
**Given** token usage data from LLM calls
**When** a call completes
**Then** cost in USD is calculated and stored

### AC3: Per-Agent Breakdown Is Available
**Given** multiple agents making LLM calls
**When** requesting cost breakdown
**Then** costs are available grouped by agent name

### AC4: Per-Story Breakdown Is Available
**Given** LLM calls associated with stories
**When** requesting cost breakdown
**Then** costs are available grouped by story_id

### AC5: Totals Are Aggregated
**Given** multiple LLM calls with tracked costs
**When** requesting totals
**Then** aggregate totals are calculated for session, sprint, and all-time

## Technical Requirements

### Functional Requirements Mapping
- **FR86:** System can track token usage and cost per operation

### Architecture References
- **ADR-003:** LiteLLM for unified provider access with built-in cost tracking
- **ADR-001:** Frozen dataclasses for internal state (token usage records)
- **Epic 11 Pattern:** Protocol-based stores, structlog logging, factory functions

### Technology Stack
- **LiteLLM:** Built-in token counting and cost calculation
  - `response.usage` contains prompt_tokens, completion_tokens, total_tokens
  - `response._hidden_params["response_cost"]` contains USD cost
  - `completion_cost()` function for cost calculation
  - `cost_per_token()` function for per-token pricing
- **structlog:** For structured logging of token/cost events
- **Frozen Dataclasses:** For immutable token usage records

## Tasks

### Task 1: Define Token/Cost Types (cost_types.py)
**File:** `src/yolo_developer/audit/cost_types.py`

Create frozen dataclasses for token and cost tracking:

```python
@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass(frozen=True)
class CostRecord:
    """Cost record for a single LLM call."""
    id: str
    timestamp: str  # ISO 8601
    model: str
    tier: str  # "routine", "complex", "critical"
    token_usage: TokenUsage
    cost_usd: float  # In USD
    agent_name: str
    session_id: str
    story_id: str | None = None
    sprint_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CostAggregation:
    """Aggregated cost statistics."""
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    call_count: int
    models: tuple[str, ...]
    period_start: str | None = None
    period_end: str | None = None
```

**Subtasks:**
1. Create Literal type for `CostGroupBy = Literal["agent", "story", "sprint", "model", "tier"]`
2. Implement `TokenUsage` dataclass with to_dict() method
3. Implement `CostRecord` dataclass with validation in __post_init__
4. Implement `CostAggregation` dataclass with to_dict() method
5. Add constants: `VALID_GROUPBY_VALUES: frozenset[str]`

### Task 2: Define Cost Store Protocol (cost_store.py)
**File:** `src/yolo_developer/audit/cost_store.py`

Create Protocol for cost storage following existing store patterns:

```python
class CostFilters(TypedDict, total=False):
    """Filters for querying cost records."""
    agent_name: str | None
    story_id: str | None
    sprint_id: str | None
    session_id: str | None
    model: str | None
    tier: str | None
    start_time: str | None
    end_time: str | None

class CostStore(Protocol):
    """Protocol for cost record storage."""

    async def store_cost(self, record: CostRecord) -> None:
        """Store a cost record."""
        ...

    async def get_cost(self, cost_id: str) -> CostRecord | None:
        """Retrieve a cost record by ID."""
        ...

    async def get_costs(self, filters: CostFilters | None = None) -> list[CostRecord]:
        """Retrieve cost records with optional filtering."""
        ...

    async def get_aggregation(
        self,
        filters: CostFilters | None = None,
    ) -> CostAggregation:
        """Get aggregated cost statistics."""
        ...

    async def get_grouped_aggregation(
        self,
        group_by: str,
        filters: CostFilters | None = None,
    ) -> dict[str, CostAggregation]:
        """Get aggregated costs grouped by a dimension."""
        ...
```

**Subtasks:**
1. Create `CostFilters` TypedDict for query filtering
2. Define `CostStore` Protocol with async methods
3. Document protocol contract in docstrings

### Task 3: Implement In-Memory Cost Store (cost_memory_store.py)
**File:** `src/yolo_developer/audit/cost_memory_store.py`

Implement thread-safe in-memory store following `InMemoryCorrelationStore` pattern:

```python
class InMemoryCostStore:
    """Thread-safe in-memory implementation of CostStore."""

    def __init__(self) -> None:
        self._costs: dict[str, CostRecord] = {}
        self._lock = asyncio.Lock()
```

**Subtasks:**
1. Implement `store_cost()` with lock protection
2. Implement `get_cost()` by ID
3. Implement `get_costs()` with filtering support
4. Implement `get_aggregation()` for total statistics
5. Implement `get_grouped_aggregation()` for per-dimension breakdowns
6. Implement helper method `_matches_filters()` for filtering logic
7. Implement helper method `_aggregate()` for computing statistics

### Task 4: Create Cost Tracking Service (cost_service.py)
**File:** `src/yolo_developer/audit/cost_service.py`

Create service that integrates with LLMRouter to track costs:

```python
class CostTrackingService:
    """Service for tracking LLM token usage and costs."""

    def __init__(
        self,
        cost_store: CostStore,
        enabled: bool = True,
    ) -> None:
        self._cost_store = cost_store
        self._enabled = enabled

    async def record_llm_call(
        self,
        model: str,
        tier: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        agent_name: str,
        session_id: str,
        story_id: str | None = None,
        sprint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record token usage and cost for an LLM call."""
        ...

    async def get_agent_costs(
        self,
        agent_name: str | None = None,
    ) -> dict[str, CostAggregation]:
        """Get cost breakdown by agent."""
        ...

    async def get_story_costs(
        self,
        story_id: str | None = None,
    ) -> dict[str, CostAggregation]:
        """Get cost breakdown by story."""
        ...

    async def get_session_total(
        self,
        session_id: str,
    ) -> CostAggregation:
        """Get total costs for a session."""
        ...

    async def get_sprint_total(
        self,
        sprint_id: str,
    ) -> CostAggregation:
        """Get total costs for a sprint."""
        ...
```

**Subtasks:**
1. Implement `record_llm_call()` with UUID generation and timestamp
2. Implement `get_agent_costs()` using grouped aggregation
3. Implement `get_story_costs()` using grouped aggregation
4. Implement `get_session_total()` with session filter
5. Implement `get_sprint_total()` with sprint filter
6. Add structlog logging for all operations
7. Create factory function `get_cost_tracking_service()`

### Task 5: Add LiteLLM Integration Helper (cost_utils.py)
**File:** `src/yolo_developer/audit/cost_utils.py`

Create utilities for extracting cost data from LiteLLM responses:

```python
def extract_token_usage(response: Any) -> TokenUsage:
    """Extract token usage from LiteLLM response.

    Args:
        response: Response object from litellm.completion/acompletion

    Returns:
        TokenUsage with prompt, completion, and total tokens
    """
    usage = response.usage
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )

def extract_cost(response: Any) -> float:
    """Extract cost in USD from LiteLLM response.

    Args:
        response: Response object from litellm.completion/acompletion

    Returns:
        Cost in USD (float)

    Note:
        Falls back to 0.0 if cost not available in response.
    """
    try:
        return float(response._hidden_params.get("response_cost", 0.0))
    except (AttributeError, KeyError, TypeError):
        return 0.0
```

**Subtasks:**
1. Implement `extract_token_usage()` with error handling
2. Implement `extract_cost()` with fallback to 0.0
3. Add `calculate_cost_if_missing()` using `litellm.completion_cost()`
4. Document LiteLLM response structure expectations

### Task 6: Update Module Exports (__init__.py)
**File:** `src/yolo_developer/audit/__init__.py`

Export new cost tracking types and services:

```python
from yolo_developer.audit.cost_types import (
    CostAggregation,
    CostGroupBy,
    CostRecord,
    TokenUsage,
)
from yolo_developer.audit.cost_store import (
    CostFilters,
    CostStore,
)
from yolo_developer.audit.cost_memory_store import InMemoryCostStore
from yolo_developer.audit.cost_service import (
    CostTrackingService,
    get_cost_tracking_service,
)
from yolo_developer.audit.cost_utils import (
    extract_cost,
    extract_token_usage,
)
```

**Subtasks:**
1. Add imports for all new cost tracking modules
2. Update `__all__` list with new exports
3. Ensure proper import order (types, protocol, store, service, utils)

## Dev Notes

### LiteLLM Integration Details

From [LiteLLM Token Usage Documentation](https://docs.litellm.ai/docs/completion/token_usage):

1. **Token Usage Access:**
   ```python
   response = await acompletion(model="...", messages=[...])
   prompt_tokens = response.usage.prompt_tokens
   completion_tokens = response.usage.completion_tokens
   total_tokens = response.usage.total_tokens
   ```

2. **Cost Access:**
   ```python
   # From response
   cost = response._hidden_params.get("response_cost", 0.0)

   # Or using completion_cost function
   from litellm import completion_cost
   cost = completion_cost(completion_response=response)
   ```

3. **Model Pricing:**
   - LiteLLM maintains pricing via api.litellm.ai
   - `cost_per_token(model, prompt_tokens, completion_tokens)` for manual calculation

### Implementation Patterns from Epic 11

Follow patterns established in Stories 11.1-11.5:

1. **Frozen Dataclasses:** All data types are immutable (`@dataclass(frozen=True)`)
2. **Protocol Pattern:** Use `Protocol` for store abstraction
3. **Factory Functions:** `get_<service>()` pattern for creating instances
4. **Validation:** Use `__post_init__` for dataclass validation with logging warnings
5. **Structured Logging:** Use structlog with consistent event names
6. **Thread Safety:** Use `asyncio.Lock()` for in-memory stores
7. **TypedDict for Filters:** Use TypedDict with `total=False` for optional filters

### Example Usage

```python
from yolo_developer.audit.cost_service import get_cost_tracking_service
from yolo_developer.audit.cost_memory_store import InMemoryCostStore
from yolo_developer.audit.cost_utils import extract_token_usage, extract_cost

# Setup
cost_store = InMemoryCostStore()
cost_service = get_cost_tracking_service(cost_store)

# After LLM call (in LLMRouter or agent)
token_usage = extract_token_usage(response)
cost_usd = extract_cost(response)

record = await cost_service.record_llm_call(
    model="gpt-4o-mini",
    tier="routine",
    prompt_tokens=token_usage.prompt_tokens,
    completion_tokens=token_usage.completion_tokens,
    cost_usd=cost_usd,
    agent_name="analyst",
    session_id="session-123",
    story_id="1-2-user-auth",
)

# Query costs
agent_breakdown = await cost_service.get_agent_costs()
# Returns: {"analyst": CostAggregation(...), "pm": CostAggregation(...)}

story_breakdown = await cost_service.get_story_costs()
# Returns: {"1-2-user-auth": CostAggregation(...)}

session_total = await cost_service.get_session_total("session-123")
# Returns: CostAggregation with totals
```

### Testing Strategy

1. **Unit Tests for Types:** Test dataclass creation, validation, to_dict()
2. **Unit Tests for Store:** Test CRUD operations, filtering, aggregation
3. **Unit Tests for Service:** Test all service methods with mocked store
4. **Unit Tests for Utils:** Test extraction from mock LiteLLM responses
5. **Integration Test:** Test full flow from LLM call to cost aggregation

### File Structure

```
src/yolo_developer/audit/
├── cost_types.py          # TokenUsage, CostRecord, CostAggregation
├── cost_store.py          # CostFilters, CostStore Protocol
├── cost_memory_store.py   # InMemoryCostStore implementation
├── cost_service.py        # CostTrackingService, factory function
├── cost_utils.py          # extract_token_usage, extract_cost
└── __init__.py            # Updated exports

tests/unit/audit/
├── test_cost_types.py
├── test_cost_store.py
├── test_cost_memory_store.py
├── test_cost_service.py
└── test_cost_utils.py
```

## Definition of Done

- [x] All acceptance criteria implemented and verified
- [x] Unit tests for all new modules with >90% coverage
- [x] Type hints on all public functions (mypy passes)
- [x] Code formatted with ruff
- [x] Docstrings following Google style on all public APIs
- [x] Integration with existing audit module exports
- [x] No breaking changes to existing audit functionality

## Dev Agent Record

### Files Created
- `src/yolo_developer/audit/cost_types.py` - TokenUsage, CostRecord, CostAggregation dataclasses
- `src/yolo_developer/audit/cost_store.py` - CostFilters TypedDict and CostStore Protocol
- `src/yolo_developer/audit/cost_memory_store.py` - InMemoryCostStore implementation
- `src/yolo_developer/audit/cost_service.py` - CostTrackingService and factory function
- `src/yolo_developer/audit/cost_utils.py` - LiteLLM extraction utilities
- `tests/unit/audit/test_cost_types.py` - Unit tests for cost types
- `tests/unit/audit/test_cost_store.py` - Unit tests for CostFilters
- `tests/unit/audit/test_cost_memory_store.py` - Unit tests for InMemoryCostStore
- `tests/unit/audit/test_cost_service.py` - Unit tests for CostTrackingService
- `tests/unit/audit/test_cost_utils.py` - Unit tests for extraction utilities

### Files Modified
- `src/yolo_developer/audit/__init__.py` - Added exports for new cost tracking modules

### Test Coverage
- 85 tests passing across all new modules
- 97-100% coverage on new cost modules

## References

- [LiteLLM Token Usage Documentation](https://docs.litellm.ai/docs/completion/token_usage)
- [LiteLLM Spend Tracking](https://docs.litellm.ai/docs/proxy/cost_tracking)
- Epic 11: Audit Trail & Observability requirements
- ADR-003: LLM Provider Abstraction (LiteLLM)
- Story 11.5: Cross-Agent Correlation (pattern reference)
