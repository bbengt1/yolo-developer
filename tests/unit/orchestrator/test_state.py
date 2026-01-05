"""Unit tests for orchestrator state module.

Tests cover:
- YoloState TypedDict structure
- Message accumulation behavior with add_messages reducer
- Agent attribution helper function
- State field defaults and optional fields
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from yolo_developer.orchestrator.state import (
    YoloState,
    create_agent_message,
    get_messages_reducer,
)


class TestYoloState:
    """Tests for YoloState TypedDict."""

    def test_state_can_be_created_with_required_fields(self) -> None:
        """YoloState should be creatable with required fields."""
        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }
        assert state["messages"] == []
        assert state["current_agent"] == "analyst"

    def test_state_accepts_messages_list(self) -> None:
        """YoloState should accept list of BaseMessage."""
        messages: list[BaseMessage] = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        state: YoloState = {
            "messages": messages,
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }
        assert len(state["messages"]) == 2

    def test_state_accepts_handoff_context(self) -> None:
        """YoloState should accept HandoffContext."""
        from yolo_developer.orchestrator.context import HandoffContext

        context = HandoffContext(source_agent="analyst", target_agent="pm")
        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": context,
            "decisions": [],
        }
        assert state["handoff_context"] is not None
        assert state["handoff_context"].source_agent == "analyst"

    def test_state_accepts_decisions_list(self) -> None:
        """YoloState should accept list of Decision."""
        from yolo_developer.orchestrator.context import Decision

        decisions = [
            Decision(agent="analyst", summary="test", rationale="reason"),
        ]
        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": decisions,
        }
        assert len(state["decisions"]) == 1


class TestMessagesReducer:
    """Tests for message accumulation with add_messages reducer."""

    def test_reducer_accumulates_messages(self) -> None:
        """Messages should be accumulated, not replaced."""
        reducer = get_messages_reducer()

        # Initial messages
        initial: list[BaseMessage] = [HumanMessage(content="First")]

        # New messages to add
        new_messages: list[BaseMessage] = [AIMessage(content="Second")]

        # Apply reducer
        result = reducer(initial, new_messages)

        assert len(result) == 2
        assert result[0].content == "First"
        assert result[1].content == "Second"

    def test_reducer_preserves_order(self) -> None:
        """Messages should maintain insertion order."""
        reducer = get_messages_reducer()

        messages: list[BaseMessage] = []
        for i in range(5):
            messages = reducer(messages, [HumanMessage(content=f"msg-{i}")])

        assert len(messages) == 5
        for i, msg in enumerate(messages):
            assert msg.content == f"msg-{i}"

    def test_reducer_handles_empty_initial(self) -> None:
        """Reducer should handle empty initial list."""
        reducer = get_messages_reducer()

        result = reducer([], [HumanMessage(content="First")])

        assert len(result) == 1
        assert result[0].content == "First"

    def test_reducer_handles_empty_new(self) -> None:
        """Reducer should handle empty new messages list."""
        reducer = get_messages_reducer()

        initial: list[BaseMessage] = [HumanMessage(content="Existing")]
        result = reducer(initial, [])

        assert len(result) == 1
        assert result[0].content == "Existing"


class TestCreateAgentMessage:
    """Tests for create_agent_message helper function."""

    def test_creates_ai_message_with_content(self) -> None:
        """create_agent_message should create AIMessage with content."""
        msg = create_agent_message(
            content="Analysis complete",
            agent="analyst",
        )
        assert isinstance(msg, AIMessage)
        assert msg.content == "Analysis complete"

    def test_includes_agent_in_metadata(self) -> None:
        """create_agent_message should include agent in additional_kwargs."""
        msg = create_agent_message(
            content="Story created",
            agent="pm",
        )
        assert msg.additional_kwargs.get("agent") == "pm"

    def test_includes_custom_metadata(self) -> None:
        """create_agent_message should include custom metadata."""
        msg = create_agent_message(
            content="Design complete",
            agent="architect",
            metadata={"decision_count": 3, "risk_level": "low"},
        )
        assert msg.additional_kwargs.get("decision_count") == 3
        assert msg.additional_kwargs.get("risk_level") == "low"

    def test_agent_does_not_override_custom_metadata(self) -> None:
        """Agent name should be set even if metadata provided."""
        msg = create_agent_message(
            content="Test",
            agent="dev",
            metadata={"other": "value"},
        )
        assert msg.additional_kwargs.get("agent") == "dev"
        assert msg.additional_kwargs.get("other") == "value"
