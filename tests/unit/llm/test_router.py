"""Unit tests for LLM router (Story 8.2 - Task 2, 8).

Tests LLM routing, tier mapping, and error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yolo_developer.config import (
    LLM_CHEAP_MODEL_DEFAULT,
    OPENAI_CODE_MODEL_DEFAULT,
)
from yolo_developer.config.schema import LLMConfig
from yolo_developer.llm.router import (
    LLMProviderError,
    LLMConfigurationError,
    LLMRouter,
    ModelTier,
    TaskRouting,
    TaskType,
)


class TestLLMRouterInit:
    """Tests for LLMRouter initialization."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default config."""
        config = LLMConfig()
        router = LLMRouter(config)

        assert router.config == config
        assert "routine" in router.model_map
        assert "complex" in router.model_map
        assert "critical" in router.model_map

    def test_init_with_custom_models(self) -> None:
        """Test initialization with custom model names."""
        config = LLMConfig(
            cheap_model="gpt-3.5-turbo",
            premium_model="gpt-4",
            best_model="gpt-4-turbo",
        )
        router = LLMRouter(config)

        assert router.model_map["routine"] == "gpt-3.5-turbo"
        assert router.model_map["complex"] == "gpt-4"
        assert router.model_map["critical"] == "gpt-4-turbo"

    def test_model_map_types(self) -> None:
        """Test that model map has correct structure."""
        config = LLMConfig()
        router = LLMRouter(config)

        assert isinstance(router.model_map, dict)
        assert all(isinstance(k, str) for k in router.model_map.keys())
        assert all(isinstance(v, str) for v in router.model_map.values())


class TestGetModelForTier:
    """Tests for get_model_for_tier method."""

    def test_routine_tier(self) -> None:
        """Test routine tier returns cheap model."""
        config = LLMConfig()
        router = LLMRouter(config)

        model = router.get_model_for_tier("routine")
        assert model == config.cheap_model

    def test_complex_tier(self) -> None:
        """Test complex tier returns premium model."""
        config = LLMConfig()
        router = LLMRouter(config)

        model = router.get_model_for_tier("complex")
        assert model == config.premium_model

    def test_critical_tier(self) -> None:
        """Test critical tier returns best model."""
        config = LLMConfig()
        router = LLMRouter(config)

        model = router.get_model_for_tier("critical")
        assert model == config.best_model

    def test_invalid_tier_raises_key_error(self) -> None:
        """Test that invalid tier raises KeyError."""
        config = LLMConfig()
        router = LLMRouter(config)

        with pytest.raises(KeyError):
            router.get_model_for_tier("invalid")  # type: ignore


class TestLLMRouterCall:
    """Tests for LLMRouter.call method."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        """Create router fixture."""
        return LLMRouter(LLMConfig())

    @pytest.fixture
    def mock_acompletion(self) -> AsyncMock:
        """Create mock for litellm.acompletion."""
        mock = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated code here"
        mock.return_value = mock_response
        return mock

    @pytest.mark.asyncio
    async def test_call_returns_string(
        self, router: LLMRouter, mock_acompletion: AsyncMock
    ) -> None:
        """Test that call returns a string."""
        with patch("litellm.acompletion", mock_acompletion):
            result = await router.call(
                messages=[{"role": "user", "content": "Hello"}],
                tier="routine",
            )
            assert isinstance(result, str)
            assert result == "Generated code here"

    @pytest.mark.asyncio
    async def test_call_uses_correct_model(
        self, router: LLMRouter, mock_acompletion: AsyncMock
    ) -> None:
        """Test that call uses the model for the specified tier."""
        with patch("litellm.acompletion", mock_acompletion):
            await router.call(
                messages=[{"role": "user", "content": "Test"}],
                tier="complex",
            )

            # Check that acompletion was called with premium model
            call_kwargs = mock_acompletion.call_args
            assert call_kwargs.kwargs["model"] == router.config.premium_model

    @pytest.mark.asyncio
    async def test_call_passes_temperature(
        self, router: LLMRouter, mock_acompletion: AsyncMock
    ) -> None:
        """Test that temperature parameter is passed."""
        with patch("litellm.acompletion", mock_acompletion):
            await router.call(
                messages=[{"role": "user", "content": "Test"}],
                tier="routine",
                temperature=0.5,
            )

            call_kwargs = mock_acompletion.call_args
            assert call_kwargs.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_call_passes_max_tokens(
        self, router: LLMRouter, mock_acompletion: AsyncMock
    ) -> None:
        """Test that max_tokens parameter is passed."""
        with patch("litellm.acompletion", mock_acompletion):
            await router.call(
                messages=[{"role": "user", "content": "Test"}],
                tier="routine",
                max_tokens=2048,
            )

            call_kwargs = mock_acompletion.call_args
            assert call_kwargs.kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_call_default_tier_is_routine(
        self, router: LLMRouter, mock_acompletion: AsyncMock
    ) -> None:
        """Test that default tier is routine."""
        with patch("litellm.acompletion", mock_acompletion):
            await router.call(messages=[{"role": "user", "content": "Test"}])

            call_kwargs = mock_acompletion.call_args
            assert call_kwargs.kwargs["model"] == router.config.cheap_model

    @pytest.mark.asyncio
    async def test_call_handles_none_content(
        self, router: LLMRouter, mock_acompletion: AsyncMock
    ) -> None:
        """Test that None content is handled."""
        mock_acompletion.return_value.choices[0].message.content = None
        with patch("litellm.acompletion", mock_acompletion):
            result = await router.call(
                messages=[{"role": "user", "content": "Test"}],
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_call_raises_provider_error_on_failure(self, router: LLMRouter) -> None:
        """Test that provider errors are raised after retries."""
        mock_acompletion = AsyncMock(side_effect=Exception("API Error"))

        with patch("litellm.acompletion", mock_acompletion):
            with pytest.raises(LLMProviderError) as exc_info:
                await router.call(messages=[{"role": "user", "content": "Test"}])

            assert "API Error" in str(exc_info.value)


class TestLLMRouterCallWithFallback:
    """Tests for call_with_fallback method."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        """Create router fixture."""
        return LLMRouter(LLMConfig())

    @pytest.fixture
    def mock_acompletion(self) -> AsyncMock:
        """Create mock for litellm.acompletion."""
        mock = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fallback response"
        mock.return_value = mock_response
        return mock

    @pytest.mark.asyncio
    async def test_fallback_returns_primary_on_success(
        self, router: LLMRouter, mock_acompletion: AsyncMock
    ) -> None:
        """Test that fallback returns primary result when successful."""
        mock_acompletion.return_value.choices[0].message.content = "Primary response"
        with patch("litellm.acompletion", mock_acompletion):
            result = await router.call_with_fallback(
                messages=[{"role": "user", "content": "Test"}],
                primary_tier="complex",
                fallback_tier="routine",
            )
            assert result == "Primary response"

    @pytest.mark.asyncio
    async def test_fallback_uses_fallback_tier_on_failure(self, router: LLMRouter) -> None:
        """Test that fallback tier is used when primary fails."""
        models_used: list[str] = []

        async def mock_call(*args: object, **kwargs: object) -> MagicMock:
            model = str(kwargs.get("model", ""))
            models_used.append(model)

            # Fail on complex model (claude-sonnet)
            if "claude-sonnet" in model:
                raise Exception("Primary failed")

            # Succeed on routine model
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Fallback response"
            return mock_response

        with patch("litellm.acompletion", mock_call):
            result = await router.call_with_fallback(
                messages=[{"role": "user", "content": "Test"}],
                primary_tier="complex",
                fallback_tier="routine",
            )
            assert result == "Fallback response"
            # Should have tried complex model (with retries) then routine
            assert any(LLM_CHEAP_MODEL_DEFAULT in m for m in models_used)


class TestLLMRouterTaskRouting:
    """Tests for task-based routing behavior."""

    def test_task_routing_uses_openai_code_model_in_hybrid(self) -> None:
        """Code tasks should use OpenAI code_model when hybrid routing is enabled."""
        config = LLMConfig(
            provider="hybrid",
            hybrid={"enabled": True},
            openai={"code_model": OPENAI_CODE_MODEL_DEFAULT},
        )
        router = LLMRouter(config)

        task_type: TaskType = "code_generation"
        routing = router.get_task_routing(task_type)

        assert isinstance(routing, TaskRouting)
        assert routing.provider == "openai"
        assert routing.model == OPENAI_CODE_MODEL_DEFAULT
        assert routing.task_type == "code_generation"

    def test_task_routing_uses_anthropic_for_architecture_in_hybrid(self) -> None:
        """Architecture tasks should route to Anthropic by default in hybrid."""
        config = LLMConfig(provider="hybrid", hybrid={"enabled": True})
        router = LLMRouter(config)

        routing = router.get_task_routing("architecture")

        assert routing.provider == "anthropic"
        assert routing.task_type == "architecture"

    def test_task_routing_defaults_to_tier_mapping_when_auto(self) -> None:
        """Auto provider should follow legacy tier mapping for tasks."""
        config = LLMConfig()
        router = LLMRouter(config)

        routing = router.get_task_routing("documentation")

        assert routing.tier == "routine"
        assert routing.model == config.cheap_model


class TestModelTierType:
    """Tests for ModelTier type."""

    def test_valid_tiers(self) -> None:
        """Test that valid tier values are accepted."""
        valid_tiers: list[ModelTier] = ["routine", "complex", "critical"]
        for tier in valid_tiers:
            config = LLMConfig()
            router = LLMRouter(config)
            # Should not raise
            _ = router.get_model_for_tier(tier)


class TestLLMExceptions:
    """Tests for LLM exception classes."""

    def test_llm_router_error_is_exception(self) -> None:
        """Test that LLMRouterError is an Exception."""
        from yolo_developer.llm.router import LLMRouterError

        assert issubclass(LLMRouterError, Exception)

    def test_llm_provider_error_is_router_error(self) -> None:
        """Test that LLMProviderError inherits from LLMRouterError."""
        from yolo_developer.llm.router import LLMProviderError, LLMRouterError

        assert issubclass(LLMProviderError, LLMRouterError)

    def test_llm_configuration_error_is_router_error(self) -> None:
        """Test that LLMConfigurationError inherits from LLMRouterError."""
        from yolo_developer.llm.router import LLMConfigurationError, LLMRouterError

        assert issubclass(LLMConfigurationError, LLMRouterError)

    def test_provider_error_message(self) -> None:
        """Test that provider error preserves message."""
        error = LLMProviderError("Test error message")
        assert str(error) == "Test error message"

    def test_configuration_error_message(self) -> None:
        """Test that configuration error preserves message."""
        error = LLMConfigurationError("Missing API key")
        assert str(error) == "Missing API key"


class TestLLMRouterImportError:
    """Tests for ImportError handling when LiteLLM is not installed."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        """Create router fixture."""
        return LLMRouter(LLMConfig())

    @pytest.mark.asyncio
    async def test_call_raises_configuration_error_on_import_error(self, router: LLMRouter) -> None:
        """Test that ImportError is converted to LLMConfigurationError."""
        import importlib
        import sys

        # Remove litellm from sys.modules to force re-import
        litellm_module = sys.modules.pop("litellm", None)
        litellm_acompletion = sys.modules.pop("litellm.acompletion", None)

        try:
            # Patch the import to raise ImportError
            def raise_import_error(*args: object, **kwargs: object) -> None:
                raise ImportError("No module named 'litellm'")

            with patch.object(importlib, "import_module", side_effect=raise_import_error):
                # Since litellm uses `from litellm import acompletion` we need
                # to patch at a different level - directly mock the import statement
                with patch.dict(sys.modules, {"litellm": None}):
                    # The router uses lazy import, so we need to trigger it
                    # by calling router.call which does `from litellm import acompletion`
                    with pytest.raises(LLMConfigurationError) as exc_info:
                        await router.call(messages=[{"role": "user", "content": "Test"}])

                    assert "litellm" in str(exc_info.value).lower()
        finally:
            # Restore litellm module
            if litellm_module:
                sys.modules["litellm"] = litellm_module
            if litellm_acompletion:
                sys.modules["litellm.acompletion"] = litellm_acompletion
