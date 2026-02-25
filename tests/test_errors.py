"""Tests for AUTOCOG error hierarchy."""

import pytest

from src.errors import (
    AutocogError,
    EngineStateError,
    LLMError,
    LLMParseError,
    LLMTimeoutError,
    SerializationError,
    StrategyError,
    ValidationError,
)


def test_autocog_error_hierarchy():
    """Test error inheritance hierarchy."""
    # All custom errors inherit from AutocogError
    assert issubclass(StrategyError, AutocogError)
    assert issubclass(LLMError, StrategyError)
    assert issubclass(LLMParseError, LLMError)
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(SerializationError, AutocogError)
    assert issubclass(ValidationError, AutocogError)
    assert issubclass(EngineStateError, AutocogError)

    # All inherit from Exception
    assert issubclass(AutocogError, Exception)


def test_llm_parse_error_attributes():
    """Test LLMParseError stores response and method."""
    raw_response = '{"goal": "explore", "reason": "need to find water"}'
    parse_method = "json.loads"

    error = LLMParseError(raw_response, parse_method)

    assert error.raw_response == raw_response
    assert error.parse_method == parse_method


def test_llm_parse_error_message():
    """Test LLMParseError generates correct message."""
    error = LLMParseError("invalid json", "json_extraction")

    assert "json_extraction" in str(error)
    assert "Failed to parse LLM response" in str(error)


def test_strategy_error_can_be_raised():
    """Test StrategyError can be raised and caught."""
    with pytest.raises(StrategyError):
        raise StrategyError("Strategy failed")


def test_llm_timeout_error_can_be_raised():
    """Test LLMTimeoutError can be raised and caught."""
    with pytest.raises(LLMTimeoutError):
        raise LLMTimeoutError("LLM call timed out after 30s")


def test_serialization_error_can_be_raised():
    """Test SerializationError can be raised and caught."""
    with pytest.raises(SerializationError):
        raise SerializationError("Failed to serialize state")


def test_validation_error_can_be_raised():
    """Test ValidationError can be raised and caught."""
    with pytest.raises(ValidationError):
        raise ValidationError("Invalid input at boundary")


def test_engine_state_error_can_be_raised():
    """Test EngineStateError can be raised and caught."""
    with pytest.raises(EngineStateError):
        raise EngineStateError("Engine not initialized")


def test_catch_by_base_class():
    """Test errors can be caught by base class."""
    with pytest.raises(AutocogError):
        raise LLMParseError("test", "test_method")

    with pytest.raises(StrategyError):
        raise LLMError("test")

    with pytest.raises(LLMError):
        raise LLMTimeoutError("test")


def test_llm_parse_error_repr():
    """Test LLMParseError has useful repr."""
    error = LLMParseError("invalid", "json")
    repr_str = repr(error)

    # Should contain class name
    assert "LLMParseError" in repr_str
