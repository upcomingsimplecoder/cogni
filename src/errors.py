"""Structured error hierarchy for AUTOCOG."""


class AutocogError(Exception):
    """Base for all AUTOCOG errors."""

    pass


class StrategyError(AutocogError):
    """Decision strategy failed."""

    pass


class LLMError(StrategyError):
    """LLM-specific failure."""

    pass


class LLMParseError(LLMError):
    """Failed to parse LLM response."""

    def __init__(self, raw_response: str, parse_method: str):
        self.raw_response = raw_response
        self.parse_method = parse_method
        super().__init__(f"Failed to parse LLM response via {parse_method}")


class LLMTimeoutError(LLMError):
    """LLM call timed out."""

    pass


class SerializationError(AutocogError):
    """State serialization/deserialization failed."""

    pass


class ValidationError(AutocogError):
    """Input validation at boundary failed."""

    pass


class EngineStateError(AutocogError):
    """Engine in invalid state for requested operation."""

    pass
