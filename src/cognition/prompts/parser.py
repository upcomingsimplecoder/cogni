"""LLM response parser — robust JSON extraction and validation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Valid goal strings that agents can choose
VALID_GOALS = [
    "rest",
    "satisfy_hunger",
    "satisfy_thirst",
    "explore",
    "socialize",
    "attack",
    "share_resources",
    "gather_opportunistic",
    "flee",
    "wait",
]


@dataclass
class LLMDecision:
    """Parsed LLM decision output.

    Attributes:
        goal: Primary goal string (must be in VALID_GOALS)
        reason: Brief explanation from LLM
        target_agent: Optional target agent ID for social/combat goals
        confidence: Parser confidence (0-1) based on parsing success
        raw_text: Original LLM response for debugging
    """

    goal: str
    reason: str
    target_agent: object | None = None
    confidence: float = 1.0
    raw_text: str = ""


class LLMResponseParser:
    """Robust parser for LLM decision outputs.

    Handles:
    - Clean JSON responses
    - JSON wrapped in markdown code fences
    - Malformed JSON with extractable goal/reason
    - Verbose prose responses with goal keywords
    - Synonym mapping for common variations
    """

    # Maps common goal synonyms to valid goals
    GOAL_SYNONYMS = {
        "satisfy_hunger": ["eat", "food", "hunger", "starving", "berries", "consume_hunger"],
        "satisfy_thirst": ["drink", "water", "thirst", "dehydrat", "consume_thirst"],
        "rest": ["sleep", "recover", "tired", "recharge", "satisfy_energy"],
        "explore": ["scout", "discover", "wander", "search", "move_to"],
        "attack": ["fight", "aggress", "hostile", "combat", "eliminate", "approach_threaten"],
        "socialize": ["cooperat", "communicat", "interact", "ally", "befriend"],
        "share_resources": ["share", "give", "donat", "help", "trade"],
        "gather_opportunistic": ["gather", "forage", "collect", "harvest", "acquire"],
        "flee": ["escape", "retreat", "run", "avoid"],
    }

    def parse(self, response_text: str) -> LLMDecision:
        """Parse LLM response into a validated LLMDecision.

        Args:
            response_text: Raw text from LLM completion

        Returns:
            LLMDecision with validated goal and metadata

        Strategy:
        1. Try JSON extraction (with/without markdown fences)
        2. Try regex pattern matching for goal/reason fields
        3. Fall back to keyword scanning with synonyms
        4. Ultimate fallback: "wait" goal with low confidence
        """
        # Strategy 1: JSON extraction
        result = self._extract_json(response_text)
        if result:
            result.raw_text = response_text
            return result

        # Strategy 2: Regex pattern matching
        result = self._extract_regex_patterns(response_text)
        if result:
            result.raw_text = response_text
            return result

        # Strategy 3: Keyword scanning
        goal = self._extract_goal_keyword(response_text)
        if goal:
            logger.info(f"Extracted goal via keyword scan: {goal}")
            return LLMDecision(
                goal=goal,
                reason="Extracted from verbose response",
                confidence=0.5,
                raw_text=response_text,
            )

        # Ultimate fallback
        logger.warning(f"Could not parse LLM response, defaulting to 'wait': {response_text[:100]}")
        return LLMDecision(
            goal="wait",
            reason="Failed to parse response",
            confidence=0.1,
            raw_text=response_text,
        )

    def validate(self, decision: LLMDecision) -> LLMDecision:
        """Validate and normalize a parsed decision.

        Args:
            decision: Parsed decision to validate

        Returns:
            Validated decision (may have modified goal if invalid)
        """
        # Normalize goal to valid list
        if decision.goal not in VALID_GOALS:
            # Try synonym mapping
            normalized = self._map_synonym(decision.goal)
            if normalized:
                logger.info(f"Normalized goal '{decision.goal}' -> '{normalized}'")
                decision.goal = normalized
            else:
                logger.warning(f"Invalid goal '{decision.goal}', defaulting to 'wait'")
                decision.goal = "wait"
                decision.confidence *= 0.5

        # Ensure reason exists
        if not decision.reason or decision.reason.strip() == "":
            decision.reason = "No reason provided"
            decision.confidence *= 0.8

        return decision

    def _extract_json(self, text: str) -> LLMDecision | None:
        """Try to extract JSON from response (with or without markdown fences).

        Returns:
            LLMDecision if successful, None otherwise
        """
        # Remove markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        if start < 0 or end <= start:
            return None

        try:
            data = json.loads(cleaned[start:end])
            goal = data.get("goal", "").strip().lower()
            reason = data.get("reason", "").strip()
            target = data.get("target_agent")

            if not goal:
                return None

            # Normalize goal
            goal = self._map_synonym(goal) or goal

            decision = LLMDecision(
                goal=goal,
                reason=reason or "No reason provided",
                target_agent=target,
                confidence=0.95,
            )
            return self.validate(decision)

        except (json.JSONDecodeError, KeyError, AttributeError):
            return None

    def _extract_regex_patterns(self, text: str) -> LLMDecision | None:
        """Try to extract goal/reason using regex patterns.

        Handles cases where JSON is malformed but fields are present.
        """
        goal_match = re.search(r'"?goal"?\s*:\s*"?([^",}\n]+)"?', text, re.IGNORECASE)
        reason_match = re.search(r'"?reason"?\s*:\s*"?([^",}\n]+)"?', text, re.IGNORECASE)

        if not goal_match:
            return None

        goal = goal_match.group(1).strip().lower()
        reason = (
            reason_match.group(1).strip() if reason_match else "Extracted from partial response"
        )

        # Normalize goal
        goal = self._map_synonym(goal) or goal

        decision = LLMDecision(
            goal=goal,
            reason=reason,
            confidence=0.7,
        )
        return self.validate(decision)

    def _extract_goal_keyword(self, text: str) -> str | None:
        """Scan text for goal keywords, prioritizing specificity.

        Returns:
            Valid goal string if found, None otherwise
        """
        text_lower = text.lower()

        # Check exact matches first (most specific)
        for goal in VALID_GOALS:
            # Match whole word or with underscores replaced by spaces
            pattern = goal.replace("_", r"[\s_]")
            if re.search(rf"\b{pattern}\b", text_lower):
                return goal

        # Check synonyms
        for goal, synonyms in self.GOAL_SYNONYMS.items():
            for syn in synonyms:
                if syn in text_lower:
                    return goal

        return None

    def _map_synonym(self, goal_text: str) -> str | None:
        """Map a goal text to a valid goal using synonym dict.

        Args:
            goal_text: Goal string from LLM (may be synonym)

        Returns:
            Valid goal string if mapping found, None otherwise
        """
        goal_text = goal_text.lower().strip()

        # Exact match
        if goal_text in VALID_GOALS:
            return goal_text

        # Synonym lookup
        for valid_goal, synonyms in self.GOAL_SYNONYMS.items():
            if goal_text in synonyms:
                return valid_goal
            # Partial match (for "eating" matching "eat")
            for syn in synonyms:
                if syn in goal_text or goal_text in syn:
                    return valid_goal

        return None
