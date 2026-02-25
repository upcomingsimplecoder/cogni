"""LLM-powered decision strategy using any OpenAI-compatible endpoint."""

from __future__ import annotations

import logging
import time

from src.awareness.types import Expression, Intention, Reflection, Sensation
from src.cognition.prompts import (
    VALID_GOALS,
    ContextBuilder,
    LLMResponseParser,
    build_system_prompt,
)
from src.cognition.strategies.personality import PersonalityStrategy
from src.errors import LLMParseError
from src.memory.episodic import EpisodicMemory
from src.memory.social import SocialMemory

logger = logging.getLogger(__name__)


class LLMStrategy:
    """Decision strategy powered by LLM calls via any OpenAI-compatible endpoint.

    Works with any provider that exposes an OpenAI-compatible chat completions API:
    Anthropic, OpenAI, Ollama, vLLM, LM Studio, Together AI, Groq, etc.

    Features:
    - Uses OpenAI-compatible API at configurable endpoint
    - Archetype-aware system prompts with personality traits
    - Structured context building with memory integration
    - Robust response parsing with synonym mapping
    - Tiered model selection (fast/standard/deep) based on situation complexity
    - Prompt caching to reduce token costs
    - Prefill support for JSON-constrained responses
    - Falls back to PersonalityStrategy if LLM unavailable or calls fail
    - Rate limited: only calls LLM every `call_interval` ticks

    Model Tiers:
    - Fast (cheap_model): Routine decisions, low stakes, no threats
    - Standard (model): Moderate complexity, social interactions
    - Deep (model): High stakes, combat, critical needs, complex planning

    Prompt Versions:
    - v1_minimal: Basic status/needs (lightweight, for testing)
    - v2_rich: Full context with memory, agents, resources (default)
    """

    def __init__(
        self,
        base_url: str = "",
        model: str = "opus",
        call_interval: int = 5,
        cheap_model: str | None = None,
        archetype: str | None = None,
        episodic_memory: EpisodicMemory | None = None,
        social_memory: SocialMemory | None = None,
        prompt_version: str = "v2_rich",
        api_key: str = "",
    ):
        """Initialize LLM strategy.

        Args:
            base_url: OpenAI-compatible API endpoint (e.g. https://api.openai.com/v1)
            model: Primary model name (provider-dependent, e.g. "gpt-4o", "opus")
            call_interval: Ticks between LLM calls (uses fallback between)
            cheap_model: Fast model for routine decisions (defaults to model if not set)
            archetype: Optional personality archetype (e.g., "gatherer", "explorer")
            episodic_memory: Optional episodic memory for context
            social_memory: Optional social memory for context
            prompt_version: Prompt template version ("v1_minimal" | "v2_rich")
            api_key: API key for the endpoint (some local endpoints don't need one)
        """
        self.base_url = base_url
        self.model = model
        self.cheap_model = cheap_model or model
        self.call_interval = call_interval
        self.archetype = archetype
        self.prompt_version = prompt_version
        self.api_key = api_key

        self._fallback = PersonalityStrategy()
        self._client = None
        self._tick_counter = 0
        self._prefill_supported: bool | None = None  # None = untested

        # Model tier mapping
        self._model_map = {
            "fast": self.cheap_model,
            "standard": model,
            "deep": model,
        }

        # Prompt components
        self._context_builder = ContextBuilder(
            episodic_memory=episodic_memory,
            social_memory=social_memory,
        )
        self._parser = LLMResponseParser()

        # System prompt cache
        self._cached_system_prompt = ""
        self._cached_system_prompt_time = 0.0
        self._system_prompt_ttl = 300.0  # 5 minutes

    def _get_client(self):
        """Lazy-init OpenAI client."""
        if self._client is None:
            if not self.base_url:
                raise ValueError(
                    "LLM base URL not configured. Set via:\n"
                    "  --llm-url=URL              (CLI)\n"
                    "  AUTOCOG_LLM_BASE_URL=URL   (.env or environment)\n"
                    "\n"
                    "Compatible endpoints:\n"
                    "  Anthropic:  https://api.anthropic.com/v1\n"
                    "  OpenAI:     https://api.openai.com/v1\n"
                    "  Ollama:     http://localhost:11434/v1\n"
                    "  vLLM:       http://localhost:8000/v1\n"
                    "  LM Studio:  http://localhost:1234/v1\n"
                    "  Any OpenAI-compatible endpoint"
                )
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "not-set",
                )
            except ImportError:
                logger.warning("openai package not installed, using fallback strategy")
                return None
        return self._client

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Form intention — LLM every N ticks, PersonalityStrategy between.

        Args:
            sensation: Current perception
            reflection: Recent evaluation

        Returns:
            Intention with chosen goal and targets
        """
        self._tick_counter += 1

        if self._tick_counter % self.call_interval != 0:
            return self._fallback.form_intention(sensation, reflection)

        # Determine complexity and select model tier
        tier = self._select_tier(sensation, reflection)

        try:
            return self._llm_form_intention(sensation, reflection, tier)
        except Exception:
            logger.warning("LLM intention failed, using fallback")
            return self._fallback.form_intention(sensation, reflection)

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Convert intention to action. Uses fallback for action mapping."""
        return self._fallback.express(sensation, reflection, intention)

    def _select_tier(self, sensation: Sensation, reflection: Reflection) -> str:
        """Select model tier based on situation complexity.

        Args:
            sensation: Current perception
            reflection: Recent evaluation

        Returns:
            Tier string: "fast" | "standard" | "deep"

        Heuristics:
        - Fast: Routine situations (no agents, low threat, needs stable)
        - Standard: Default tier (agents present, moderate needs)
        - Deep: High stakes (critical needs, combat, multiple agents)
        """
        needs = sensation.own_needs

        # High stakes situations → deep
        critical_needs = sum(
            1 for k, v in needs.items() if k in ("hunger", "thirst", "energy") and v < 15
        )
        if critical_needs >= 2:
            return "deep"

        if reflection.threat_level > 0.6:
            return "deep"

        if len(sensation.visible_agents) > 3:
            return "deep"

        # Routine situations → fast
        if not sensation.visible_agents and reflection.threat_level < 0.2:
            all_needs_ok = all(
                v >= 30 for k, v in needs.items() if k in ("hunger", "thirst", "energy")
            )
            if all_needs_ok:
                return "fast"

        # Default to standard
        return "standard"

    def _llm_form_intention(
        self, sensation: Sensation, reflection: Reflection, tier: str
    ) -> Intention:
        """Call LLM to form an intention.

        Args:
            sensation: Current perception
            reflection: Recent evaluation
            tier: Model tier to use

        Returns:
            Parsed Intention from LLM decision

        Raises:
            LLMParseError: If response parsing fails after all fallbacks
        """
        client = self._get_client()
        if client is None:
            return self._fallback.form_intention(sensation, reflection)

        # Build prompts
        system_prompt = self._get_or_build_system_prompt(sensation.own_traits)

        if self.prompt_version == "v1_minimal":
            user_prompt = self._build_minimal_prompt(sensation, reflection)
        else:  # v2_rich (default)
            user_prompt = self._context_builder.build_full_context(sensation, reflection)

        # Select model
        model = self._model_map.get(tier, self.model)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Token limits by tier
        max_tokens = {"fast": 50, "standard": 80, "deep": 150}.get(tier, 80)

        # Prefill forces JSON start — but some proxies (GitHub Copilot)
        # reject trailing assistant messages. Auto-detect on first call.
        if self._prefill_supported is not False:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                    + [
                        {"role": "assistant", "content": '{"goal": "'},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                self._prefill_supported = True
                text = '{"goal": "' + (response.choices[0].message.content or "")
                return self._parse_to_intention(text, sensation)
            except Exception:
                if self._prefill_supported is None:
                    logger.info("Prefill not supported by proxy, disabling")
                    self._prefill_supported = False
                # Fall through to no-prefill path

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        text = response.choices[0].message.content or ""
        return self._parse_to_intention(text, sensation)

    def _get_or_build_system_prompt(self, traits: dict[str, float]) -> str:
        """Get cached system prompt or build new one.

        Args:
            traits: Current agent traits

        Returns:
            System prompt string
        """
        now = time.time()

        # Rebuild if cache expired or empty
        if (
            not self._cached_system_prompt
            or now - self._cached_system_prompt_time > self._system_prompt_ttl
        ):
            self._cached_system_prompt = build_system_prompt(
                archetype=self.archetype,
                traits=traits,
                valid_goals=VALID_GOALS,
            )
            self._cached_system_prompt_time = now
            logger.debug("Built new system prompt (cache miss)")

        return self._cached_system_prompt

    def _build_minimal_prompt(self, sensation: Sensation, reflection: Reflection) -> str:
        """Build minimal prompt (v1 style) for lightweight testing.

        Args:
            sensation: Current perception
            reflection: Recent evaluation

        Returns:
            Basic status string
        """
        lines = []
        lines.append(f"Tick: {sensation.tick}")
        lines.append(f"Position: {sensation.own_position}")
        lines.append(
            f"Needs: hunger={sensation.own_needs.get('hunger', 0):.0f}, "
            f"thirst={sensation.own_needs.get('thirst', 0):.0f}, "
            f"energy={sensation.own_needs.get('energy', 0):.0f}, "
            f"health={sensation.own_needs.get('health', 0):.0f}"
        )
        lines.append(f"Inventory: {sensation.own_inventory or 'empty'}")
        lines.append(f"Personality: {sensation.own_traits}")
        lines.append(f"Visible agents: {len(sensation.visible_agents)}")
        total_resources = sum(qty for t in sensation.visible_tiles for _, qty in t.resources)
        lines.append(f"Nearby resources: {total_resources}")
        lines.append(f"Threat level: {reflection.threat_level:.1f}")
        lines.append(f"Opportunity: {reflection.opportunity_score:.1f}")
        lines.append(f"Need trends: {reflection.need_trends}")
        lines.append(f"Last action succeeded: {reflection.last_action_succeeded}")
        if sensation.incoming_messages:
            lines.append(f"Messages received: {len(sensation.incoming_messages)}")
        return "\n".join(lines)

    def _parse_to_intention(self, response_text: str, sensation: Sensation) -> Intention:
        """Parse LLM response into an Intention.

        Args:
            response_text: Raw LLM completion text
            sensation: Current sensation for target selection

        Returns:
            Intention with validated goal and targets

        Raises:
            LLMParseError: If parsing fails completely (shouldn't happen - parser has fallbacks)
        """
        try:
            decision = self._parser.parse(response_text)
        except Exception as e:
            logger.error(f"Parser raised exception: {e}", exc_info=True)
            raise LLMParseError(response_text, "LLMResponseParser.parse") from e

        # Map goal to internal format and find targets
        goal = decision.goal
        target_agent_id = decision.target_agent
        target_pos = None

        # For goals needing agent targets, find nearest visible agent
        if goal in ("socialize", "attack", "share_resources", "flee"):
            if not target_agent_id and sensation.visible_agents:
                nearest = sensation.visible_agents[0]
                target_agent_id = nearest.agent_id
                target_pos = nearest.position
            elif target_agent_id:
                # Find position of specified agent
                for agent in sensation.visible_agents:
                    if agent.agent_id == target_agent_id:
                        target_pos = agent.position
                        break

        # For flee, find direction away from threat
        if goal == "flee" and target_pos:
            # Invert direction to flee away
            dx = sensation.own_position[0] - target_pos[0]
            dy = sensation.own_position[1] - target_pos[1]
            # Amplify to move away
            flee_x = sensation.own_position[0] + (2 * (1 if dx > 0 else -1) if dx != 0 else 0)
            flee_y = sensation.own_position[1] + (2 * (1 if dy > 0 else -1) if dy != 0 else 0)
            target_pos = (flee_x, flee_y)

        logger.info(
            f"LLM decision: {goal} (confidence {decision.confidence:.2f}) - {decision.reason}"
        )

        return Intention(
            primary_goal=goal,
            target_position=target_pos,
            target_agent_id=target_agent_id,
            planned_actions=[goal],
            confidence=decision.confidence,
        )
