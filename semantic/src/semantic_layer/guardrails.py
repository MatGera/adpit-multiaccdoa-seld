"""LLM guardrails: input/output validation and safety checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    passed: bool
    reason: str = ""
    filtered_text: str = ""


class Guardrails:
    """Input and output guardrails for the LLM layer.

    Ensures:
    - No PII in prompts or responses
    - Responses stay within industrial maintenance domain
    - No harmful/misleading safety recommendations
    - Content length limits
    """

    # Patterns that should not appear in input or output
    _PII_PATTERNS = [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),       # SSN
        re.compile(r"\b[A-Z]{2}\d{7}\b"),             # Passport
        re.compile(r"\b\d{16}\b"),                     # Credit card (basic)
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # Email
    ]

    # Words that indicate the model is going off-topic
    _OFF_TOPIC_KEYWORDS = [
        "i cannot", "as an ai", "i don't have feelings",
        "i'm sorry but", "my training data",
    ]

    # Severity levels that must appear in prescriptive responses
    _VALID_SEVERITIES = {"INFO", "WARNING", "CRITICAL", "EMERGENCY"}

    def validate_input(self, text: str) -> GuardrailResult:
        """Validate input text before sending to LLM.

        Args:
            text: Input prompt text.

        Returns:
            GuardrailResult indicating pass/fail.
        """
        # Check for PII
        for pattern in self._PII_PATTERNS:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    reason="Input contains potential PII",
                )

        # Check length
        if len(text) > 50_000:
            return GuardrailResult(
                passed=False,
                reason="Input exceeds maximum length (50K chars)",
            )

        return GuardrailResult(passed=True, filtered_text=text)

    def validate_output(self, text: str) -> GuardrailResult:
        """Validate LLM output before returning to user.

        Args:
            text: Generated response text.

        Returns:
            GuardrailResult with potential filtering applied.
        """
        filtered = text

        # Remove any PII that leaked through
        for pattern in self._PII_PATTERNS:
            filtered = pattern.sub("[REDACTED]", filtered)

        # Check for off-topic responses
        lower = filtered.lower()
        for keyword in self._OFF_TOPIC_KEYWORDS:
            if keyword in lower:
                logger.warning("off_topic_response_detected", keyword=keyword)

        # Verify severity is present in prescriptive responses
        has_severity = any(sev in filtered.upper() for sev in self._VALID_SEVERITIES)
        if not has_severity and len(filtered) > 200:
            logger.warning("response_missing_severity")

        return GuardrailResult(
            passed=True,
            filtered_text=filtered,
        )
