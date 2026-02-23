"""Unit tests for LLM guardrails."""

from __future__ import annotations

import pytest

from semantic_layer.guardrails import Guardrails


class TestGuardrails:
    def setup_method(self):
        self.guard = Guardrails()

    def test_valid_input(self):
        result = self.guard.validate_input("What maintenance is needed for motor M-101?")
        assert result.passed

    def test_pii_ssn_detected(self):
        result = self.guard.validate_input("User 123-45-6789 reported an issue")
        assert not result.passed
        assert "PII" in result.reason

    def test_pii_email_detected(self):
        result = self.guard.validate_input("Contact john@example.com for details")
        assert not result.passed

    def test_input_too_long(self):
        result = self.guard.validate_input("x" * 60_000)
        assert not result.passed
        assert "length" in result.reason

    def test_output_redacts_pii(self):
        result = self.guard.validate_output(
            "Contact the technician at 123-45-6789 for assistance."
        )
        assert result.passed
        assert "[REDACTED]" in result.filtered_text

    def test_clean_output_passes(self):
        result = self.guard.validate_output(
            "CRITICAL: Replace bearing on motor M-101 within 24 hours."
        )
        assert result.passed
        assert result.filtered_text == "CRITICAL: Replace bearing on motor M-101 within 24 hours."
