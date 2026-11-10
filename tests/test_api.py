"""Tests for API validation and liveness."""

import pytest
from pydantic import ValidationError

from app.api import PredictionInput, health


def test_health_endpoint() -> None:
    """The health endpoint responds without requiring a model artifact."""
    assert health() == {"status": "ok"}


def test_prediction_input_rejects_invalid_hours() -> None:
    """Hours per day must remain within the supported survey range."""
    with pytest.raises(ValidationError):
        PredictionInput(**{"Hours per day": 25})
