"""End-to-end integration test for the POST /submit endpoint.

Uses FastAPI TestClient with MOCK_MODE=1 so no live Ollama or Gemini API key
is required. The test verifies that a well-formed JSON submission travels the
full pipeline (three judges → critique round → synthesis) and returns a
structurally valid response with all expected fields.
"""
from __future__ import annotations

import os
import sys

# Ensure MOCK_MODE is active before any app modules are imported so that
# is_mock_mode() returns True throughout the entire test run.
os.environ["MOCK_MODE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "submission_id": "test_integration_001",
    "submitted_by": "test_runner",
    "agent_name": "VeriMedia",
    "agent_description": (
        "VeriMedia is a Flask-based media verification agent that uses GPT-4o as its backend LLM "
        "and OpenAI Whisper for audio/video transcription. It accepts unauthenticated file uploads "
        "via a public endpoint and processes multimedia content to detect disinformation. "
        "GitHub: https://github.com/FlashCarrot/VeriMedia"
    ),
    "use_case": "Automated disinformation detection for multimedia content.",
    "deployment_context": "Public-facing web application, accessible without authentication.",
    "selected_frameworks": ["EU AI Act", "US NIST AI RMF"],
    "risk_focus": ["Prompt Injection", "PII Leakage", "Evasion"],
    "submitted_evidence": [],
    "notes": "Integration test submission using MOCK_MODE=1.",
}


def test_submit_returns_200():
    """POST /submit should return HTTP 200 for a valid JSON body."""
    response = client.post("/submit", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"


def test_submit_response_has_required_top_level_keys():
    """Response body must contain message, submission, results, and artifacts."""
    response = client.post("/submit", json=SAMPLE_PAYLOAD)
    body = response.json()
    for key in ("message", "submission", "results", "artifacts"):
        assert key in body, f"Missing top-level key: '{key}'"


def test_submit_results_contain_three_judge_outputs():
    """Pipeline must return exactly three judge outputs."""
    response = client.post("/submit", json=SAMPLE_PAYLOAD)
    judge_outputs = response.json()["results"]["judge_outputs"]
    assert isinstance(judge_outputs, list), "judge_outputs should be a list"
    assert len(judge_outputs) == 3, f"Expected 3 judge outputs, got {len(judge_outputs)}"


def test_submit_judge_outputs_have_no_error_flag():
    """In mock mode all three judges should succeed (error_flag=False)."""
    response = client.post("/submit", json=SAMPLE_PAYLOAD)
    for idx, judge in enumerate(response.json()["results"]["judge_outputs"]):
        assert judge["error_flag"] is False, f"Judge {idx + 1} returned error_flag=True: {judge.get('error_message')}"


def test_submit_synthesis_has_verdict_field():
    """Synthesis output must include the APPROVE/REVIEW/REJECT verdict field."""
    response = client.post("/submit", json=SAMPLE_PAYLOAD)
    synthesis = response.json()["results"]["synthesis_output"]
    assert "verdict" in synthesis, "synthesis_output missing 'verdict' field"
    assert synthesis["verdict"] in ("APPROVE", "REVIEW", "REJECT"), (
        f"Unexpected verdict value: {synthesis['verdict']}"
    )


def test_submit_synthesis_rationale_is_readable():
    """Rationale should be a non-empty plain-English string mentioning the agent name."""
    response = client.post("/submit", json=SAMPLE_PAYLOAD)
    rationale = response.json()["results"]["synthesis_output"].get("rationale", "")
    assert isinstance(rationale, str) and len(rationale) > 50, "Rationale is missing or too short"
    assert "VeriMedia" in rationale, "Rationale should reference the agent name 'VeriMedia'"


def test_submit_critique_round_present():
    """Response must include a critique_round section with a reconciled score."""
    response = client.post("/submit", json=SAMPLE_PAYLOAD)
    critique = response.json()["results"]["critique_round"]
    assert "reconciled_risk_score" in critique, "critique_round missing 'reconciled_risk_score'"
    score = critique["reconciled_risk_score"]
    assert 0 <= score <= 100, f"reconciled_risk_score out of range: {score}"
