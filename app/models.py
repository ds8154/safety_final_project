from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CritiqueRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participating_modules: list[str]
    agreement_points: list[str]
    disagreement_points: list[str]
    arbitration_notes: list[str]
    reconciled_risk_score: int = Field(ge=0, le=100)
    reconciled_risk_tier: Literal["Low", "Medium", "High", "Critical"]
    recommended_action: str


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    reference: str
    description: str


class PolicyAlignmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    status: str
    note: str


class DetectedRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_name: str
    severity: Literal["Low", "Medium", "High", "Critical"]
    description: str
    evidence_reference: str
    mitigation: str


class ExpertJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str
    module_name: str
    module_version: str
    assessment_timestamp: str
    perspective_type: str
    overall_risk_score: int = Field(ge=0, le=100)
    risk_tier: Literal["Low", "Medium", "High", "Critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_findings: list[str]
    reasoning_summary: str
    evidence: list[EvidenceItem]
    policy_alignment: list[PolicyAlignmentItem]
    detected_risks: list[DetectedRisk]
    recommended_action: str
    raw_output_reference: str
    error_flag: bool
    error_message: str
