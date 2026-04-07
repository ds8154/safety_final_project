from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

try:
    from app.judge1 import run_judge_1
    from app.judge2 import run_judge_2
    from app.judge3 import run_judge_3
    from app.models import EvidenceItem, PolicyAlignmentItem, DetectedRisk, ExpertJudgeOutput
    from app.synthesis import run_synthesis
except ModuleNotFoundError:
    from judge1 import run_judge_1
    from judge2 import run_judge_2
    from judge3 import run_judge_3
    from models import EvidenceItem, PolicyAlignmentItem, DetectedRisk, ExpertJudgeOutput
    from synthesis import run_synthesis

# Re-export so external code that imports these names from orchestrator
# continues to work without changes.
__all__ = [
    "EvidenceItem",
    "PolicyAlignmentItem",
    "DetectedRisk",
    "ExpertJudgeOutput",
    "run_pipeline",
]


class SubmittedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    file_type: str = ""
    file_path: str = ""
    description: str = ""


class SubmissionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str
    submitted_by: str
    submission_timestamp: str = ""
    agent_name: str
    agent_description: str
    use_case: str
    deployment_context: str
    selected_frameworks: list[str] = Field(default_factory=list)
    risk_focus: list[str] = Field(default_factory=list)
    submitted_evidence: list[SubmittedEvidence] = Field(default_factory=list)
    notes: str = ""


class CritiqueRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participating_modules: list[str]
    agreement_points: list[str]
    disagreement_points: list[str]
    arbitration_notes: list[str]
    reconciled_risk_score: int = Field(ge=0, le=100)
    reconciled_risk_tier: Literal["Low", "Medium", "High", "Critical"]
    recommended_action: str


TIER_ORDER = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}


def _keywords_for_findings(findings: list[str], risk_focus: list[str] | None = None) -> set[str]:
    joined = " ".join(findings).lower()
    topics: set[str] = set()
    keyword_map = {
        "bias":           ("bias", "fair", "discrimin", "demographic"),
        "privacy":        ("privacy", "pii", "data", "sensitive", "leak", "inference attack", "memoriz"),
        "transparency":   ("transparency", "document", "explain", "disclosure", "model card"),
        "misuse":         ("jailbreak", "misuse", "harm", "attack", "adversarial", "abuse", "exploit"),
        "compliance":     ("compliance", "legal", "regulat", "governance", "policy", "gdpr", "eu ai act", "nist"),
        "security":       ("prompt injection", "evasion", "backdoor", "poison", "security", "breach",
                           "unauthenticated", "rate limit", "upload", "supply chain"),
        "operations":     ("oversight", "monitor", "audit", "owner", "deployment", "operational",
                           "transcription", "ffmpeg", "whisper", "fine-tun"),
        "disinformation": ("disinformation", "misinformation", "deepfake", "media verif", "fact-check"),
    }
    for topic, patterns in keyword_map.items():
        if any(pattern in joined for pattern in patterns):
            topics.add(topic)
    for focus_term in (risk_focus or []):
        topics.add(focus_term.lower().strip())
    return topics


def _enrich_evidence(normalized_input: dict[str, Any]) -> dict[str, Any]:
    enriched = normalized_input.copy()
    snippets: list[str] = []
    for item in normalized_input.get("submitted_evidence", []):
        desc = item.get("description", "")
        ref = item.get("reference", item.get("file_name", ""))
        if "github.com" in desc or "github.com" in ref:
            url = next((w for w in (desc + " " + ref).split() if "github.com" in w), None)
            if url:
                try:
                    r = httpx.get(url, timeout=10, follow_redirects=True)
                    snippets.append(f"[Fetched {url}]\n{r.text[:3000]}")
                except Exception:
                    snippets.append(f"[Could not fetch {url}]")
    if snippets:
        enriched["notes"] = (normalized_input.get("notes", "") + "\n\n" + "\n\n".join(snippets)).strip()
    return enriched


def _risk_tier_from_score(score: int) -> Literal["Low", "Medium", "High", "Critical"]:
    if score >= 78:
        return "Critical"
    if score >= 58:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _highest_tier(tiers: list[str]) -> Literal["Low", "Medium", "High", "Critical"]:
    highest = "Low"
    for tier in tiers:
        if TIER_ORDER[tier] > TIER_ORDER[highest]:
            highest = tier
    return highest  # type: ignore[return-value]


def _majority_tier(results: list[ExpertJudgeOutput]) -> str | None:
    counts = Counter(result.risk_tier for result in results)
    tier, count = counts.most_common(1)[0]
    if count >= 2:
        return tier
    return None


def _confidence_weight(result: ExpertJudgeOutput) -> float:
    weight = max(result.confidence, 0.35)
    if result.error_flag:
        weight *= 0.5
    return weight


def _reconciled_score(results: list[ExpertJudgeOutput]) -> int:
    scores = [result.overall_risk_score for result in results]
    weighted_average = sum(result.overall_risk_score * _confidence_weight(result) for result in results) / sum(
        _confidence_weight(result) for result in results
    )
    median_score = float(median(scores))
    max_score = float(max(scores))
    score_spread = max(scores) - min(scores)

    if score_spread >= 20:
        return int(round(weighted_average * 0.45 + median_score * 0.25 + max_score * 0.30))
    if len({result.risk_tier for result in results}) > 1:
        return int(round(weighted_average * 0.65 + median_score * 0.35))
    return int(round(weighted_average))


def _critique_judges(judge_outputs: list[dict[str, Any]], risk_focus: list[str] | None = None) -> CritiqueRound:
    validated_results = TypeAdapter(list[ExpertJudgeOutput]).validate_python(judge_outputs)
    agreement_points: list[str] = []
    disagreement_points: list[str] = []
    arbitration_notes: list[str] = []

    participating_modules = [result.module_name for result in validated_results]
    tiers = [result.risk_tier for result in validated_results]
    tier_counts = Counter(tiers)
    score_spread = max(result.overall_risk_score for result in validated_results) - min(
        result.overall_risk_score for result in validated_results
    )

    majority_tier = _majority_tier(validated_results)
    if len(tier_counts) == 1:
        agreement_points.append(f"All three judges assign a {tiers[0]} overall risk tier.")
    elif majority_tier:
        agreement_points.append(
            f"Two of three judges converge on a {majority_tier} overall risk tier."
        )
        disagreement_points.append(
            "At least one judge assigns a different severity tier, so the council is not in full consensus."
        )
    else:
        disagreement_points.append(
            "All three judges split across different risk tiers, indicating a materially contested assessment."
        )

    topic_counter: Counter[str] = Counter()
    for result in validated_results:
        topic_counter.update(_keywords_for_findings(result.key_findings, risk_focus))

    shared_topics = sorted(topic for topic, count in topic_counter.items() if count >= 2)
    if shared_topics:
        agreement_points.append(
            "At least two judges flag overlapping topics: " + ", ".join(shared_topics) + "."
        )
    else:
        disagreement_points.append(
            "The judges emphasize different risk clusters across technical, governance, and operational perspectives."
        )

    if score_spread >= 20:
        disagreement_points.append(
            f"The council differs materially on severity by {score_spread} risk-score points."
        )

    if any(result.error_flag for result in validated_results):
        disagreement_points.append(
            "One or more judge modules returned fallback outputs because their primary evaluation path failed."
        )

    arbitration_notes.append(
        "The critique round uses a confidence-weighted reconciliation across all participating judges rather than a fixed Judge 1 vs Judge 2 comparison."
    )
    if score_spread >= 20:
        arbitration_notes.append(
            "Because score spread is material, the reconciled score is biased upward toward the highest-risk judgment to avoid understating unresolved concerns."
        )
    else:
        arbitration_notes.append(
            "Because the score spread is limited, the reconciled score stays close to the confidence-weighted council average."
        )
    if any(result.error_flag for result in validated_results):
        arbitration_notes.append(
            "Fallback or low-confidence outputs were downweighted, but their concerns were still retained in the disagreement log."
        )

    reconciled_risk_score = _reconciled_score(validated_results)
    reconciled_risk_tier = _risk_tier_from_score(reconciled_risk_score)

    if majority_tier and TIER_ORDER[majority_tier] > TIER_ORDER[reconciled_risk_tier]:
        reconciled_risk_tier = majority_tier  # type: ignore[assignment]

    highest_tier = _highest_tier(tiers)
    if highest_tier == "Critical" and TIER_ORDER[reconciled_risk_tier] < TIER_ORDER["High"]:
        reconciled_risk_tier = "High"

    if any(result.error_flag for result in validated_results) or reconciled_risk_tier == "Critical":
        recommended_action = "Escalate to human review after reconciling the combined technical, governance, and operational concerns."
    elif reconciled_risk_tier == "High":
        recommended_action = "Retest after addressing the combined technical, governance, and operational concerns."
    elif disagreement_points:
        recommended_action = "Proceed only with documented conditions and explicit review of the remaining cross-judge disagreements."
    else:
        recommended_action = "Use the shared findings as the primary remediation backlog before the next review."

    return CritiqueRound(
        participating_modules=participating_modules,
        agreement_points=agreement_points or ["All judges found at least some review-worthy concerns."],
        disagreement_points=disagreement_points,
        arbitration_notes=arbitration_notes,
        reconciled_risk_score=reconciled_risk_score,
        reconciled_risk_tier=reconciled_risk_tier,
        recommended_action=recommended_action,
    )


def run_pipeline(input_data: dict[str, Any]) -> dict[str, Any]:
    validated_input = SubmissionInput.model_validate(input_data)
    normalized_input = _enrich_evidence(validated_input.model_dump())

    raw_outputs = [
        run_judge_1(normalized_input),
        run_judge_2(normalized_input),
        run_judge_3(normalized_input),
    ]
    validated_outputs = TypeAdapter(list[ExpertJudgeOutput]).validate_python(raw_outputs)

    critique_round = _critique_judges(
        [result.model_dump() for result in validated_outputs],
        risk_focus=normalized_input.get("risk_focus"),
    )
    synthesis_output = run_synthesis(
        [result.model_dump() for result in validated_outputs],
        critique_round.model_dump(),
        agent_name=normalized_input.get("agent_name", ""),
    )

    return {
        "judge_outputs": [result.model_dump() for result in validated_outputs],
        "critique_round": critique_round.model_dump(),
        "synthesis_output": synthesis_output,
    }
