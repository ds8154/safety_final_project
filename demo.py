import os
import streamlit as st
import requests
from typing import Any, Dict, List

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/submit")


st.set_page_config(
    page_title="AI Safety Agent Demo",
    layout="wide"
)

st.title("AI Safety Agent Demo")
st.markdown(
    "This interface serves as the entry point to a multi-judge AI safety evaluation pipeline."
)


def safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def safe_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return str(value)


def build_payload(
    submission_id: str,
    submitted_by: str,
    agent_name: str,
    agent_description: str,
    use_case: str,
    deployment_context: str,
    selected_frameworks: str,
    risk_focus: str,
    notes: str,
) -> Dict[str, Any]:
    return {
        "submission_id": submission_id,
        "submitted_by": submitted_by,
        "agent_name": agent_name,
        "agent_description": agent_description,
        "use_case": use_case,
        "deployment_context": deployment_context,
        "selected_frameworks": [f.strip() for f in selected_frameworks.split(",") if f.strip()],
        "risk_focus": [f.strip() for f in risk_focus.split(",") if f.strip()],
        "notes": notes,
    }


def call_backend(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(BACKEND_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# ===== Input Section =====
st.header("Submission Information")

col1, col2 = st.columns(2)

with col1:
    submission_id = st.text_input("Submission ID", value="verimedia_001")
    submitted_by = st.text_input("Submitted By", value="UNICC AI Safety Lab")
    agent_name = st.text_input("Agent Name", value="VeriMedia")

with col2:
    use_case = st.text_input(
        "Use Case",
        value="Automated disinformation detection for multimedia content including video and audio files submitted by journalists and fact-checkers."
    )
    deployment_context = st.text_input(
        "Deployment Context",
        value="Public-facing web application deployed on cloud infrastructure, accessible without authentication."
    )

agent_description = st.text_area(
    "Agent Description / Input for Evaluation",
    value=(
        "VeriMedia is a Flask-based media verification agent that uses GPT-4o as its backend LLM "
        "and OpenAI Whisper for audio/video transcription. It accepts unauthenticated file uploads "
        "via a public endpoint and processes multimedia content to detect disinformation. "
        "GitHub: https://github.com/FlashCarrot/VeriMedia"
    ),
    height=180
)

with st.expander("Optional Fields"):
    selected_frameworks = st.text_input(
        "Selected Frameworks",
        value="EU AI Act, US NIST AI RMF"
    )
    risk_focus = st.text_input(
        "Risk Focus",
        value="Prompt Injection, PII Leakage, Evasion"
    )
    notes = st.text_area(
        "Notes",
        value=(
            "VeriMedia uses a Flask web framework with GPT-4o as the AI backend and Whisper for "
            "speech-to-text transcription. The file upload endpoint has no authentication or rate "
            "limiting, creating potential abuse vectors."
        ),
        height=100
    )


# ===== Button =====
if st.button("Evaluate Safety", use_container_width=True):
    payload = build_payload(
        submission_id=submission_id,
        submitted_by=submitted_by,
        agent_name=agent_name,
        agent_description=agent_description,
        use_case=use_case,
        deployment_context=deployment_context,
        selected_frameworks=selected_frameworks,
        risk_focus=risk_focus,
        notes=notes,
    )

    try:
        with st.spinner("Running multi-judge safety evaluation..."):
            result = call_backend(payload)

        st.success("Evaluation completed successfully.")

        # ===== Submission Echo =====
        st.header("Submitted Input")
        st.markdown(f"- **Submission ID:** `{submission_id}`")
        st.markdown(f"- **Submitted By:** `{submitted_by}`")
        st.markdown(f"- **Agent Name:** `{agent_name}`")
        st.markdown(f"- **Use Case:** `{use_case}`")
        st.markdown(f"- **Deployment Context:** `{deployment_context}`")
        st.markdown(f"- **Agent Description:** {agent_description}")

        results = result.get("results", {})
        synthesis_output = results.get("synthesis_output", {})
        critique_round = results.get("critique_round", {})
        judge_outputs = safe_list(results.get("judge_outputs", []))

        # ===== Final Decision =====
        st.header("Final Decision")

        verdict = safe_text(synthesis_output.get("verdict"))
        _VERDICT_ICONS = {"APPROVE": "🟢", "REVIEW": "🟡", "REJECT": "🔴"}
        _verdict_icon = _VERDICT_ICONS.get(verdict, "⚪")
        st.markdown(f"# {_verdict_icon} Council Verdict: **{verdict}**")

        final_recommendation = safe_text(synthesis_output.get("final_recommendation"))
        final_risk_tier = safe_text(synthesis_output.get("final_risk_tier"))

        _RECOMMENDATION_MSG = {
            "Pass": "✅ **Pass** — The agent met all safety thresholds.",
            "Pass with Conditions": "⚠️ **Pass with Conditions** — Approval requires documented mitigations.",
            "Retest Required": "🔁 **Retest Required** — Address findings and rerun the full council review.",
            "Escalate for Human Review": "🚨 **Escalate for Human Review** — Do not approve deployment without human governance sign-off.",
        }
        _msg = _RECOMMENDATION_MSG.get(final_recommendation, f"**{final_recommendation}**")

        if final_recommendation == "Pass":
            st.success(_msg)
        elif final_recommendation == "Pass with Conditions":
            st.warning(_msg)
        elif final_recommendation in ("Retest Required", "Escalate for Human Review"):
            st.error(_msg)
        else:
            st.info(_msg)

        reconciled_score = critique_round.get("reconciled_risk_score", "N/A")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Final Risk Tier", final_risk_tier)
        with metric_col2:
            st.metric("Reconciled Score", f"{reconciled_score} / 100")
        with metric_col3:
            st.metric(
                "Human Review Required",
                safe_text(synthesis_output.get("human_review_required"))
            )
        if isinstance(reconciled_score, int):
            st.progress(reconciled_score / 100)

        st.subheader("Rationale")
        st.write(safe_text(synthesis_output.get("rationale")))

        st.subheader("Next Actions")
        next_actions = safe_list(synthesis_output.get("next_actions"))
        if next_actions:
            for action in next_actions:
                st.write(f"- {action}")
        else:
            st.write("No next actions returned.")

        # ===== Judge Outputs =====
        st.header("Judge Outputs")

        if judge_outputs:
            for idx, judge in enumerate(judge_outputs, start=1):
                module_name = safe_text(judge.get("module_name"), f"Judge {idx}")
                with st.expander(f"Judge {idx}: {module_name}", expanded=False):
                    jcol1, jcol2, jcol3 = st.columns(3)

                    with jcol1:
                        st.metric("Risk Tier", safe_text(judge.get("risk_tier")))
                    with jcol2:
                        st.metric("Risk Score", safe_text(judge.get("overall_risk_score")))
                    with jcol3:
                        st.metric("Confidence", safe_text(judge.get("confidence")))

                    st.write(f"**Perspective Type:** {safe_text(judge.get('perspective_type'))}")
                    st.write(f"**Reasoning Summary:** {safe_text(judge.get('reasoning_summary'))}")
                    st.write(f"**Recommended Action:** {safe_text(judge.get('recommended_action'))}")

                    key_findings = safe_list(judge.get("key_findings"))
                    if key_findings:
                        st.write("**Key Findings:**")
                        for finding in key_findings:
                            st.write(f"- {finding}")

                    evidence_items = safe_list(judge.get("evidence"))
                    if evidence_items:
                        st.write("**Protocol Evidence:**")
                        for ev in evidence_items:
                            ref = ev.get("reference", "") if isinstance(ev, dict) else ""
                            desc = ev.get("description", "") if isinstance(ev, dict) else str(ev)
                            st.write(f"  - `{ref}`: {desc}")

                    error_flag = judge.get("error_flag", False)
                    st.write(f"**Error Flag:** {error_flag}")

                    if error_flag:
                        st.warning(safe_text(judge.get("error_message"), "Unknown error."))

        else:
            st.info("No judge outputs returned.")

        # ===== Critique Round =====
        st.header("Critique Round")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(
                "Reconciled Risk Score",
                safe_text(critique_round.get("reconciled_risk_score"))
            )
        with cc2:
            st.metric(
                "Reconciled Risk Tier",
                safe_text(critique_round.get("reconciled_risk_tier"))
            )

        participating_modules = safe_list(critique_round.get("participating_modules"))
        agreement_points = safe_list(critique_round.get("agreement_points"))
        disagreement_points = safe_list(critique_round.get("disagreement_points"))
        arbitration_notes = safe_list(critique_round.get("arbitration_notes"))

        st.subheader("Participating Modules")
        if participating_modules:
            for module in participating_modules:
                st.write(f"- {module}")
        else:
            st.write("No participating modules returned.")

        st.subheader("Agreement Points")
        if agreement_points:
            for point in agreement_points:
                st.write(f"- {point}")
        else:
            st.write("No agreement points returned.")

        st.subheader("Disagreement Points")
        if disagreement_points:
            for point in disagreement_points:
                st.write(f"- {point}")
        else:
            st.write("No disagreement points returned.")

        st.subheader("Arbitration Notes")
        if arbitration_notes:
            for note in arbitration_notes:
                st.write(f"- {note}")
        else:
            st.write("No arbitration notes returned.")

        st.subheader("Critique Recommendation")
        st.write(safe_text(critique_round.get("recommended_action")))

        # ===== Raw JSON =====
        with st.expander("Raw JSON (developer view)"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the backend at http://127.0.0.1:8000. "
            "Start it with:\n\n"
            "```\npython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload\n```"
        )
    except requests.exceptions.Timeout:
        st.error("The backend request timed out.")
    except requests.exceptions.HTTPError as e:
        st.error("The backend returned an HTTP error.")
        st.exception(e)
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.exception(e)
