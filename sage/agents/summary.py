"""Summary Agent (Spec 10A).

Reads results_interpreter + hypothesis_expansion + novelty_debate.
Does NOT read coding (via Results Interpreter indirectly) or explainability
(merged by pipeline orchestrator after Summary runs).
"""

from __future__ import annotations

from typing import Any

from sage.agents.base import BaseAgent
from sage.llm import call_llm


class SummaryAgent(BaseAgent):
    name = "summary"

    def _build_user_prompt(
        self,
        hyp: dict[str, Any],
        interpretation: dict[str, Any],
        debate_results: list[dict[str, Any]],
    ) -> str:
        # Find matching debate result
        hyp_id = hyp.get("hypothesis_id", "")
        debate = next(
            (d for d in debate_results if d.get("hypothesis_id") == hyp_id),
            debate_results[0] if debate_results else {},
        )

        return (
            f"Hypothesis:\n"
            f"  ID: {hyp.get('hypothesis_id', '')}\n"
            f"  Statement: {hyp.get('hypothesis_statement', '')}\n"
            f"  Population: {hyp.get('population', '')}\n"
            f"  Variables: {hyp.get('variables', {})}\n"
            f"  Outcome: {hyp.get('outcome', '')}\n"
            f"  Expected directionality: {hyp.get('expected_directionality', '')}\n"
            f"  Mechanistic rationale: {hyp.get('mechanistic_rationale', '')}\n"
            f"  Clinical significance: {hyp.get('clinical_significance', '')}\n\n"
            f"Novelty Debate:\n"
            f"  Score: {debate.get('novelty_score', 'N/A')}\n"
            f"  Debate triggered: {debate.get('debate_triggered', False)}\n"
            f"  Rounds: {debate.get('total_rounds', 0)}\n"
            f"  Prover: {debate.get('final_prover_argument', '')}\n"
            f"  Verifier: {debate.get('final_verifier_argument', '')}\n"
            f"  Judge: {debate.get('final_judge_synthesis', '')}\n\n"
            f"Validation Results:\n"
            f"  Supported: {interpretation.get('hypothesis_supported', False)}\n"
            f"  Evidence: {interpretation.get('statistical_evidence', '')}\n"
            f"  Effect sizes: {interpretation.get('effect_sizes', '')}\n"
            f"  P-values: {interpretation.get('p_values', '')}\n"
            f"  Feasibility score: {interpretation.get('feasibility_score', 0)}\n"
            f"  Limitations: {interpretation.get('limitations', '')}\n"
            f"  Conclusion: {interpretation.get('conclusion', '')}\n"
        )

    def _fallback(
        self,
        hyp: dict[str, Any],
        interpretation: dict[str, Any],
        debate_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        hyp_id = hyp.get("hypothesis_id", "")
        debate = next(
            (d for d in debate_results if d.get("hypothesis_id") == hyp_id),
            debate_results[0] if debate_results else {},
        )

        return {
            "report": {
                "hypothesis_statement": str(hyp.get("hypothesis_statement", "")),
                "scientific_rationale": str(hyp.get("mechanistic_rationale", "")),
                "novelty_assessment": {
                    "score": float(debate.get("novelty_score", 0)),
                    "debate_summary": (
                        f"Debate {'triggered' if debate.get('debate_triggered') else 'not triggered'} "
                        f"({debate.get('total_rounds', 0)} round(s)). "
                        f"Prover: {debate.get('final_prover_argument', 'N/A')[:200]}. "
                        f"Verifier: {debate.get('final_verifier_argument', 'N/A')[:200]}. "
                        f"Judge: {debate.get('final_judge_synthesis', 'N/A')[:200]}."
                    ),
                },
                "validation_results": {
                    "hypothesis_supported": bool(interpretation.get("hypothesis_supported", False)),
                    "statistical_evidence": str(interpretation.get("statistical_evidence", "")),
                    "feasibility_score": float(interpretation.get("feasibility_score", 0)),
                },
                "limitations": str(interpretation.get("limitations", "")),
                "next_steps": (
                    "Validate on independent clinical cohorts with larger sample sizes. "
                    "Consider multi-center studies for external validation."
                ),
                "conclusion": str(interpretation.get("conclusion", "")),
            }
        }

    def _normalize(
        self,
        raw: Any,
        hyp: dict[str, Any],
        interpretation: dict[str, Any],
        debate_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._fallback(hyp, interpretation, debate_results)

        report = raw.get("report")
        if not isinstance(report, dict):
            return self._fallback(hyp, interpretation, debate_results)

        novelty = report.get("novelty_assessment", {})
        if not isinstance(novelty, dict):
            novelty = {}

        validation = report.get("validation_results", {})
        if not isinstance(validation, dict):
            validation = {}

        try:
            novelty_score = float(novelty.get("score", 0))
        except (TypeError, ValueError):
            novelty_score = 0.0

        try:
            feas_score = float(validation.get("feasibility_score", 0))
        except (TypeError, ValueError):
            feas_score = 0.0

        return {
            "report": {
                "hypothesis_statement": str(
                    report.get("hypothesis_statement", hyp.get("hypothesis_statement", ""))
                ),
                "scientific_rationale": str(
                    report.get("scientific_rationale", hyp.get("mechanistic_rationale", ""))
                ),
                "novelty_assessment": {
                    "score": novelty_score,
                    "debate_summary": str(novelty.get("debate_summary", "")),
                },
                "validation_results": {
                    "hypothesis_supported": bool(validation.get("hypothesis_supported", False)),
                    "statistical_evidence": str(validation.get("statistical_evidence", "")),
                    "feasibility_score": max(0.0, min(10.0, feas_score)),
                },
                "limitations": str(report.get("limitations", "")),
                "next_steps": str(report.get("next_steps", "")),
                "conclusion": str(report.get("conclusion", "")),
            }
        }

    def run(self) -> dict[str, Any]:
        context = self.read_context()

        hypothesis_data = context.get("hypothesis_expansion")
        interp_data = context.get("results_interpreter")
        debate_data = context.get("novelty_debate")

        if not isinstance(hypothesis_data, dict):
            raise ValueError("SummaryAgent requires 'hypothesis_expansion' in context.")
        if not isinstance(interp_data, dict):
            raise ValueError("SummaryAgent requires 'results_interpreter' in context.")

        expanded = hypothesis_data.get("expanded_hypotheses", [])
        if not expanded:
            raise ValueError("SummaryAgent requires non-empty expanded_hypotheses.")
        hyp = expanded[0]

        interpretation = interp_data.get("interpretation", {})
        debate_results = (debate_data or {}).get("debate_results", [])

        system_prompt = (
            "You are a research report writer. Synthesize all inputs into a structured "
            "research report. Return strict JSON only in this schema: "
            "{\"report\": {\"hypothesis_statement\": str, \"scientific_rationale\": str, "
            "\"novelty_assessment\": {\"score\": float, \"debate_summary\": str}, "
            "\"validation_results\": {\"hypothesis_supported\": bool, "
            "\"statistical_evidence\": str, \"feasibility_score\": float}, "
            "\"limitations\": str, \"next_steps\": str, \"conclusion\": str}}"
        )
        user_prompt = self._build_user_prompt(hyp, interpretation, debate_results)

        try:
            raw = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, response_format="json")
            output = self._normalize(raw, hyp, interpretation, debate_results)
        except Exception:
            output = self._fallback(hyp, interpretation, debate_results)

        self.write_output(output)
        return output
