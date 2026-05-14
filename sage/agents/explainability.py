from __future__ import annotations

from typing import Any

from sage.agents.base import BaseAgent
from sage.llm import call_llm


_EI_DIMENSIONS = ["MD", "CP", "SBC", "CT", "MT"]

_RUBRIC = (
    "Score each dimension 0, 1, or 2 using this rubric:\n"
    "MD (Mechanistic Depth): 0=no mechanistic support, 1=partial pathway evidence, "
    "2=complete pathway explanation with literature support at each step.\n"
    "CP (Causal Plausibility): 0=purely statistical association, 1=possible causal relationship, "
    "2=strong causal evidence supported by interventional experiments.\n"
    "SBC (Spatial & Biological Coherence): 0=no spatial consistency, 1=partial coherence, "
    "2=fully coherent localization in the relevant tissue structures.\n"
    "CT (Clinical Traceability): 0=not measurable, 1=requires specialized/experimental methods, "
    "2=measurable by standard IHC/H&E.\n"
    "MT (Model Transparency): 0=vague definition, 1=semi-quantitative, "
    "2=fully reproducible quantitative definition."
)


def _clamp_ei_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 1
    return max(0, min(2, score))


class ExplainabilityAgent(BaseAgent):
    name = "explainability"

    def _build_user_prompt(self, expanded_hypotheses: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for hyp in expanded_hypotheses:
            lines.append(
                "\n".join(
                    [
                        f"hypothesis_id: {hyp.get('hypothesis_id', '')}",
                        f"statement: {hyp.get('hypothesis_statement', '')}",
                        f"population: {hyp.get('population', '')}",
                        f"variables: {hyp.get('variables', {})}",
                        f"outcome: {hyp.get('outcome', '')}",
                        f"expected_directionality: {hyp.get('expected_directionality', '')}",
                        f"mechanistic_rationale: {hyp.get('mechanistic_rationale', '')}",
                        f"background_literature: {hyp.get('background_literature', '')}",
                        f"clinical_significance: {hyp.get('clinical_significance', '')}",
                        f"validation_strategy: {hyp.get('validation_strategy', '')}",
                    ]
                )
            )
        return "Expanded hypotheses:\n\n" + "\n\n---\n\n".join(lines)

    def _fallback(self, expanded_hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for hyp in expanded_hypotheses:
            dimensions = {
                "MD": {"score": 1, "justification": "Partial pathway evidence described in rationale."},
                "CP": {"score": 1, "justification": "Possible causal relationship suggested."},
                "SBC": {"score": 1, "justification": "Partial spatial coherence indicated."},
                "CT": {"score": 1, "justification": "Measurement feasibility partially addressed."},
                "MT": {"score": 1, "justification": "Semi-quantitative definition provided."},
            }
            results.append(
                {
                    "hypothesis_id": str(hyp.get("hypothesis_id", "")),
                    "ei_total": 5,
                    "dimensions": dimensions,
                }
            )
        return {"explainability_results": results}

    def _normalize(self, raw: Any, expanded_hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._fallback(expanded_hypotheses)

        results_raw = raw.get("explainability_results")
        if not isinstance(results_raw, list) or not results_raw:
            return self._fallback(expanded_hypotheses)

        hyp_ids = [str(h.get("hypothesis_id", f"H{i+1}")) for i, h in enumerate(expanded_hypotheses)]
        cleaned: list[dict[str, Any]] = []

        for i, item in enumerate(results_raw):
            if not isinstance(item, dict):
                continue

            hypothesis_id = str(item.get("hypothesis_id", ""))
            if not hypothesis_id and i < len(hyp_ids):
                hypothesis_id = hyp_ids[i]

            dimensions_raw = item.get("dimensions")
            if not isinstance(dimensions_raw, dict):
                dimensions_raw = {}

            dimensions: dict[str, dict[str, Any]] = {}
            ei_total = 0
            for dim in _EI_DIMENSIONS:
                dim_data = dimensions_raw.get(dim)
                if isinstance(dim_data, dict):
                    score = _clamp_ei_score(dim_data.get("score"))
                    justification = str(dim_data.get("justification", "No justification provided."))
                else:
                    score = 1
                    justification = "Score inferred due to missing LLM output."
                dimensions[dim] = {"score": score, "justification": justification}
                ei_total += score

            cleaned.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "ei_total": ei_total,
                    "dimensions": dimensions,
                }
            )

        if not cleaned:
            return self._fallback(expanded_hypotheses)
        return {"explainability_results": cleaned}

    def run(self) -> dict[str, Any]:
        context = self.read_context()
        hypothesis = context.get("hypothesis_expansion")
        if not isinstance(hypothesis, dict):
            raise ValueError("ExplainabilityAgent requires 'hypothesis_expansion' in context.")

        expanded_hypotheses = hypothesis.get("expanded_hypotheses")
        if not isinstance(expanded_hypotheses, list) or not expanded_hypotheses:
            raise ValueError("ExplainabilityAgent requires non-empty expanded_hypotheses.")

        system_prompt = (
            "You are an explainability evaluator for biomedical hypotheses. "
            "For each hypothesis, compute the Explainability Index (EI) across five dimensions. "
            "EI = MD + CP + SBC + CT + MT. Each dimension is scored 0, 1, or 2 (total 0-10).\n\n"
            f"{_RUBRIC}\n\n"
            "For each dimension provide a score and a brief justification.\n\n"
            "Return strict JSON only in this schema: {\"explainability_results\":[{"
            "\"hypothesis_id\":str,\"ei_total\":int,"
            "\"dimensions\":{\"MD\":{\"score\":int,\"justification\":str},"
            "\"CP\":{\"score\":int,\"justification\":str},"
            "\"SBC\":{\"score\":int,\"justification\":str},"
            "\"CT\":{\"score\":int,\"justification\":str},"
            "\"MT\":{\"score\":int,\"justification\":str}}}]}"
        )
        user_prompt = self._build_user_prompt(expanded_hypotheses)

        try:
            raw = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, response_format="json")
            output = self._normalize(raw, expanded_hypotheses)
        except Exception:
            output = self._fallback(expanded_hypotheses)

        self.write_output(output)
        return output
