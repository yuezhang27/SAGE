"""Results Interpreter Agent (Spec 9B).

Single LLM call to interpret Coding Agent output and produce structured evaluation.
"""

from __future__ import annotations

from typing import Any

from sage import config
from sage.agents.base import BaseAgent
from sage.llm import call_llm


class ResultsInterpreterAgent(BaseAgent):
    name = "results_interpreter"

    def _build_user_prompt(
        self,
        hyp: dict[str, Any],
        coding_result: dict[str, Any],
        selected_files: list[str],
    ) -> str:
        return (
            f"Hypothesis:\n"
            f"  ID: {hyp.get('hypothesis_id', '')}\n"
            f"  Statement: {hyp.get('hypothesis_statement', '')}\n"
            f"  Population: {hyp.get('population', '')}\n"
            f"  Variables: {hyp.get('variables', {})}\n"
            f"  Outcome: {hyp.get('outcome', '')}\n"
            f"  Expected directionality: {hyp.get('expected_directionality', '')}\n\n"
            f"Statistical results (stdout):\n{coding_result.get('stdout', 'No output')}\n\n"
            f"Execution success: {coding_result.get('execution_success', False)}\n"
            f"Repair attempts: {coding_result.get('repair_attempts', 0)}\n"
            f"Selected files: {selected_files}\n"
        )

    def _fallback(self, hyp: dict[str, Any], coding_result: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback when LLM is unavailable."""
        exec_success = coding_result.get("execution_success", False)
        stdout = coding_result.get("stdout", "")

        # Try to infer basic results from stdout
        supported = False
        p_values = "Not available"
        effect_sizes = "Not available"
        evidence = "No statistical output to interpret."

        if exec_success and stdout.strip():
            evidence = f"Coding agent produced output: {stdout[:500]}"
            # Simple heuristic: look for p-values < 0.05 in the output
            import json as _json
            try:
                parsed = _json.loads(stdout)
                tests = parsed.get("tests", [])
                sig_tests = [t for t in tests if t.get("significant", False)]
                supported = len(sig_tests) > 0
                p_vals = [f"{t.get('name', '?')}: p={t.get('p_value', '?')}" for t in tests]
                p_values = "; ".join(p_vals) if p_vals else "Not available"
                effects = []
                for t in tests:
                    if "high_mean" in t and "low_mean" in t:
                        effects.append(f"{t['name']}: high={t['high_mean']}, low={t['low_mean']}")
                    elif "spearman_r" in t:
                        effects.append(f"{t['name']}: r={t['spearman_r']}")
                effect_sizes = "; ".join(effects) if effects else "Not available"
            except Exception:
                evidence = f"Raw output (non-JSON): {stdout[:500]}"

        feasibility = 7.0 if (exec_success and supported) else 4.0 if exec_success else 2.0

        return {
            "interpretation": {
                "hypothesis_id": str(hyp.get("hypothesis_id", "")),
                "hypothesis_supported": supported,
                "statistical_evidence": evidence,
                "effect_sizes": effect_sizes,
                "p_values": p_values,
                "feasibility_score": feasibility,
                "limitations": (
                    "Analysis performed on synthetic mock data with 100 patients. "
                    "Results require validation on real clinical cohorts."
                ),
                "conclusion": (
                    f"Hypothesis {'supported' if supported else 'not supported'} "
                    f"based on mock data analysis."
                ),
            }
        }

    def _normalize(self, raw: Any, hyp: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._fallback(hyp, {})

        interp = raw.get("interpretation")
        if not isinstance(interp, dict):
            return self._fallback(hyp, {})

        # Clamp feasibility_score
        try:
            feas = float(interp.get("feasibility_score", 5.0))
        except (TypeError, ValueError):
            feas = 5.0
        feas = max(0.0, min(10.0, feas))

        return {
            "interpretation": {
                "hypothesis_id": str(interp.get("hypothesis_id", hyp.get("hypothesis_id", ""))),
                "hypothesis_supported": bool(interp.get("hypothesis_supported", False)),
                "statistical_evidence": str(interp.get("statistical_evidence", "")),
                "effect_sizes": str(interp.get("effect_sizes", "")),
                "p_values": str(interp.get("p_values", "")),
                "feasibility_score": round(feas, 2),
                "limitations": str(interp.get("limitations", "")),
                "conclusion": str(interp.get("conclusion", "")),
            }
        }

    def run(self) -> dict[str, Any]:
        context = self.read_context()

        hypothesis_data = context.get("hypothesis_expansion")
        coding_data = context.get("coding")
        discovery_data = context.get("dataset_discovery")

        if not isinstance(hypothesis_data, dict):
            raise ValueError("ResultsInterpreterAgent requires 'hypothesis_expansion' in context.")
        if not isinstance(coding_data, dict):
            raise ValueError("ResultsInterpreterAgent requires 'coding' in context.")

        expanded = hypothesis_data.get("expanded_hypotheses", [])
        if not expanded:
            raise ValueError("ResultsInterpreterAgent requires non-empty expanded_hypotheses.")
        hyp = expanded[0]

        coding_result = coding_data.get("coding_result", {})
        selected_files = []
        if isinstance(discovery_data, dict):
            dr = discovery_data.get("discovery_result", {})
            selected_files = dr.get("selected_files", [])

        system_prompt = (
            "You are a results interpreter and statistical reviewer. "
            "Analyze the statistical output from the coding agent and evaluate whether "
            "the hypothesis is supported by the data.\n\n"
            "Return strict JSON only in this schema: {\"interpretation\": {"
            "\"hypothesis_id\": str, \"hypothesis_supported\": bool, "
            "\"statistical_evidence\": str, \"effect_sizes\": str, "
            "\"p_values\": str, \"feasibility_score\": float (0-10), "
            "\"limitations\": str, \"conclusion\": str}}"
        )
        user_prompt = self._build_user_prompt(hyp, coding_result, selected_files)

        try:
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.STRONG_MODEL,
                response_format="json",
            )
            output = self._normalize(raw, hyp)
        except Exception:
            output = self._fallback(hyp, coding_result)

        self.write_output(output)
        return output
