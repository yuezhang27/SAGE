from __future__ import annotations

from typing import Any

from sage.agents.base import BaseAgent
from sage.audit import log_agent_start, log_agent_end, log_detail, step_for
from sage.llm import call_llm


class HypothesisExpansionAgent(BaseAgent):
    name = "hypothesis_expansion"

    def _build_user_prompt(self, hypotheses: list[dict[str, Any]], selected_paths: list[dict[str, Any]]) -> str:
        hyp_lines: list[str] = []
        for hyp in hypotheses:
            hyp_lines.append(
                "\n".join(
                    [
                        f"hypothesis_id: {hyp.get('hypothesis_id', '')}",
                        f"source_path_id: {hyp.get('source_path_id', '')}",
                        f"statement: {hyp.get('hypothesis_statement', '')}",
                        f"population: {hyp.get('population', '')}",
                        f"variables: {hyp.get('variables', {})}",
                        f"outcome: {hyp.get('outcome', '')}",
                        f"expected_directionality: {hyp.get('expected_directionality', '')}",
                        f"validation_feasibility: {hyp.get('validation_feasibility', '')}",
                    ]
                )
            )

        path_lines: list[str] = []
        for path in selected_paths:
            path_lines.append(
                "\n".join(
                    [
                        f"path_id: {path.get('path_id', '')}",
                        f"narrative_summary: {path.get('narrative_summary', '')}",
                        "nodes: "
                        + "; ".join(
                            f"{n.get('name', '')}({n.get('type', '')}): {n.get('definition', '')}"
                            for n in path.get("nodes", [])
                        ),
                        "edges: "
                        + "; ".join(
                            f"{e.get('source', '')}->{e.get('target', '')}({e.get('relation', '')}): "
                            f"{e.get('biological_interpretation', '')}"
                            for e in path.get("edges", [])
                        ),
                    ]
                )
            )

        return (
            "Scientist hypotheses:\n\n"
            + "\n\n".join(hyp_lines)
            + "\n\nOntologist path annotations:\n\n"
            + "\n\n".join(path_lines)
        )

    def _fallback(self, hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
        expanded: list[dict[str, Any]] = []
        for hyp in hypotheses:
            variables = hyp.get("variables") if isinstance(hyp.get("variables"), dict) else {}
            biological = str(variables.get("biological", "biomarker"))
            imaging = str(variables.get("imaging", "imaging feature"))
            outcome = str(hyp.get("outcome", "overall survival"))
            expanded.append(
                {
                    "hypothesis_id": str(hyp.get("hypothesis_id", "")),
                    "hypothesis_statement": str(hyp.get("hypothesis_statement", "")),
                    "population": str(hyp.get("population", "bladder cancer cohort")),
                    "variables": {
                        "biological": biological,
                        "imaging": imaging,
                        "clinical": str(variables.get("clinical", "age, sex, stage")),
                    },
                    "outcome": outcome,
                    "expected_directionality": str(hyp.get("expected_directionality", "positive association")),
                    "mechanistic_rationale": (
                        f"{biological} is positioned as a mechanistic contributor in tumor-immune dynamics, "
                        f"while {imaging} captures spatial phenotype linked to {outcome}."
                    ),
                    "background_literature": (
                        "Prior translational studies in bladder cancer support integrating molecular and "
                        "pathology-derived features for prognostic stratification."
                    ),
                    "clinical_significance": (
                        "A combined molecular-imaging signature can improve risk stratification and "
                        "support treatment planning."
                    ),
                    "validation_strategy": (
                        "Use matched molecular, imaging, and clinical variables to test association with "
                        "survival endpoints using survival analysis and subgroup checks."
                    ),
                }
            )
        return {"expanded_hypotheses": expanded}

    def _normalize(self, raw: Any, hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._fallback(hypotheses)

        expanded = raw.get("expanded_hypotheses")
        if not isinstance(expanded, list) or not expanded:
            return self._fallback(hypotheses)

        source_map = {str(h.get("hypothesis_id", "")): h for h in hypotheses}
        cleaned: list[dict[str, Any]] = []
        for i, item in enumerate(expanded, start=1):
            if not isinstance(item, dict):
                continue

            hypothesis_id = str(item.get("hypothesis_id", ""))
            if not hypothesis_id and hypotheses:
                hypothesis_id = str(hypotheses[min(i - 1, len(hypotheses) - 1)].get("hypothesis_id", f"H{i}"))

            source = source_map.get(hypothesis_id, {})
            src_vars = source.get("variables") if isinstance(source.get("variables"), dict) else {}
            variables = item.get("variables") if isinstance(item.get("variables"), dict) else {}

            cleaned.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_statement": str(
                        item.get("hypothesis_statement", source.get("hypothesis_statement", ""))
                    ),
                    "population": str(item.get("population", source.get("population", "bladder cancer cohort"))),
                    "variables": {
                        "biological": str(variables.get("biological", src_vars.get("biological", "biomarker"))),
                        "imaging": str(variables.get("imaging", src_vars.get("imaging", "imaging feature"))),
                        "clinical": str(variables.get("clinical", src_vars.get("clinical", "age, sex, stage"))),
                    },
                    "outcome": str(item.get("outcome", source.get("outcome", "overall survival"))),
                    "expected_directionality": str(
                        item.get("expected_directionality", source.get("expected_directionality", "positive association"))
                    ),
                    "mechanistic_rationale": str(item.get("mechanistic_rationale", "Mechanistic support is described.")),
                    "background_literature": str(
                        item.get("background_literature", "Relevant literature context is summarized.")
                    ),
                    "clinical_significance": str(
                        item.get("clinical_significance", "Potential clinical utility is articulated.")
                    ),
                    "validation_strategy": str(
                        item.get("validation_strategy", "Validation plan with measurable variables is provided.")
                    ),
                }
            )

        if not cleaned:
            return self._fallback(hypotheses)
        return {"expanded_hypotheses": cleaned}

    def run(self) -> dict[str, Any]:
        context = self.read_context()
        _step = step_for(self.name)
        log_agent_start(self.name, _step, list(context.keys()))

        scientist = context.get("scientist")
        ontologist = context.get("ontologist")

        if not isinstance(scientist, dict):
            raise ValueError("HypothesisExpansionAgent requires 'scientist' output in context.")
        if not isinstance(ontologist, dict):
            raise ValueError("HypothesisExpansionAgent requires 'ontologist' output in context.")

        hypotheses = scientist.get("hypotheses")
        selected_paths = ontologist.get("selected_paths")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError("HypothesisExpansionAgent requires non-empty scientist hypotheses.")
        if not isinstance(selected_paths, list) or not selected_paths:
            raise ValueError("HypothesisExpansionAgent requires non-empty ontologist selected_paths.")

        log_detail("Input hypotheses", f"{len(hypotheses)} from scientist")
        log_detail("Input paths", f"{len(selected_paths)} from ontologist")

        system_prompt = (
            "You are a hypothesis expansion scientist. Expand each scientist hypothesis into a full scientific "
            "proposal with mechanistic rationale, literature context, clinical significance, and validation strategy. "
            "Return strict JSON only in this schema: {\"expanded_hypotheses\":[{\"hypothesis_id\":str,"
            "\"hypothesis_statement\":str,\"population\":str,\"variables\":{\"biological\":str,"
            "\"imaging\":str,\"clinical\":str},\"outcome\":str,\"expected_directionality\":str,"
            "\"mechanistic_rationale\":str,\"background_literature\":str,\"clinical_significance\":str,"
            "\"validation_strategy\":str}]}"
        )
        user_prompt = self._build_user_prompt(hypotheses, selected_paths)

        try:
            raw = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, response_format="json")
            output = self._normalize(raw, hypotheses)
        except Exception:
            output = self._fallback(hypotheses)

        for eh in output.get("expanded_hypotheses", []):
            log_detail(eh.get("hypothesis_id", "?"), eh.get("hypothesis_statement", "")[:100])
        log_agent_end(self.name, _step, f"{len(output.get('expanded_hypotheses', []))} hypotheses expanded (trunk document)")
        self.write_output(output)
        return output
