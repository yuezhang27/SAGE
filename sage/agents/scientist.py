from __future__ import annotations

import json
from typing import Any

from sage import config
from sage.agents.base import BaseAgent
from sage.llm import call_llm


class ScientistAgent(BaseAgent):
    name = "scientist"

    def __init__(self, memory):
        super().__init__(memory)
        self.last_checkpoint_action = ""

    def _build_user_prompt(self, selected_paths: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for path in selected_paths:
            nodes = " -> ".join(node.get("name", "") for node in path.get("nodes", []))
            node_defs = "; ".join(
                f"{n.get('name', '')}: {n.get('definition', '')}" for n in path.get("nodes", [])
            )
            edge_explain = "; ".join(
                f"{e.get('source', '')}->{e.get('target', '')} ({e.get('relation', '')}): "
                f"{e.get('biological_interpretation', '')}"
                for e in path.get("edges", [])
            )
            blocks.append(
                "\n".join(
                    [
                        f"path_id: {path.get('path_id', '')}",
                        f"node_chain: {nodes}",
                        f"node_definitions: {node_defs}",
                        f"edge_interpretations: {edge_explain}",
                        f"narrative: {path.get('narrative_summary', '')}",
                        f"scores: {path.get('scores', {})}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _fallback(self, selected_paths: list[dict[str, Any]]) -> dict[str, Any]:
        hypotheses: list[dict[str, Any]] = []
        for i, path in enumerate(selected_paths[:3], start=1):
            node_names = [n.get("name", "") for n in path.get("nodes", [])]

            biological = next(
                (n.get("name", "") for n in path.get("nodes", []) if n.get("type") == "Gene"),
                node_names[0] if node_names else "biological_marker",
            )
            imaging = next(
                (
                    n.get("name", "")
                    for n in path.get("nodes", [])
                    if n.get("type") in {"Biomarker", "TissueRegion"}
                ),
                "TLS_density",
            )
            clinical = "age, sex, stage"

            endpoint = next(
                (
                    n.get("name", "")
                    for n in path.get("nodes", [])
                    if n.get("type") == "ClinicalEndpoint"
                ),
                "overall_survival",
            )
            endpoint_text = endpoint.replace("_", " ")

            hypotheses.append(
                {
                    "hypothesis_id": f"H{i}",
                    "hypothesis_statement": (
                        f"In bladder cancer patients, higher {biological} combined with {imaging} "
                        f"is associated with improved {endpoint_text}."
                    ),
                    "population": "bladder cancer patients (focus on MIBC stage II-IV when available)",
                    "variables": {
                        "biological": biological,
                        "imaging": imaging,
                        "clinical": clinical,
                    },
                    "outcome": endpoint_text,
                    "expected_directionality": (
                        f"Higher {biological} and favorable {imaging} signal indicate better {endpoint_text}."
                    ),
                    "validation_feasibility": (
                        "Feasible using matched gene expression, pathology-derived imaging features, "
                        "and clinical follow-up metadata in the data bank."
                    ),
                    "source_path_id": path.get("path_id", f"P{i}"),
                }
            )
        return {"hypotheses": hypotheses}

    def _normalize(self, raw: Any, selected_paths: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._fallback(selected_paths)

        hypotheses = raw.get("hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            return self._fallback(selected_paths)

        cleaned: list[dict[str, Any]] = []
        known_path_ids = [p.get("path_id", "") for p in selected_paths]
        for i, item in enumerate(hypotheses, start=1):
            if not isinstance(item, dict):
                continue

            variables = item.get("variables") if isinstance(item.get("variables"), dict) else {}
            source_path_id = str(item.get("source_path_id", ""))
            if not source_path_id:
                source_path_id = known_path_ids[min(i - 1, len(known_path_ids) - 1)] if known_path_ids else f"P{i}"

            cleaned.append(
                {
                    "hypothesis_id": str(item.get("hypothesis_id", f"H{i}")),
                    "hypothesis_statement": str(item.get("hypothesis_statement", ""))
                    or "Biologically grounded and testable bladder cancer hypothesis.",
                    "population": str(item.get("population", "")) or "bladder cancer cohort",
                    "variables": {
                        "biological": str(variables.get("biological", "biological_marker")),
                        "imaging": str(variables.get("imaging", "imaging_feature")),
                        "clinical": str(variables.get("clinical", "age, sex, stage")),
                    },
                    "outcome": str(item.get("outcome", "overall survival")),
                    "expected_directionality": str(item.get("expected_directionality", "positive association")),
                    "validation_feasibility": str(
                        item.get(
                            "validation_feasibility",
                            "Feasible with matched molecular, imaging, and clinical variables.",
                        )
                    ),
                    "source_path_id": source_path_id,
                }
            )

        if not cleaned:
            return self._fallback(selected_paths)
        return {"hypotheses": cleaned}

    def _apply_human_checkpoint(self, output: dict[str, Any]) -> dict[str, Any]:
        if config.AUTO_APPROVE:
            self.last_checkpoint_action = "auto_approve_skipped"
            return output

        print("\n[Scientist] Human checkpoint - generated hypotheses:")
        for hyp in output.get("hypotheses", []):
            print(json.dumps(hyp, indent=2))

        feedback = input("Enter approve / reject / custom feedback: ").strip()
        normalized = feedback.lower()

        if normalized == "approve":
            self.last_checkpoint_action = "approved"
            return output
        if normalized == "reject":
            self.last_checkpoint_action = "rejected"
            raise RuntimeError("Hypotheses rejected at human checkpoint.")

        for hyp in output.get("hypotheses", []):
            hyp["human_feedback"] = feedback
        self.last_checkpoint_action = "feedback_attached"
        return output

    def run(self) -> dict[str, Any]:
        context = self.read_context()
        ontologist_output = context.get("ontologist")
        if not isinstance(ontologist_output, dict):
            raise ValueError("ScientistAgent requires 'ontologist' output in MemoryStore context.")

        selected_paths = ontologist_output.get("selected_paths")
        if not isinstance(selected_paths, list) or not selected_paths:
            raise ValueError("ScientistAgent requires non-empty selected_paths from ontologist output.")

        system_prompt = (
            "You are a hypothesis scientist. Generate concise, testable, dataset-aware hypotheses. "
            "Return strict JSON only with key 'hypotheses'. For each hypothesis include: "
            "hypothesis_id, hypothesis_statement, population, variables.biological, variables.imaging, "
            "variables.clinical, outcome, expected_directionality, validation_feasibility, source_path_id."
        )
        user_prompt = (
            "Ontologist-annotated paths:\n\n"
            f"{self._build_user_prompt(selected_paths)}\n\n"
            "Use one hypothesis per selected path where appropriate."
        )

        try:
            raw_output = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.STRONG_MODEL,
                response_format="json",
            )
            output = self._normalize(raw_output, selected_paths)
        except Exception:
            output = self._fallback(selected_paths)

        output = self._apply_human_checkpoint(output)
        self.write_output(output)
        return output
