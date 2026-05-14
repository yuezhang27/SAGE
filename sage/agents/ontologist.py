from __future__ import annotations

from typing import Any

from sage.agents.base import BaseAgent
from sage.llm import call_llm


class OntologistAgent(BaseAgent):
    name = "ontologist"

    def _format_paths_for_prompt(self, paths: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for path in paths:
            nodes = " -> ".join(f"{n['name']}({n['type']})" for n in path.get("nodes", []))
            edges = " | ".join(
                f"{e['source']}-[{e['relation_type']}]->{e['target']}" for e in path.get("edges", [])
            )
            blocks.append(
                "\n".join(
                    [
                        f"path_id: {path.get('path_id', '')}",
                        f"nodes: {nodes}",
                        f"edges: {edges}",
                        f"scores: {path.get('scores', {})}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _fallback(self, paths: list[dict[str, Any]]) -> dict[str, Any]:
        type_definition = {
            "Gene": "A gene-level molecular feature associated with tumor biology or immune regulation.",
            "Pathway": "A biological signaling or process-level pathway connecting molecular activity to phenotype.",
            "Disease": "A disease-level entity used to ground the hypothesis in clinical pathology context.",
            "ClinicalEndpoint": "A measurable patient outcome endpoint used for prognostic or predictive evaluation.",
            "TissueRegion": "A tissue compartment in histopathology where biological processes are localized.",
            "Biomarker": "A quantifiable marker derived from molecular or imaging measurements.",
            "StainingMethod": "A pathology measurement method used to operationalize biomarker assessment.",
            "Algorithm": "An analysis method used to statistically validate endpoint associations.",
            "Unknown": "A biomedical entity used in pathway-level interpretation.",
        }

        relation_interpretation = {
            "upregulates": "The upstream entity is interpreted as increasing the activity or abundance of the downstream entity.",
            "downregulates": "The upstream entity is interpreted as suppressing the downstream process or feature.",
            "associated_with": "The entities are interpreted as biologically linked within bladder cancer context.",
            "located_in": "The source process or entity is interpreted as spatially enriched in the target region.",
            "measured_by": "The target is interpreted as an operational method for quantifying the source entity.",
            "predicts": "The source is interpreted as carrying prognostic or predictive signal for the target endpoint.",
            "part_of": "The source is interpreted as a component of the target biological system.",
            "interacts_with": "The entities are interpreted as functionally interacting in disease-relevant biology.",
            "correlates_with": "The entities are interpreted as co-varying in clinically relevant observations.",
            "used_for": "The source method is interpreted as suitable for validating the target variable or endpoint.",
            "related_to": "The entities are interpreted as biologically related in a pathway narrative.",
        }

        selected: list[dict[str, Any]] = []
        for path in paths[:3]:
            nodes = []
            for node in path.get("nodes", []):
                node_type = node.get("type", "Unknown")
                nodes.append(
                    {
                        "name": node.get("name", ""),
                        "type": node_type,
                        "definition": type_definition.get(node_type, type_definition["Unknown"]),
                    }
                )

            edges = []
            for edge in path.get("edges", []):
                relation = edge.get("relation_type", "related_to")
                edges.append(
                    {
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                        "relation": relation,
                        "biological_interpretation": relation_interpretation.get(
                            relation, relation_interpretation["related_to"]
                        ),
                    }
                )

            node_chain = " -> ".join(node.get("name", "") for node in path.get("nodes", []))
            narrative = (
                "This pathway links molecular and tissue-level features to clinically relevant endpoints in "
                f"bladder cancer: {node_chain}."
            )

            selected.append(
                {
                    "path_id": path.get("path_id", ""),
                    "nodes": nodes,
                    "edges": edges,
                    "narrative_summary": narrative,
                    "scores": path.get("scores", {}),
                }
            )

        return {"selected_paths": selected}

    def _normalize_output(self, model_output: Any, original_paths: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(model_output, dict):
            return self._fallback(original_paths)

        selected_paths = model_output.get("selected_paths")
        if not isinstance(selected_paths, list) or not selected_paths:
            return self._fallback(original_paths)

        original_map = {p.get("path_id", ""): p for p in original_paths}
        normalized: list[dict[str, Any]] = []

        for item in selected_paths:
            if not isinstance(item, dict):
                continue

            path_id = str(item.get("path_id", ""))
            source = original_map.get(path_id, {})

            nodes = item.get("nodes") if isinstance(item.get("nodes"), list) else []
            edges = item.get("edges") if isinstance(item.get("edges"), list) else []

            if not nodes and source:
                for n in source.get("nodes", []):
                    nodes.append(
                        {
                            "name": n.get("name", ""),
                            "type": n.get("type", "Unknown"),
                            "definition": "Biological entity in bladder cancer pathway context.",
                        }
                    )
            else:
                clean_nodes = []
                for n in nodes:
                    if not isinstance(n, dict):
                        continue
                    clean_nodes.append(
                        {
                            "name": n.get("name", ""),
                            "type": n.get("type", "Unknown"),
                            "definition": n.get("definition", "Biological entity in pathway context."),
                        }
                    )
                nodes = clean_nodes

            if not edges and source:
                for e in source.get("edges", []):
                    edges.append(
                        {
                            "source": e.get("source", ""),
                            "target": e.get("target", ""),
                            "relation": e.get("relation_type", "related_to"),
                            "biological_interpretation": "Edge relation interpreted in disease-specific biological context.",
                        }
                    )
            else:
                clean_edges = []
                for e in edges:
                    if not isinstance(e, dict):
                        continue
                    clean_edges.append(
                        {
                            "source": e.get("source", ""),
                            "target": e.get("target", ""),
                            "relation": e.get("relation", e.get("relation_type", "related_to")),
                            "biological_interpretation": e.get(
                                "biological_interpretation",
                                "Edge relation interpreted in disease-specific biological context.",
                            ),
                        }
                    )
                edges = clean_edges

            narrative = item.get(
                "narrative_summary",
                "This path summarizes a plausible biological route linked to clinical outcomes.",
            )
            scores = item.get("scores") if isinstance(item.get("scores"), dict) else source.get("scores", {})

            normalized.append(
                {
                    "path_id": path_id,
                    "nodes": nodes,
                    "edges": edges,
                    "narrative_summary": narrative,
                    "scores": scores,
                }
            )

        if not normalized:
            return self._fallback(original_paths)
        return {"selected_paths": normalized}

    def run(self) -> dict[str, Any]:
        context = self.read_context()
        path_output = context.get("path_generation")
        if not isinstance(path_output, dict):
            raise ValueError("OntologistAgent requires 'path_generation' output in MemoryStore context.")

        paths = path_output.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("OntologistAgent requires non-empty path list from path_generation.")

        system_prompt = (
            "You are a biomedical ontologist. "
            "Task: (1) select the most impactful subset of paths, "
            "(2) provide standardized biological definition for each node, "
            "(3) provide biological interpretation for each edge, "
            "(4) provide one narrative summary per selected path. "
            "Return strict JSON only with this schema: "
            "{\"selected_paths\":[{\"path_id\":str,\"nodes\":[{\"name\":str,\"type\":str,\"definition\":str}],"
            "\"edges\":[{\"source\":str,\"target\":str,\"relation\":str,\"biological_interpretation\":str}],"
            "\"narrative_summary\":str,\"scores\":{\"logic\":float,\"relevance\":float,\"novelty\":float,\"surprise\":float,\"total\":float}}]}"
        )
        user_prompt = (
            "Input paths from Path Generation:\n\n"
            f"{self._format_paths_for_prompt(paths)}\n\n"
            "Keep score values aligned with input path scores."
        )

        try:
            result = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, response_format="json")
            return self._normalize_output(result, paths)
        except Exception:
            return self._fallback(paths)
