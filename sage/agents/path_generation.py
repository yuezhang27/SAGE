from __future__ import annotations

import math
from collections import Counter
from typing import Any

import networkx as nx

from sage import config
from sage.agents.base import BaseAgent
from sage.audit import log_agent_start, log_agent_end, log_detail, log_section, step_for
from sage.llm import call_llm
from sage.mock_kg import load_mock_kg


def novelty_score(path_nodes: list[str], graph: nx.DiGraph) -> float:
    epsilon = 1.0
    total_degree = sum(graph.degree(v) + epsilon for v in graph.nodes())
    info_content: list[float] = []
    for node in path_nodes:
        p_e = (graph.degree(node) + epsilon) / total_degree
        info_content.append(-math.log(p_e))
    return sum(info_content) / len(info_content)


def surprise_score(path_nodes: list[str], graph: nx.DiGraph) -> float:
    epsilon = 1e-9

    global_types = [graph.nodes[node].get("entity_type", "Unknown") for node in graph.nodes()]
    global_counter = Counter(global_types)
    total_global = sum(global_counter.values())

    path_types = [graph.nodes[node].get("entity_type", "Unknown") for node in path_nodes]
    path_counter = Counter(path_types)
    total_path = sum(path_counter.values())

    kl = 0.0
    for entity_type, count in path_counter.items():
        p_actual = count / total_path
        p_expected = global_counter.get(entity_type, 0) / total_global
        if p_expected == 0 and p_actual > 0:
            p_expected = epsilon
        if p_actual > 0:
            kl += p_actual * math.log(p_actual / p_expected)
    return kl


class PathGenerationAgent(BaseAgent):
    name = "path_generation"

    def __init__(self, memory, graph: nx.DiGraph | None = None):
        super().__init__(memory)
        self.graph = graph or load_mock_kg()
        self.last_candidate_count = 0
        self.last_raw_scores: dict[tuple[str, ...], dict[str, float]] = {}

    def _llm_or_heuristic_score(self, query: str, path_nodes: list[str], score_type: str) -> float:
        node_text = " -> ".join(path_nodes)
        edge_text = []
        for src, dst in zip(path_nodes[:-1], path_nodes[1:]):
            relation = self.graph.edges[src, dst].get("relation_type", "related_to")
            edge_text.append(f"{src} -[{relation}]-> {dst}")
        relation_text = "; ".join(edge_text)

        system_prompt = (
            "You are a biomedical reasoning scorer. Return strict JSON: {\"score\": float}. "
            "The score must be in [0,1]."
        )
        if score_type == "logic":
            user_prompt = (
                f"Query: {query}\n"
                f"Path nodes: {node_text}\n"
                f"Path relations: {relation_text}\n"
                "Evaluate semantic logic coherence between the query and this path."
            )
        else:
            user_prompt = (
                f"Query: {query}\n"
                f"Path nodes: {node_text}\n"
                f"Path relations: {relation_text}\n"
                "Evaluate disease relevance for each node and return overall average relevance."
            )

        try:
            result = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, response_format="json")
            score = float(result.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except Exception:
            return self._heuristic_score(query, path_nodes, score_type)

    def _heuristic_score(self, query: str, path_nodes: list[str], score_type: str) -> float:
        q_tokens = {token.lower() for token in query.replace("_", " ").split()}
        path_text = " ".join(path_nodes).replace("_", " ").lower()

        overlap = sum(1 for token in q_tokens if token in path_text)
        overlap_score = min(1.0, overlap / max(1, len(q_tokens)))

        disease_terms = {"bladder", "cancer", "survival", "prognostic", "biomarker", "pfs"}
        disease_hits = sum(1 for node in path_nodes if any(t in node.lower() for t in disease_terms))
        disease_score = min(1.0, disease_hits / max(1, len(path_nodes)))

        if score_type == "logic":
            return 0.55 * overlap_score + 0.45 * disease_score
        return 0.35 * overlap_score + 0.65 * disease_score

    def _candidate_paths(self) -> list[list[str]]:
        graph = self.graph
        type_pairs = [
            ("Gene", "ClinicalEndpoint"),
            ("Biomarker", "ClinicalEndpoint"),
            ("Pathway", "ClinicalEndpoint"),
            ("Gene", "Disease"),
            ("Gene", "Biomarker"),
        ]

        typed_nodes: dict[str, list[str]] = {}
        for node, data in graph.nodes(data=True):
            typed_nodes.setdefault(data.get("entity_type", "Unknown"), []).append(node)

        dedup: set[tuple[str, ...]] = set()
        collected: list[list[str]] = []

        for src_type, dst_type in type_pairs:
            for src in typed_nodes.get(src_type, []):
                for dst in typed_nodes.get(dst_type, []):
                    if src == dst:
                        continue
                    try:
                        for path in nx.all_simple_paths(graph, source=src, target=dst, cutoff=5):
                            hop_len = len(path) - 1
                            if hop_len < 2 or hop_len > 5:
                                continue
                            key = tuple(path)
                            if key in dedup:
                                continue
                            dedup.add(key)
                            collected.append(path)
                    except nx.NetworkXNoPath:
                        continue

        collected.sort(key=lambda p: sum(graph.degree(node) for node in p) / len(p))
        max_candidates = 800
        return collected[:max_candidates]

    @staticmethod
    def _minmax_scale(values: list[float]) -> list[float]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if math.isclose(min_v, max_v):
            return [0.5 for _ in values]
        return [(v - min_v) / (max_v - min_v) for v in values]

    def run(self) -> dict[str, Any]:
        context = self.read_context()
        if "__user_query__" not in context:
            raise ValueError("PathGenerationAgent requires __user_query__ in MemoryStore context.")

        query = str(context["__user_query__"])
        _step = step_for(self.name)
        log_agent_start(self.name, _step, list(context.keys()))
        log_detail("User query", query)
        log_detail("KG size", f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        candidates = self._candidate_paths()
        self.last_candidate_count = len(candidates)
        log_detail("Candidate paths found", len(candidates))

        logic_raw: list[float] = []
        relevance_raw: list[float] = []
        novelty_raw: list[float] = []
        surprise_raw: list[float] = []

        for path in candidates:
            logic_raw.append(self._heuristic_score(query, path, "logic"))
            relevance_raw.append(self._heuristic_score(query, path, "relevance"))
            novelty_raw.append(novelty_score(path, self.graph))
            surprise_raw.append(surprise_score(path, self.graph))

        logic_scaled = self._minmax_scale(logic_raw)
        relevance_scaled = self._minmax_scale(relevance_raw)
        novelty_scaled = self._minmax_scale(novelty_raw)
        surprise_scaled = self._minmax_scale(surprise_raw)

        weights = {"logic": 0.15, "relevance": 0.15, "novelty": 0.35, "surprise": 0.35}

        scored_paths: list[dict[str, Any]] = []
        for i, path in enumerate(candidates):
            total_score = (
                weights["logic"] * logic_scaled[i]
                + weights["relevance"] * relevance_scaled[i]
                + weights["novelty"] * novelty_scaled[i]
                + weights["surprise"] * surprise_scaled[i]
            )

            edges = []
            for src, dst in zip(path[:-1], path[1:]):
                edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "relation_type": self.graph.edges[src, dst].get("relation_type", "related_to"),
                    }
                )

            node_records = [
                {"name": node, "type": self.graph.nodes[node].get("entity_type", "Unknown")}
                for node in path
            ]

            scores = {
                "logic": round(logic_scaled[i], 6),
                "relevance": round(relevance_scaled[i], 6),
                "novelty": round(novelty_scaled[i], 6),
                "surprise": round(surprise_scaled[i], 6),
                "total": round(total_score, 6),
            }
            self.last_raw_scores[tuple(path)] = {
                "logic": logic_raw[i],
                "relevance": relevance_raw[i],
                "novelty": novelty_raw[i],
                "surprise": surprise_raw[i],
            }

            scored_paths.append(
                {
                    "nodes": node_records,
                    "edges": edges,
                    "scores": scores,
                }
            )

        scored_paths.sort(key=lambda x: x["scores"]["total"], reverse=True)
        top_k = scored_paths[: config.TOP_K_PATHS]
        for i, path in enumerate(top_k, start=1):
            path["path_id"] = f"P{i}"

        log_section("Top-K paths (Eq.14 weighted: logic=0.15, rel=0.15, nov=0.35, sur=0.35)")
        for p in top_k:
            nodes = " -> ".join(n["name"] for n in p["nodes"])
            s = p["scores"]
            log_detail(p["path_id"], f"total={s['total']:.3f} nov={s['novelty']:.3f} sur={s['surprise']:.3f} | {nodes}")
        log_agent_end(self.name, _step, f"{len(top_k)} paths selected from {len(candidates)} candidates")

        return {
            "query": query,
            "paths": top_k,
        }
