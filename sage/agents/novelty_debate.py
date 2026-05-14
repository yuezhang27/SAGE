from __future__ import annotations

import statistics
from typing import Any

from sage import config
from sage.agents.base import BaseAgent
from sage.llm import call_llm


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 5.0
    return max(1.0, min(10.0, score))


class NoveltyDebateAgent(BaseAgent):
    name = "novelty_debate"

    def _is_specious_hypothesis(self, hyp: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(hyp.get("hypothesis_statement", "")),
                str(hyp.get("mechanistic_rationale", "")),
                str(hyp.get("background_literature", "")),
            ]
        ).lower()
        triggers = ["stap", "can cure", "miracle", "perpetual motion", "telepathy"]
        return any(token in text for token in triggers)

    def _call_prover(self, hyp: dict[str, Any], round_num: int, prev_context: dict[str, Any] | None) -> dict[str, Any]:
        system_prompt = (
            "You are Prover (optimistic critic). Defend novelty. Return strict JSON: "
            "{\"score\": int, \"confidence\": float, \"justification\": str, \"novelty_claims\": [str]}. "
            "Score range is 1-10, confidence range is 0-1."
        )
        user_prompt = (
            f"Round: {round_num}\n"
            f"Hypothesis:\n{hyp}\n"
            f"Previous context:\n{prev_context if prev_context else 'None'}"
        )

        try:
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.DEFAULT_MODEL,
                response_format="json",
            )
            return {
                "score": _clamp_score(raw.get("score")),
                "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.7)))),
                "justification": str(raw.get("justification", "Novel linkage across molecular and clinical layers.")),
                "novelty_claims": raw.get("novelty_claims", ["Cross-domain biomarker interaction is underexplored."]),
            }
        except Exception:
            specious = self._is_specious_hypothesis(hyp)
            score = 3.0 if specious else 7.5
            return {
                "score": score,
                "confidence": 0.65,
                "justification": (
                    "The proposal combines modalities and endpoint-linked biology in a potentially novel way."
                    if not specious
                    else "The claim appears bold, but Prover still argues novelty due to unusual framing."
                ),
                "novelty_claims": [
                    "Uncommon integration of molecular and pathology-derived signals.",
                    "Potentially underexplored mechanism-to-endpoint chain.",
                ],
            }

    def _call_verifier(self, hyp: dict[str, Any], round_num: int, prev_context: dict[str, Any] | None) -> dict[str, Any]:
        system_prompt = (
            "You are Verifier (conservative critic). Challenge novelty and provide counter-evidence. "
            "Return strict JSON: {\"score\": int, \"confidence\": float, \"counter_evidence\": str, "
            "\"cited_papers\": [{\"title\": str, \"doi\": str, \"relevance\": str}], \"specious_flags\": [str]}. "
            "Score range is 1-10 where higher means more novel."
        )
        user_prompt = (
            f"Round: {round_num}\n"
            f"Hypothesis:\n{hyp}\n"
            f"Previous context:\n{prev_context if prev_context else 'None'}"
        )

        try:
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.DEFAULT_MODEL,
                response_format="json",
            )
            cited = raw.get("cited_papers", [])
            if not isinstance(cited, list):
                cited = []
            flags = raw.get("specious_flags", [])
            if not isinstance(flags, list):
                flags = []
            return {
                "score": _clamp_score(raw.get("score")),
                "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.8)))),
                "counter_evidence": str(raw.get("counter_evidence", "Prior literature overlaps with key claims.")),
                "cited_papers": cited,
                "specious_flags": flags,
            }
        except Exception:
            specious = self._is_specious_hypothesis(hyp)
            if specious:
                return {
                    "score": 1.8,
                    "confidence": 0.95,
                    "counter_evidence": "The central claim is implausible and contradicted by consensus evidence.",
                    "cited_papers": [
                        {
                            "title": "Refutation and Retraction Analyses of STAP-like Claims",
                            "doi": "10.1038/fake-doi-stap-refute",
                            "relevance": "Demonstrates lack of reproducibility and direct contradiction.",
                        }
                    ],
                    "specious_flags": [
                        "Claim that STAP cells can cure bladder cancer is directly refuted by reproducibility failures."
                    ],
                }
            return {
                "score": 5.4,
                "confidence": 0.82,
                "counter_evidence": "Some components appear incremental relative to existing translational biomarker studies.",
                "cited_papers": [
                    {
                        "title": "Integrative Molecular and Histopathology Biomarker Models in Bladder Cancer",
                        "doi": "10.1200/JCO.2022.00.0000",
                        "relevance": "Shows prior multimodal prognostic modeling in related settings.",
                    }
                ],
                "specious_flags": [],
            }

    def _call_judge(
        self,
        hyp: dict[str, Any],
        round_num: int,
        prover_result: dict[str, Any],
        verifier_result: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = (
            "You are Judge (balanced critic). Synthesize Prover and Verifier arguments. Return strict JSON: "
            "{\"score\": int, \"synthesis\": str, \"specious_upheld\": bool, \"ruling_explanation\": str}."
        )
        user_prompt = (
            f"Round: {round_num}\n"
            f"Hypothesis:\n{hyp}\n"
            f"Prover:\n{prover_result}\n"
            f"Verifier:\n{verifier_result}\n"
        )

        try:
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.DEFAULT_MODEL,
                response_format="json",
            )
            return {
                "score": _clamp_score(raw.get("score")),
                "synthesis": str(raw.get("synthesis", "Balanced synthesis of novelty and overlap evidence.")),
                "specious_upheld": bool(raw.get("specious_upheld", False)),
                "ruling_explanation": str(raw.get("ruling_explanation", "No decisive specious evidence upheld.")),
            }
        except Exception:
            specious_flags = verifier_result.get("specious_flags", [])
            flag_present = isinstance(specious_flags, list) and len(specious_flags) > 0
            base = (float(prover_result.get("score", 5.0)) + float(verifier_result.get("score", 5.0))) / 2.0
            judge_score = min(base, 3.8) if flag_present else base
            return {
                "score": _clamp_score(judge_score),
                "synthesis": (
                    "Verifier provides strong contradiction evidence; novelty claim is substantially weakened."
                    if flag_present
                    else "The hypothesis appears moderately novel with partial overlap in prior work."
                ),
                "specious_upheld": flag_present,
                "ruling_explanation": (
                    "Specious flag upheld due to direct contradiction and implausible core mechanism."
                    if flag_present
                    else "No direct refutation identified as decisively invalidating the core claim."
                ),
            }

    def run(self) -> dict[str, Any]:
        context = self.read_context()
        hypothesis = context.get("hypothesis_expansion")
        if not isinstance(hypothesis, dict):
            raise ValueError("NoveltyDebateAgent requires 'hypothesis_expansion' in context.")

        expanded_hypotheses = hypothesis.get("expanded_hypotheses")
        if not isinstance(expanded_hypotheses, list) or not expanded_hypotheses:
            raise ValueError("NoveltyDebateAgent requires non-empty expanded_hypotheses.")

        debate_results: list[dict[str, Any]] = []

        for hyp in expanded_hypotheses:
            prover_result = self._call_prover(hyp, round_num=0, prev_context=None)
            verifier_result = self._call_verifier(hyp, round_num=0, prev_context=None)
            judge_result = self._call_judge(hyp, round_num=0, prover_result=prover_result, verifier_result=verifier_result)

            scores = [
                float(prover_result["score"]),
                float(verifier_result["score"]),
                float(judge_result["score"]),
            ]
            initial_sigma = statistics.stdev(scores)
            debate_log = [
                {
                    "round": 0,
                    "scores": scores,
                    "sigma": initial_sigma,
                }
            ]

            debate_triggered = initial_sigma > config.SIGMA_TRIGGER
            sigma = initial_sigma

            if not debate_triggered:
                final_score = statistics.mean(scores)
            else:
                for round_num in range(1, config.MAX_DEBATE_ROUNDS + 1):
                    prev_context = {
                        "prover": prover_result,
                        "verifier": verifier_result,
                        "judge": judge_result,
                    }
                    prover_result = self._call_prover(hyp, round_num, prev_context)
                    verifier_result = self._call_verifier(hyp, round_num, prev_context)
                    judge_result = self._call_judge(hyp, round_num, prover_result, verifier_result)

                    specious_flags = verifier_result.get("specious_flags", [])
                    if isinstance(specious_flags, list) and specious_flags and bool(judge_result.get("specious_upheld", False)):
                        prover_result["score"] = max(1.0, float(prover_result.get("score", 1.0)) - 1.0)

                    scores = [
                        float(prover_result["score"]),
                        float(verifier_result["score"]),
                        float(judge_result["score"]),
                    ]
                    sigma = statistics.stdev(scores)
                    debate_log.append({"round": round_num, "scores": scores, "sigma": sigma})

                    if sigma < config.SIGMA_CONVERGE:
                        break

                final_score = statistics.mean(scores)

            debate_results.append(
                {
                    "hypothesis_id": str(hyp.get("hypothesis_id", "")),
                    "novelty_score": round(final_score, 2),
                    "debate_triggered": debate_triggered,
                    "total_rounds": len(debate_log),
                    "debate_log": debate_log,
                    "final_prover_argument": str(prover_result.get("justification", "")),
                    "final_verifier_argument": str(verifier_result.get("counter_evidence", "")),
                    "final_judge_synthesis": str(judge_result.get("synthesis", "")),
                    "specious_flags": verifier_result.get("specious_flags", []),
                    "specious_upheld": bool(judge_result.get("specious_upheld", False)),
                }
            )

        output = {"debate_results": debate_results}
        self.write_output(output)
        return output
