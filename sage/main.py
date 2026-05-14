"""SAGE Pipeline — End-to-End orchestrator (Spec 10B).

Runs all 10 agents in fixed sequential order, merges Explainability Index
into the final report, and saves to outputs/final_report.json.
"""

from __future__ import annotations

import argparse
import json
import os

from sage import config
from sage.memory_store import MemoryStore
from sage.mock_kg import build_mock_kg
from sage.mock_data_bank import generate_data_bank

from sage.agents.path_generation import PathGenerationAgent
from sage.agents.ontologist import OntologistAgent
from sage.agents.scientist import ScientistAgent
from sage.agents.hypothesis_expansion import HypothesisExpansionAgent
from sage.agents.novelty_debate import NoveltyDebateAgent
from sage.agents.explainability import ExplainabilityAgent
from sage.agents.dataset_discovery import DatasetDiscoveryAgent
from sage.agents.coding import CodingAgent
from sage.agents.results_interpreter import ResultsInterpreterAgent
from sage.agents.summary import SummaryAgent


def _ensure_infrastructure() -> None:
    """Make sure mock KG and data bank exist before running the pipeline."""
    # KG
    kg_path = config.KG_PATH
    if not os.path.isfile(kg_path):
        print("Building mock KG...")
        build_mock_kg()

    # Data bank
    bank_dir = config.DATA_BANK_DIR
    if not os.path.isdir(bank_dir):
        print("Generating mock data bank...")
        generate_data_bank(output_dir=bank_dir)

    # Output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def run_pipeline(query: str) -> dict:
    _ensure_infrastructure()

    memory = MemoryStore()
    memory.set_user_query(query)

    agents = [
        PathGenerationAgent(memory),        # 1
        OntologistAgent(memory),             # 2
        ScientistAgent(memory),              # 3 (with human checkpoint)
        HypothesisExpansionAgent(memory),    # 4
        NoveltyDebateAgent(memory),          # 5
        ExplainabilityAgent(memory),         # 6
        DatasetDiscoveryAgent(memory),       # 7
        CodingAgent(memory),                 # 8
        ResultsInterpreterAgent(memory),     # 9
        SummaryAgent(memory),                # 10
    ]

    for agent in agents:
        print(f"\n{'=' * 60}")
        print(f"Running: {agent.name}")
        print(f"{'=' * 60}")
        output = agent.run()
        # write_output is already called inside each agent's run(),
        # but the spec shows an explicit call here — make it idempotent
        if agent.name not in memory._store:
            agent.write_output(output)
        print(f"  -> Output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")

    # Merge Explainability results into the final report
    final_report = memory._store["summary"]
    ei_result = memory._store.get("explainability", {})
    final_report["report"]["explainability_index"] = ei_result.get("explainability_results", [])

    # Save final report
    report_path = os.path.join(config.OUTPUT_DIR, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(json.dumps(final_report["report"], indent=2))
    print(f"\nReport saved to: {report_path}")

    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE Pipeline")
    parser.add_argument(
        "--query",
        default="prognostic biomarkers for bladder cancer survival",
        help="Research query to drive hypothesis generation",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=True,
        help="Skip human checkpoint (default: True)",
    )
    args = parser.parse_args()

    config.AUTO_APPROVE = args.auto_approve
    run_pipeline(args.query)


if __name__ == "__main__":
    main()
