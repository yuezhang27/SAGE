from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage.agents.base import BaseAgent
from sage.memory_store import MemoryStore
from sage.mock_data_bank import generate_data_bank
from sage.mock_kg import build_mock_kg, load_mock_kg


class _DummyAgent(BaseAgent):
    def __init__(self, memory: MemoryStore, name: str):
        super().__init__(memory)
        self.name = name

    def run(self) -> dict:
        return {}


def _status(flag: bool, title: str) -> None:
    print(f"{'[PASS]' if flag else '[FAIL]'} {title}")


def verify_kg() -> bool:
    print("\n== KG Statistics ==")
    kg = build_mock_kg()
    loaded = load_mock_kg()

    node_count = kg.number_of_nodes()
    edge_count = kg.number_of_edges()
    type_counter = Counter(data.get("entity_type", "Unknown") for _, data in kg.nodes(data=True))
    degrees = [deg for _, deg in kg.degree()]

    print(f"nodes: {node_count}")
    print(f"edges: {edge_count}")
    print("entity_type counts:")
    for key in sorted(type_counter.keys()):
        print(f"  - {key}: {type_counter[key]}")
    print(
        "degree stats: "
        f"min={min(degrees)}, max={max(degrees)}, mean={sum(degrees) / len(degrees):.2f}"
    )

    ok_counts = node_count >= 45 and edge_count >= 75
    ok_round_trip = (
        kg.number_of_nodes() == loaded.number_of_nodes()
        and kg.number_of_edges() == loaded.number_of_edges()
    )

    _status(ok_counts, "KG node/edge scale check")
    _status(ok_round_trip, "GraphML round-trip node/edge consistency")

    return ok_counts and ok_round_trip


def verify_data_bank() -> bool:
    print("\n== Data Bank ==")
    paths = generate_data_bank()

    all_ok = True
    for filename, filepath in paths.items():
        df = pd.read_csv(filepath)
        print(f"\n{filename}")
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
        valid_shape = df.shape[0] >= 95
        all_ok = all_ok and valid_shape
        _status(valid_shape, f"{filename} row-count >= 95")

    clinical = pd.read_csv(paths["clinical_metadata.csv"])
    genes = pd.read_csv(paths["gene_expression.csv"])
    merged = clinical.merge(genes[["patient_id", "CXCL13"]], on="patient_id", how="inner")
    q75 = merged["CXCL13"].quantile(0.75)
    high = merged[merged["CXCL13"] > q75]["os_months"].mean()
    low = merged[merged["CXCL13"] <= q75]["os_months"].mean()
    direction_ok = high > low

    print("\nCXCL13 high-vs-low OS check")
    print(f"mean_os_high: {high:.3f}")
    print(f"mean_os_low : {low:.3f}")
    _status(direction_ok, "CXCL13 high group has longer OS")

    return all_ok and direction_ok


def verify_memory_acl() -> bool:
    print("\n== MemoryStore ACL ==")
    memory = MemoryStore()
    memory.set_user_query("prognostic biomarkers for bladder cancer survival")
    memory.write("path_generation", {"paths": ["mock_path"]})

    ontologist = _DummyAgent(memory, "ontologist")
    scientist = _DummyAgent(memory, "scientist")
    novelty = _DummyAgent(memory, "novelty_debate")

    ontologist_ctx = ontologist.read_context()
    ontologist_ok = "path_generation" in ontologist_ctx
    _status(ontologist_ok, "ontologist can read path_generation")

    memory.write("ontologist", {"selected_paths": ["mock_selected"]})
    scientist_ctx = scientist.read_context()
    scientist_ok = "path_generation" not in scientist_ctx and "ontologist" in scientist_ctx
    _status(scientist_ok, "scientist can read only ontologist")

    memory.write("hypothesis_expansion", {"expanded_hypotheses": ["mock_hypothesis"]})
    novelty_ctx = novelty.read_context()
    novelty_ok = set(novelty_ctx.keys()) == {"hypothesis_expansion"}
    _status(novelty_ok, "novelty_debate reads only hypothesis_expansion")

    return ontologist_ok and scientist_ok and novelty_ok


def main() -> None:
    ok_kg = verify_kg()
    ok_data = verify_data_bank()
    ok_acl = verify_memory_acl()

    print("\n== SPEC1 RESULT ==")
    all_ok = ok_kg and ok_data and ok_acl
    _status(all_ok, "Spec 1 acceptance checks")


if __name__ == "__main__":
    main()
