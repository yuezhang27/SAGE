"""Dataset Discovery Agent (Spec 8).

Deterministic pipeline (no LLM except Phase 1 hypothesis parsing):
  Phase 1: Parse hypothesis into explicit constraints (LLM)
  Phase 2: Build Artifact Requirement Graph (ARG)
  Phase 3: File coverage evaluation
  Phase 4: Greedy file selection
  Phase 5: Completeness check
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sage import config
from sage.agents.base import BaseAgent
from sage.llm import call_llm


# ---------------------------------------------------------------------------
# Target dataclass
# ---------------------------------------------------------------------------

@dataclass
class Target:
    name: str
    required_columns: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Column-name mapping helpers
# ---------------------------------------------------------------------------

# Maps common predictor keywords to actual CSV column names in the mock data bank.
_PREDICTOR_COLUMN_MAP: dict[str, str] = {
    "cxcl13": "CXCL13",
    "fabp5": "FABP5",
    "lag3": "LAG3",
    "fn1": "FN1",
    "col1a1": "COL1A1",
    "egfr": "EGFR",
    "tp53": "TP53",
    "tls_density": "tls_density",
    "tls density": "tls_density",
    "tumor_immune_ratio": "tumor_immune_ratio",
    "tumor immune ratio": "tumor_immune_ratio",
    "collagen_alignment": "collagen_alignment",
    "collagen alignment": "collagen_alignment",
    "lag3_tumor_density": "lag3_tumor_density",
    "lag3 tumor density": "lag3_tumor_density",
}

_OUTCOME_COLUMN_MAP: dict[str, list[str]] = {
    "overall survival": ["os_months", "os_status"],
    "os": ["os_months", "os_status"],
    "progression-free survival": ["pfs_months", "pfs_status"],
    "progression free survival": ["pfs_months", "pfs_status"],
    "pfs": ["pfs_months", "pfs_status"],
    "treatment response": ["treatment_response"],
    "treatment_response": ["treatment_response"],
    "recurrence": ["pfs_months", "pfs_status"],
}

_COVARIATE_COLUMN_MAP: dict[str, str] = {
    "age": "age",
    "sex": "sex",
    "stage": "stage",
    "tumor stage": "stage",
    "treatment": "treatment",
    "gender": "sex",
}


def _resolve_predictor(name: str) -> str | None:
    """Map a predictor description to a CSV column name."""
    key = name.strip().lower()
    # Direct match
    if key in _PREDICTOR_COLUMN_MAP:
        return _PREDICTOR_COLUMN_MAP[key]
    # Check if any known token is a substring (e.g. "CXCL13 expression" → CXCL13)
    for token, col in _PREDICTOR_COLUMN_MAP.items():
        if token in key:
            return col
    return None


def _resolve_outcome(name: str) -> list[str]:
    """Map an outcome description to required CSV column names."""
    key = name.strip().lower()
    for token, cols in _OUTCOME_COLUMN_MAP.items():
        if token in key:
            return cols
    # Default to OS
    return ["os_months", "os_status"]


def _resolve_covariates(names: list[str]) -> list[str]:
    """Map covariate descriptions to CSV column names."""
    cols: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key in _COVARIATE_COLUMN_MAP:
            cols.append(_COVARIATE_COLUMN_MAP[key])
        else:
            # Try substring matching
            for token, col in _COVARIATE_COLUMN_MAP.items():
                if token in key:
                    cols.append(col)
                    break
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Phase 1: Parse hypothesis into constraints
# ---------------------------------------------------------------------------

def _parse_constraints_llm(hyp: dict[str, Any]) -> dict[str, Any]:
    """Use LLM to extract structured constraints from an expanded hypothesis."""
    system_prompt = (
        "You are a data requirements analyst. Extract structured constraints from a "
        "biomedical hypothesis. Return strict JSON only in this schema: "
        "{\"target_cohort\": str, \"predictors\": [str], \"outcome\": str, "
        "\"covariates\": [str], \"analysis_type\": str}. "
        "predictors should be individual variable names (e.g. 'CXCL13 expression', 'TLS density'). "
        "covariates should be individual clinical variables (e.g. 'age', 'sex', 'stage')."
    )
    user_prompt = (
        f"Hypothesis: {hyp.get('hypothesis_statement', '')}\n"
        f"Population: {hyp.get('population', '')}\n"
        f"Variables: {hyp.get('variables', {})}\n"
        f"Outcome: {hyp.get('outcome', '')}\n"
        f"Validation strategy: {hyp.get('validation_strategy', '')}"
    )
    raw = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, response_format="json")
    if not isinstance(raw, dict):
        raise ValueError("LLM did not return a dict")
    return {
        "target_cohort": str(raw.get("target_cohort", "")),
        "predictors": raw.get("predictors", []) if isinstance(raw.get("predictors"), list) else [],
        "outcome": str(raw.get("outcome", "")),
        "covariates": raw.get("covariates", []) if isinstance(raw.get("covariates"), list) else [],
        "analysis_type": str(raw.get("analysis_type", "survival analysis")),
    }


def _parse_constraints_fallback(hyp: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback: extract constraints directly from structured hypothesis fields."""
    variables = hyp.get("variables", {})
    if not isinstance(variables, dict):
        variables = {}

    predictors: list[str] = []
    for key in ("biological", "imaging"):
        val = str(variables.get(key, ""))
        if val:
            # Split on commas or " and "
            parts = re.split(r",| and ", val)
            predictors.extend(p.strip() for p in parts if p.strip())

    covariates: list[str] = []
    clinical = str(variables.get("clinical", ""))
    if clinical:
        parts = re.split(r",| and ", clinical)
        covariates.extend(p.strip() for p in parts if p.strip())

    return {
        "target_cohort": str(hyp.get("population", "bladder cancer cohort")),
        "predictors": predictors,
        "outcome": str(hyp.get("outcome", "overall survival")),
        "covariates": covariates,
        "analysis_type": "survival analysis",
    }


# ---------------------------------------------------------------------------
# Phase 2: Build ARG
# ---------------------------------------------------------------------------

def _build_arg(constraints: dict[str, Any]) -> list[Target]:
    """Build Artifact Requirement Graph from parsed constraints."""
    targets: list[Target] = []

    # τ_1: Cohort definition — always needs patient_id; add stage if cohort mentions it
    cohort_cols = ["patient_id"]
    cohort_text = constraints.get("target_cohort", "").lower()
    if any(kw in cohort_text for kw in ("stage", "ii", "iii", "iv", "mibc", "nmibc")):
        cohort_cols.append("stage")
    targets.append(Target(name="cohort_definition", required_columns=cohort_cols, depends_on=[]))

    # τ_2: Predictor measurements — one target per predictor
    for pred_name in constraints.get("predictors", []):
        col = _resolve_predictor(str(pred_name))
        if col:
            targets.append(
                Target(
                    name=f"predictor_{col}",
                    required_columns=["patient_id", col],
                    depends_on=["cohort_definition"],
                )
            )

    # τ_3: Outcome measurement
    outcome_cols = _resolve_outcome(str(constraints.get("outcome", "overall survival")))
    targets.append(
        Target(
            name="outcome_measurement",
            required_columns=["patient_id"] + outcome_cols,
            depends_on=["cohort_definition"],
        )
    )

    # τ_4: Covariates
    cov_cols = _resolve_covariates([str(c) for c in constraints.get("covariates", [])])
    if cov_cols:
        targets.append(
            Target(
                name="covariates",
                required_columns=["patient_id"] + cov_cols,
                depends_on=["cohort_definition"],
            )
        )

    return targets


# ---------------------------------------------------------------------------
# Phase 3: File coverage evaluation
# ---------------------------------------------------------------------------

def evaluate_coverage(file_path: str, target: Target) -> str:
    """Return 'none' | 'partial' | 'full'."""
    try:
        df = pd.read_csv(file_path, nrows=0)
    except Exception:
        return "none"
    columns = set(df.columns)
    required = set(target.required_columns)
    if required.issubset(columns):
        return "full"
    elif required & columns:
        return "partial"
    else:
        return "none"


def _build_coverage_matrix(files: list[str], targets: list[Target]) -> dict[str, dict[str, str]]:
    matrix: dict[str, dict[str, str]] = {}
    for f in files:
        basename = os.path.basename(f)
        matrix[basename] = {}
        for t in targets:
            matrix[basename][t.name] = evaluate_coverage(f, t)
    return matrix


# ---------------------------------------------------------------------------
# Phase 4: Greedy selection
# ---------------------------------------------------------------------------

def _greedy_select(
    files: list[str], targets: list[Target]
) -> tuple[list[str], list[dict[str, Any]], set[str]]:
    """Return (selected_files, selection_log, remaining_unmet)."""
    selected: list[str] = []
    unmet = {t.name for t in targets}
    selection_log: list[dict[str, Any]] = []
    step = 0

    while unmet:
        best_file: str | None = None
        best_gain = 0

        for f in files:
            if f in selected:
                continue
            gain = sum(
                1 for t in targets
                if t.name in unmet and evaluate_coverage(f, t) == "full"
            )
            if gain > best_gain:
                best_gain = gain
                best_file = f

        if best_gain == 0:
            break

        assert best_file is not None
        step += 1
        selected.append(best_file)
        selection_log.append({
            "step": step,
            "file": os.path.basename(best_file),
            "marginal_gain": best_gain,
        })

        # Update unmet
        for t in targets:
            if t.name in unmet and evaluate_coverage(best_file, t) == "full":
                unmet.discard(t.name)

    return selected, selection_log, unmet


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class DatasetDiscoveryAgent(BaseAgent):
    name = "dataset_discovery"

    def _discover_files(self) -> list[str]:
        """Scan DATA_BANK_DIR for CSV files."""
        bank_dir = config.DATA_BANK_DIR
        if not os.path.isdir(bank_dir):
            # Try relative to sage/ package directory
            alt = os.path.join(os.path.dirname(os.path.dirname(__file__)), bank_dir)
            if os.path.isdir(alt):
                bank_dir = alt
            else:
                return []
        files = sorted(
            os.path.join(bank_dir, f)
            for f in os.listdir(bank_dir)
            if f.endswith(".csv")
        )
        return files

    def run(self) -> dict[str, Any]:
        context = self.read_context()
        hypothesis = context.get("hypothesis_expansion")
        if not isinstance(hypothesis, dict):
            raise ValueError("DatasetDiscoveryAgent requires 'hypothesis_expansion' in context.")

        expanded_hypotheses = hypothesis.get("expanded_hypotheses")
        if not isinstance(expanded_hypotheses, list) or not expanded_hypotheses:
            raise ValueError("DatasetDiscoveryAgent requires non-empty expanded_hypotheses.")

        # Use the first hypothesis for dataset discovery
        hyp = expanded_hypotheses[0]

        # Phase 1: Parse constraints
        try:
            constraints = _parse_constraints_llm(hyp)
        except Exception:
            constraints = _parse_constraints_fallback(hyp)

        # Phase 2: Build ARG
        targets = _build_arg(constraints)

        # Phase 3: File coverage evaluation
        files = self._discover_files()
        coverage_matrix = _build_coverage_matrix(files, targets)

        # Phase 4: Greedy selection
        selected_files, selection_log, unmet = _greedy_select(files, targets)

        # Phase 5: Completeness check
        complete = len(unmet) == 0

        output: dict[str, Any] = {
            "discovery_result": {
                "constraints": constraints,
                "arg": {
                    "targets": [
                        {
                            "name": t.name,
                            "required_columns": t.required_columns,
                            "depends_on": t.depends_on,
                        }
                        for t in targets
                    ],
                },
                "coverage_matrix": coverage_matrix,
                "selected_files": selected_files,
                "selection_log": selection_log,
                "complete": complete,
                "missing_targets": sorted(unmet),
            }
        }

        self.write_output(output)
        return output
