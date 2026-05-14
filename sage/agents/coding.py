"""Coding Agent (Spec 9A).

Three-stage execution:
  Stage 1: File Inspection — read selected CSVs, extract schema info
  Stage 2: Code Generation — LLM generates self-contained Python script
  Stage 3: Sandboxed Execution + Repair Loop — subprocess run with retry
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import Any

import pandas as pd

from sage import config
from sage.agents.base import BaseAgent
from sage.llm import call_llm


def _inspect_files(file_paths: list[str]) -> str:
    """Stage 1: Inspect selected CSV files and return a text summary."""
    sections: list[str] = []
    for fpath in file_paths:
        if not os.path.isfile(fpath):
            sections.append(f"File: {fpath}\n  NOT FOUND\n")
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            sections.append(f"File: {fpath}\n  READ ERROR: {exc}\n")
            continue

        lines = [
            f"File: {fpath}",
            f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"  Columns: {list(df.columns)}",
            f"  Dtypes:\n{textwrap.indent(df.dtypes.to_string(), '    ')}",
            f"  Missing values:\n{textwrap.indent(df.isnull().sum().to_string(), '    ')}",
            f"  First 5 rows:\n{textwrap.indent(df.head().to_string(), '    ')}",
        ]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _execute_code(code: str, timeout: int = 30) -> tuple[bool, str, str]:
    """Execute code in a subprocess. Returns (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as exc:
        return False, "", str(exc)


_FALLBACK_CODE_TEMPLATE = textwrap.dedent("""\
    import json
    import pandas as pd
    import numpy as np
    from scipy import stats

    # Load data
    clinical = pd.read_csv("{clinical_path}")
    gene_expr = pd.read_csv("{gene_path}")
    imaging = pd.read_csv("{imaging_path}")

    # Merge datasets on patient_id
    df = clinical.merge(gene_expr, on="patient_id").merge(imaging, on="patient_id")

    results = {{"analysis": "survival_association", "tests": []}}

    # Test 1: CXCL13 high vs low overall survival (if columns exist)
    if "CXCL13" in df.columns and "os_months" in df.columns:
        threshold = df["CXCL13"].quantile(0.75)
        high = df[df["CXCL13"] > threshold]["os_months"]
        low = df[df["CXCL13"] <= threshold]["os_months"]
        stat, p_val = stats.mannwhitneyu(high, low, alternative="two-sided")
        results["tests"].append({{
            "name": "CXCL13_high_vs_low_OS",
            "high_mean": round(float(high.mean()), 2),
            "low_mean": round(float(low.mean()), 2),
            "U_statistic": round(float(stat), 2),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
        }})

    # Test 2: TLS density correlation with OS
    if "tls_density" in df.columns and "os_months" in df.columns:
        r, p_val = stats.spearmanr(df["tls_density"], df["os_months"])
        results["tests"].append({{
            "name": "tls_density_OS_correlation",
            "spearman_r": round(float(r), 4),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
        }})

    # Test 3: FABP5 high + TLS low survival
    if "FABP5" in df.columns and "tls_density" in df.columns and "os_months" in df.columns:
        fabp5_high = df["FABP5"] > df["FABP5"].quantile(0.75)
        tls_low = df["tls_density"] < df["tls_density"].quantile(0.25)
        poor = df[fabp5_high & tls_low]["os_months"]
        rest = df[~(fabp5_high & tls_low)]["os_months"]
        if len(poor) > 2 and len(rest) > 2:
            stat, p_val = stats.mannwhitneyu(poor, rest, alternative="two-sided")
            results["tests"].append({{
                "name": "FABP5_high_TLS_low_vs_rest_OS",
                "poor_mean": round(float(poor.mean()), 2),
                "rest_mean": round(float(rest.mean()), 2),
                "U_statistic": round(float(stat), 2),
                "p_value": round(float(p_val), 4),
                "significant": bool(p_val < 0.05),
            }})

    results["n_patients"] = len(df)
    results["columns_used"] = list(df.columns)
    print(json.dumps(results, indent=2))
""")


class CodingAgent(BaseAgent):
    name = "coding"

    def _generate_code(
        self,
        hyp: dict[str, Any],
        file_info: str,
        selected_files: list[str],
    ) -> str:
        """Stage 2: Generate analysis code via LLM."""
        file_paths_str = "\n".join(f"  - {f}" for f in selected_files)

        system_prompt = (
            "You are a biostatistics coding agent. Generate a self-contained Python script that:\n"
            "1. Imports all dependencies (pandas, numpy, scipy.stats, json). "
            "Do NOT import lifelines unless absolutely necessary.\n"
            "2. Reads CSV files from the exact paths provided\n"
            "3. Performs statistical analysis relevant to the hypothesis\n"
            "4. Prints results as a single JSON object to stdout using print(json.dumps(...))\n\n"
            "The script must be completely self-contained and runnable with `python -c`.\n"
            "Do NOT use markdown formatting. Output ONLY the Python code, nothing else.\n"
            "Do NOT use f-strings with nested braces that could break. Keep the code simple."
        )
        user_prompt = (
            f"Hypothesis:\n"
            f"  Statement: {hyp.get('hypothesis_statement', '')}\n"
            f"  Variables: {hyp.get('variables', {})}\n"
            f"  Outcome: {hyp.get('outcome', '')}\n"
            f"  Expected directionality: {hyp.get('expected_directionality', '')}\n"
            f"  Validation strategy: {hyp.get('validation_strategy', '')}\n\n"
            f"Available files:\n{file_paths_str}\n\n"
            f"File inspection results:\n{file_info}\n\n"
            f"Generate a Python script that tests this hypothesis using appropriate "
            f"statistical methods (e.g., Mann-Whitney U, Spearman correlation, "
            f"stratified survival comparison). Print results as JSON."
        )

        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.STRONG_MODEL,
            response_format="text",
        )
        code = str(raw).strip()
        # Strip markdown code fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line (```python) and last line (```)
            start = 1
            end = len(lines)
            if lines[-1].strip() == "```":
                end = -1
            code = "\n".join(lines[start:end])
        return code

    def _repair(
        self,
        code: str,
        stderr: str,
        file_info: str,
        hyp: dict[str, Any],
    ) -> str:
        """Repair failed code by sending the error back to LLM."""
        system_prompt = (
            "You are a code repair agent. The previous Python script failed. "
            "Fix the error and return ONLY the corrected Python code. "
            "No markdown formatting, no explanation, just the code.\n"
            "The code must be self-contained and print results as JSON to stdout.\n"
            "Do NOT import lifelines unless the error specifically requires it.\n"
            "Prefer scipy.stats for statistical tests."
        )
        user_prompt = (
            f"Previous code:\n{code}\n\n"
            f"Error:\n{stderr}\n\n"
            f"File schema info:\n{file_info}\n\n"
            f"Hypothesis variables: {hyp.get('variables', {})}\n"
            f"Hypothesis outcome: {hyp.get('outcome', '')}\n\n"
            f"Fix the code so it runs successfully."
        )

        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.STRONG_MODEL,
            response_format="text",
        )
        code = str(raw).strip()
        if code.startswith("```"):
            lines = code.split("\n")
            start = 1
            end = len(lines)
            if lines[-1].strip() == "```":
                end = -1
            code = "\n".join(lines[start:end])
        return code

    def _get_fallback_code(self, selected_files: list[str]) -> str:
        """Generate deterministic fallback code when LLM is unavailable."""
        clinical_path = ""
        gene_path = ""
        imaging_path = ""
        for f in selected_files:
            base = os.path.basename(f)
            if "clinical" in base:
                clinical_path = f.replace("\\", "/")
            elif "gene" in base:
                gene_path = f.replace("\\", "/")
            elif "imaging" in base:
                imaging_path = f.replace("\\", "/")

        return _FALLBACK_CODE_TEMPLATE.format(
            clinical_path=clinical_path,
            gene_path=gene_path,
            imaging_path=imaging_path,
        )

    def run(self) -> dict[str, Any]:
        context = self.read_context()

        hypothesis_data = context.get("hypothesis_expansion")
        discovery_data = context.get("dataset_discovery")

        if not isinstance(hypothesis_data, dict):
            raise ValueError("CodingAgent requires 'hypothesis_expansion' in context.")
        if not isinstance(discovery_data, dict):
            raise ValueError("CodingAgent requires 'dataset_discovery' in context.")

        expanded = hypothesis_data.get("expanded_hypotheses", [])
        if not expanded:
            raise ValueError("CodingAgent requires non-empty expanded_hypotheses.")
        hyp = expanded[0]

        discovery_result = discovery_data.get("discovery_result", {})
        selected_files = discovery_result.get("selected_files", [])

        # Stage 1: File Inspection
        file_info = _inspect_files(selected_files)

        # Stage 2: Code Generation
        try:
            code = self._generate_code(hyp, file_info, selected_files)
        except Exception:
            code = self._get_fallback_code(selected_files)

        # Stage 3: Sandboxed Execution + Repair Loop
        repair_attempts = 0
        success = False
        stdout = ""
        stderr = ""

        for attempt in range(config.MAX_REPAIR_ATTEMPTS + 1):
            success, stdout, stderr = _execute_code(code, timeout=30)
            if success:
                break
            if attempt < config.MAX_REPAIR_ATTEMPTS:
                repair_attempts += 1
                try:
                    code = self._repair(code, stderr, file_info, hyp)
                except Exception:
                    # LLM unavailable for repair, try fallback code
                    code = self._get_fallback_code(selected_files)

        output: dict[str, Any] = {
            "coding_result": {
                "generated_code": code,
                "execution_success": success,
                "stdout": stdout,
                "stderr": stderr,
                "repair_attempts": repair_attempts,
                "statistical_results": stdout if success else "",
            }
        }

        self.write_output(output)
        return output
