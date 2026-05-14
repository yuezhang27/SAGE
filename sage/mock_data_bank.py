from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_data_bank(
    output_dir: str = "data/data_bank",
    n_patients: int = 100,
    seed: int = 42,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)

    patient_ids = np.array([f"P{i:03d}" for i in range(1, n_patients + 1)])
    age = rng.integers(45, 86, size=n_patients)
    sex = rng.choice(["M", "F"], size=n_patients, p=[0.72, 0.28])
    stage = rng.choice(["II", "III", "IV"], size=n_patients, p=[0.35, 0.4, 0.25])
    treatment = rng.choice(["chemo", "immunotherapy", "combined"], size=n_patients, p=[0.4, 0.25, 0.35])

    gene_expression = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "FABP5": rng.normal(0.0, 1.0, size=n_patients),
            "CXCL13": rng.normal(0.0, 1.0, size=n_patients),
            "LAG3": rng.normal(0.0, 1.0, size=n_patients),
            "FN1": rng.normal(0.0, 1.0, size=n_patients),
            "COL1A1": rng.normal(0.0, 1.0, size=n_patients),
            "EGFR": rng.normal(0.0, 1.0, size=n_patients),
            "TP53": rng.normal(0.0, 1.0, size=n_patients),
        }
    )

    imaging_features = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "tls_density": np.clip(rng.beta(2.0, 2.5, size=n_patients), 0.01, 0.99),
            "tumor_immune_ratio": np.clip(rng.beta(2.3, 2.0, size=n_patients), 0.01, 0.99),
            "collagen_alignment": np.clip(rng.beta(2.2, 2.2, size=n_patients), 0.01, 0.99),
            "lag3_tumor_density": np.clip(rng.beta(2.4, 2.0, size=n_patients), 0.01, 0.99),
        }
    )

    baseline_os = rng.normal(30.0, 8.0, size=n_patients)
    baseline_pfs = rng.normal(20.0, 6.0, size=n_patients)

    stage_adjustment_os = np.select(
        [stage == "II", stage == "III", stage == "IV"],
        [4.0, 0.0, -6.0],
        default=0.0,
    )
    stage_adjustment_pfs = np.select(
        [stage == "II", stage == "III", stage == "IV"],
        [3.0, 0.0, -4.0],
        default=0.0,
    )

    os_months = baseline_os + stage_adjustment_os
    pfs_months = baseline_pfs + stage_adjustment_pfs

    cxcl13_q75 = float(np.percentile(gene_expression["CXCL13"], 75))
    cxcl13_high = gene_expression["CXCL13"].to_numpy() > cxcl13_q75
    os_months[cxcl13_high] += 8.0
    pfs_months[cxcl13_high] += 4.5

    fabp5_q75 = float(np.percentile(gene_expression["FABP5"], 75))
    tls_q25 = float(np.percentile(imaging_features["tls_density"], 25))
    poor_group = (gene_expression["FABP5"].to_numpy() > fabp5_q75) & (
        imaging_features["tls_density"].to_numpy() < tls_q25
    )
    os_months[poor_group] -= 9.0
    pfs_months[poor_group] -= 4.0

    response_score = (
        0.8 * imaging_features["tls_density"].to_numpy()
        - 0.9 * imaging_features["lag3_tumor_density"].to_numpy()
        + rng.normal(0.0, 0.2, size=n_patients)
    )
    treatment_response = np.where(response_score > np.percentile(response_score, 55), "responder", "non_responder")
    responder_mask = treatment_response == "responder"
    imaging_features.loc[responder_mask, "lag3_tumor_density"] *= 0.7
    imaging_features.loc[~responder_mask, "lag3_tumor_density"] *= 1.12
    imaging_features["lag3_tumor_density"] = imaging_features["lag3_tumor_density"].clip(0.01, 0.99)

    os_months = np.clip(os_months, 2.5, None)
    pfs_months = np.clip(pfs_months, 1.5, None)
    pfs_months = np.minimum(pfs_months, os_months - rng.uniform(0.2, 2.5, size=n_patients))
    pfs_months = np.clip(pfs_months, 1.0, None)

    os_event_prob = np.clip(0.85 - (os_months / 70.0), 0.1, 0.9)
    pfs_event_prob = np.clip(0.9 - (pfs_months / 50.0), 0.2, 0.95)
    os_status = rng.binomial(1, os_event_prob, size=n_patients)
    pfs_status = rng.binomial(1, pfs_event_prob, size=n_patients)

    clinical_metadata = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": age,
            "sex": sex,
            "stage": stage,
            "treatment": treatment,
            "os_months": np.round(os_months, 2),
            "os_status": os_status,
            "pfs_months": np.round(pfs_months, 2),
            "pfs_status": pfs_status,
            "treatment_response": treatment_response,
        }
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clinical_path = output_path / "clinical_metadata.csv"
    gene_path = output_path / "gene_expression.csv"
    imaging_path = output_path / "imaging_features.csv"

    clinical_metadata.to_csv(clinical_path, index=False)
    gene_expression.to_csv(gene_path, index=False)
    imaging_features.to_csv(imaging_path, index=False)

    return {
        "clinical_metadata.csv": str(clinical_path),
        "gene_expression.csv": str(gene_path),
        "imaging_features.csv": str(imaging_path),
    }
