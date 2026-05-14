from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from sage import config


def _add_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    relation_type: str,
    evidence_text: str,
    rng: np.random.Generator,
) -> None:
    confidence = float(np.round(rng.uniform(0.55, 0.98), 3))
    graph.add_edge(
        source,
        target,
        relation_type=relation_type,
        confidence=confidence,
        evidence_text=evidence_text,
    )


def build_mock_kg() -> nx.DiGraph:
    graph = nx.DiGraph()
    rng = np.random.default_rng(42)

    entity_groups: dict[str, list[str]] = {
        "Gene": [
            "FABP5",
            "CXCL13",
            "FN1",
            "COL1A1",
            "LAG3",
            "EGFR",
            "TP53",
            "MMP9",
            "VEGFA",
            "CD8A",
            "PDCD1",
            "FOXP3",
        ],
        "Pathway": [
            "ECM_remodeling",
            "TLS_formation",
            "immune_exhaustion",
            "PI3K_AKT",
            "angiogenesis",
            "apoptosis",
            "EMT",
            "DNA_repair",
        ],
        "Disease": ["bladder_cancer", "MIBC", "NMIBC", "urothelial_carcinoma"],
        "ClinicalEndpoint": ["overall_survival", "PFS", "treatment_response", "recurrence"],
        "TissueRegion": [
            "TME",
            "peritumoral_stroma",
            "TLS_region",
            "tumor_core",
            "invasive_front",
            "lymphoid_aggregate",
        ],
        "Biomarker": [
            "TLS_density",
            "tumor_immune_ratio",
            "collagen_alignment",
            "lag3_tumor_density",
            "cd8_infiltration",
            "ki67_index",
        ],
        "StainingMethod": ["HE", "IHC_LAG3", "IHC_CD8"],
        "Algorithm": [
            "cox_regression",
            "kaplan_meier",
            "spatial_analysis",
            "mann_whitney",
            "log_rank",
        ],
    }

    for entity_type, entities in entity_groups.items():
        for name in entities:
            graph.add_node(name, entity_type=entity_type)

    curated_edges: list[tuple[str, str, str, str]] = [
        ("CXCL13", "TLS_formation", "upregulates", "CXCL13 recruits B cells and supports TLS assembly."),
        ("TLS_formation", "TLS_region", "located_in", "TLS formation is observed in lymphoid-rich tumor niches."),
        ("TLS_region", "overall_survival", "predicts", "TLS-rich regions correlate with improved survival."),
        ("FABP5", "immune_exhaustion", "upregulates", "FABP5 signaling links lipid metabolism to immune dysfunction."),
        ("immune_exhaustion", "TME", "part_of", "Immune exhaustion is a TME-level immunologic state."),
        ("TME", "treatment_response", "associated_with", "Immune landscape in TME influences therapy response."),
        ("FN1", "ECM_remodeling", "upregulates", "FN1 promotes extracellular matrix remodeling programs."),
        ("ECM_remodeling", "invasive_front", "located_in", "ECM remodeling features localize at invasive fronts."),
        ("invasive_front", "recurrence", "predicts", "Invasive front biology predicts recurrence risk."),
        ("TP53", "DNA_repair", "associated_with", "TP53 status modulates DNA damage repair pathways."),
        ("DNA_repair", "urothelial_carcinoma", "part_of", "DNA repair dysfunction is central in urothelial cancer biology."),
        ("urothelial_carcinoma", "PFS", "associated_with", "Tumor subtype affects progression-free survival."),
        ("LAG3", "immune_exhaustion", "upregulates", "LAG3 is a canonical exhaustion-associated checkpoint marker."),
        ("LAG3", "lag3_tumor_density", "measured_by", "LAG3 burden can be quantified as tumor density."),
        ("lag3_tumor_density", "IHC_LAG3", "measured_by", "LAG3 density is assessed via LAG3 IHC staining."),
        ("CD8A", "cd8_infiltration", "measured_by", "CD8A maps to CD8+ infiltration burden in tissue."),
        ("cd8_infiltration", "IHC_CD8", "measured_by", "CD8 infiltration is measured by CD8 IHC staining."),
        ("TLS_density", "HE", "measured_by", "TLS density can be estimated from H&E histology."),
        ("TLS_density", "overall_survival", "predicts", "Higher TLS density tends to indicate better outcomes."),
        ("tumor_immune_ratio", "overall_survival", "predicts", "Tumor-immune composition ratio tracks prognosis."),
        ("collagen_alignment", "recurrence", "associated_with", "Aligned collagen fibers associate with invasion and relapse."),
        ("MMP9", "ECM_remodeling", "upregulates", "MMP9 drives extracellular matrix turnover."),
        ("VEGFA", "angiogenesis", "upregulates", "VEGFA is a key angiogenic driver."),
        ("angiogenesis", "tumor_core", "located_in", "Angiogenic signaling is prominent in hypoxic tumor core regions."),
        ("tumor_core", "PFS", "associated_with", "Core-region traits correlate with disease progression."),
        ("EGFR", "PI3K_AKT", "upregulates", "EGFR activation stimulates PI3K-AKT signaling."),
        ("PI3K_AKT", "apoptosis", "downregulates", "PI3K-AKT signaling suppresses apoptotic programs."),
        ("apoptosis", "overall_survival", "associated_with", "Apoptotic competence is linked to longer survival."),
        ("FOXP3", "immune_exhaustion", "associated_with", "FOXP3+ regulatory T cells shape suppressive immunity."),
        ("PDCD1", "immune_exhaustion", "associated_with", "PDCD1 expression marks exhausted T-cell states."),
        ("bladder_cancer", "MIBC", "part_of", "MIBC is a clinical subtype within bladder cancer."),
        ("bladder_cancer", "NMIBC", "part_of", "NMIBC is a clinical subtype within bladder cancer."),
        ("bladder_cancer", "urothelial_carcinoma", "associated_with", "Most bladder tumors are urothelial carcinomas."),
        ("MIBC", "overall_survival", "associated_with", "MIBC stage tends to have worse survival than NMIBC."),
        ("NMIBC", "recurrence", "associated_with", "NMIBC often shows recurrence despite lower progression."),
        ("MIBC", "PFS", "associated_with", "MIBC generally has shorter progression-free survival."),
        ("bladder_cancer", "overall_survival", "associated_with", "Disease severity strongly affects overall survival."),
        ("bladder_cancer", "PFS", "associated_with", "Disease biology drives progression timing."),
        ("bladder_cancer", "treatment_response", "associated_with", "Tumor subtype influences treatment sensitivity."),
        ("bladder_cancer", "recurrence", "associated_with", "Recurrence risk is a major clinical endpoint in bladder cancer."),
        ("TP53", "bladder_cancer", "associated_with", "TP53 is frequently altered in aggressive bladder tumors."),
        ("TP53", "MIBC", "associated_with", "TP53 alterations are enriched in muscle-invasive disease."),
        ("TP53", "EMT", "upregulates", "TP53 dysregulation can facilitate EMT-like transitions."),
        ("TP53", "apoptosis", "downregulates", "Mutant TP53 weakens apoptotic control."),
        ("TP53", "PI3K_AKT", "interacts_with", "TP53 status intersects with PI3K/AKT signaling outputs."),
        ("TP53", "angiogenesis", "associated_with", "TP53 dysfunction supports pro-angiogenic states."),
        ("TP53", "recurrence", "predicts", "TP53 aberrations are linked to higher relapse risk."),
        ("TP53", "overall_survival", "predicts", "TP53 aberration burden predicts poor overall survival."),
        ("TP53", "treatment_response", "associated_with", "TP53 status affects treatment susceptibility."),
        ("bladder_cancer", "TME", "located_in", "Bladder tumor progression is conditioned by microenvironment context."),
        ("bladder_cancer", "peritumoral_stroma", "located_in", "Peritumoral stroma contributes to bladder tumor behavior."),
        ("bladder_cancer", "tumor_core", "located_in", "Tumor core microanatomy influences downstream phenotypes."),
        ("bladder_cancer", "invasive_front", "located_in", "Invasive front represents active tumor-stroma interaction zone."),
        ("bladder_cancer", "TLS_region", "located_in", "TLS regions appear in subsets of bladder cancers."),
        ("bladder_cancer", "lymphoid_aggregate", "located_in", "Lymphoid aggregates may indicate anti-tumor immunity."),
        ("bladder_cancer", "TLS_density", "correlates_with", "TLS density varies by bladder cancer immune phenotype."),
        ("bladder_cancer", "tumor_immune_ratio", "correlates_with", "Tumor-immune ratio stratifies immune microenvironments."),
        ("bladder_cancer", "collagen_alignment", "correlates_with", "Collagen architecture shifts with bladder cancer progression."),
        ("bladder_cancer", "lag3_tumor_density", "correlates_with", "LAG3 density patterns differ across disease states."),
        ("bladder_cancer", "cd8_infiltration", "correlates_with", "CD8 infiltration is linked to anti-tumor immunity."),
        ("ECM_remodeling", "collagen_alignment", "correlates_with", "ECM remodeling changes collagen orientation."),
        ("TLS_formation", "TLS_density", "associated_with", "TLS biology is reflected in TLS density measurements."),
        ("immune_exhaustion", "lag3_tumor_density", "correlates_with", "Exhaustion states correlate with high LAG3 density."),
        ("immune_exhaustion", "CD8A", "interacts_with", "Exhaustion signatures overlap activated CD8 programs."),
        ("TME", "TLS_region", "part_of", "TLS regions are a structured subset of the tumor microenvironment."),
        ("TME", "peritumoral_stroma", "part_of", "Peritumoral stroma is a major TME compartment."),
        ("TME", "tumor_core", "part_of", "Tumor core forms the central compartment of TME."),
        ("TME", "invasive_front", "part_of", "Invasive front is a specialized edge compartment in TME."),
        ("TLS_region", "lymphoid_aggregate", "part_of", "Lymphoid aggregates compose TLS-rich regions."),
        ("cox_regression", "overall_survival", "used_for", "Cox models estimate hazard relationships with OS."),
        ("cox_regression", "PFS", "used_for", "Cox models can evaluate progression-free survival."),
        ("kaplan_meier", "overall_survival", "used_for", "Kaplan-Meier curves estimate OS distributions."),
        ("kaplan_meier", "PFS", "used_for", "Kaplan-Meier curves estimate PFS distributions."),
        ("log_rank", "overall_survival", "used_for", "Log-rank tests compare OS across groups."),
        ("log_rank", "PFS", "used_for", "Log-rank tests compare PFS across groups."),
        ("mann_whitney", "treatment_response", "used_for", "Mann-Whitney tests compare biomarker levels by response."),
        ("spatial_analysis", "TLS_density", "used_for", "Spatial analysis quantifies TLS organization."),
        ("spatial_analysis", "tumor_immune_ratio", "used_for", "Spatial analysis captures immune-tumor composition."),
        ("spatial_analysis", "collagen_alignment", "used_for", "Spatial analysis quantifies collagen orientation."),
        ("IHC_LAG3", "treatment_response", "predicts", "LAG3 IHC patterns may stratify treatment response."),
        ("IHC_CD8", "treatment_response", "predicts", "CD8 IHC burden can indicate immunotherapy sensitivity."),
        ("ki67_index", "recurrence", "predicts", "Higher Ki67 proliferative index links to recurrence risk."),
        ("ki67_index", "HE", "measured_by", "Ki67-like proliferative patterns can be approximated on H&E in mock setting."),
        ("HE", "TLS_region", "measured_by", "H&E allows pathologists to identify TLS regions."),
        ("HE", "peritumoral_stroma", "measured_by", "H&E captures stromal architecture for region annotation."),
        ("EMT", "invasive_front", "located_in", "EMT-like states are enriched near invasive fronts."),
        ("MMP9", "invasive_front", "located_in", "MMP9 activity localizes to invasion-associated regions."),
        ("VEGFA", "overall_survival", "associated_with", "Pro-angiogenic signaling often indicates worse prognosis."),
        ("EGFR", "treatment_response", "predicts", "EGFR pathway status modulates therapy benefit."),
    ]

    for edge in curated_edges:
        _add_edge(graph, edge[0], edge[1], edge[2], edge[3], rng)

    kg_path = Path(config.KG_PATH)
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, kg_path)
    return graph


def load_mock_kg() -> nx.DiGraph:
    kg_path = Path(config.KG_PATH)
    if not kg_path.exists():
        return build_mock_kg()
    return nx.read_graphml(kg_path)
