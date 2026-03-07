# Datasets used in the manuscript

This document lists the datasets included in the submitted TAKO manuscript.

## Synthetic datasets

The manuscript reports synthetic benchmarking on three SERGIO-generated directed gene regulatory networks:

- 100 genes
- 400 genes
- 1,200 genes

These simulated datasets are used for quantitative benchmark evaluation.

---

## Real datasets

The manuscript reports four real single-gene perturbation datasets.

### 1. Microglia / Trem2
- perturbation gene: Trem2
- accession: GSE130627

### 2. AT1 epithelial cells / Nkx2-1
- perturbation gene: Nkx2-1
- accession: GSE129628

### 3. Human airway epithelial cells / STAT1
- perturbation gene: STAT1
- accession: EGAS00001004481
- note: this dataset is available under controlled access via EGA

### 4. Neurons / Mecp2
- perturbation gene: Mecp2
- accession: SRP135960

---

## Not part of the main manuscript scope

The intestine Hnf4a/Smad4 analysis was explored during method development, but it is not part of the main manuscript-reported dataset set.

---

## Downstream interpretation in the manuscript

For the four real datasets listed above, TAKO scores are interpreted at the pathway level using preranked GSEA.

The manuscript uses species-appropriate gene-set collections from:

- Gene Ontology
- KEGG
- Reactome
