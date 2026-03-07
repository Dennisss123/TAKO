# Reproducibility

This repository contains the public TAKO implementation corresponding to the manuscript submission.

## Scope

The submitted manuscript reports results on:

### Synthetic SERGIO benchmarks
- 100-gene GRN
- 400-gene GRN
- 1,200-gene GRN

### Real single-gene perturbation datasets
- Microglia / Trem2
- AT1 epithelial cells / Nkx2-1
- Human airway epithelial cells / STAT1
- Neurons / Mecp2

The intestine Hnf4a/Smad4 analysis was explored during method development, but it is not part of the main manuscript scope.

---

## Core method used in the manuscript

TAKO is a training-free framework for WT-only single-gene knockout impact ranking on single-cell gene regulatory networks.

The manuscript-reported workflow is:

1. Start from wild-type single-cell expression data
2. Perform preprocessing and highly variable gene selection
3. Construct a PCR-based directed graph
4. Convert the graph to a row-stochastic transition matrix
5. Apply strict no-in-out topology editing for the KO gene
6. Compute WT and KO stationary propagation profiles
7. Rank genes by the raw stationary difference (`delta_raw`)
8. For real datasets, use the resulting ranking for preranked GSEA

---

## Fixed parameter settings used in the manuscript

Unless otherwise noted, the manuscript-reported analyses use the following settings:

- graph construction: PCR-based
- number of highly variable genes: H = 3000
- PCR latent dimension: D = 50
- top-edge retention: 15%
- ridge penalty: 0.05
- continuation probability: alpha = 0.5
- maximum iterations: 200
- convergence tolerance: 1e-8
- KO semantics: strict no-in-out
- restart distribution: uniform over non-KO genes
- KO impact ranking: raw differential score (`delta_raw`)

---

## Real-data interpretation

For the four real datasets reported in the manuscript, TAKO is run on wild-type cells only.

Pathway-level interpretation is based on preranked GSEA using the TAKO `delta_raw` score, with species-appropriate gene-set collections from:

- Gene Ontology
- KEGG
- Reactome

The manuscript reports enrichment using normalized enrichment score (NES) and FDR-adjusted p-values.

---

## Public repository content

This public repository provides:

- the core TAKO source code
- the fixed manuscript-level analysis settings
- dataset accession documentation
- release and version information corresponding to the manuscript submission

For the dataset list used in the manuscript, see:

- `reproducibility/DATASETS.md`

---

## Version corresponding to the manuscript

- Release: `v1.0.0`
- Commit: `277d1cf`
