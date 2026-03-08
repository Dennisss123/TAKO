# TAKO

**Topology-Adjusted KnockOut (TAKO): interpretable in silico single-gene knockout ranking on wild-type single-cell gene regulatory networks**

This repository provides the reference Python implementation of the TAKO algorithm described in our manuscript.

TAKO estimates knockout (KO) impact by comparing stationary propagation profiles on

1. the wild-type (WT) transition graph, and  
2. a topology-edited KO graph under strict **no-in-out** semantics.

The framework provides an interpretable and computationally efficient method for **WT-only single-gene KO impact ranking** on single-cell gene regulatory networks.

---

## Method summary

Given a WT single-cell expression matrix `X` (cells × genes):

1. Construct a directed WT gene regulatory graph using **PCR-based inference** with top-`p%` edge masking.
2. Convert the graph to a **row-stochastic transition matrix** `P`.
3. For a KO gene `g`, apply a strict topology intervention:
   - remove inflow to `g` (set column `g` to zero),
   - remove outflow from `g` (set row `g` to zero and assign a self-loop),
   - re-normalize rows with fallback self-loops.
4. Run personalised propagation on both WT and KO graphs using the same restart distribution.
5. Define KO impact as the stationary difference:

   `Δs_raw = s_WT - s_KO`

Optional derived scores include:

- positive component: `Δs_pos`
- absolute magnitude: `Δs_abs`

Genes ranked by these scores represent predicted **KO-responsive targets**.

---

## Reproducibility

The code in this repository corresponds to the version used in the manuscript:

```text
Release: v1.0.0
Commit: 277d1cf
```

This public release provides the core TAKO implementation.

The manuscript-reported analyses use the fixed settings described below, and the datasets included in the manuscript are listed in:

- `DATASETS.md`

---

## Repository structure

```text
TAKO/
├── tako/
│   ├── __init__.py
│   ├── TAKO.py
│   ├── core.py
│   └── grn.py
├── .gitignore
├── DATASETS.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python >= 3.9
```

---

## Quickstart

Create a toy expression matrix:

```bash
python - << 'PY'
import numpy as np

np.random.seed(0)
X = np.random.poisson(1.0, size=(200, 50)).astype(float)
np.save("toy.npy", X)
PY
```

Run TAKO on the toy dataset:

```bash
python -m tako.TAKO --x toy.npy --ko-index 3 --out tako_out.npz
```

The output file `tako_out.npz` contains the WT stationary vector, KO stationary vector, and KO impact scores.

To inspect available command-line arguments:

```bash
python -m tako.TAKO --help
```

---

## Implementation details

TAKO uses:

- PCR-based graph construction
- top-edge sparsification
- row-normalized transition matrices
- personalised PageRank propagation

Default parameters used in the manuscript:

```text
H = 3000
D = 50
p = 0.15
lambda = 0.05
alpha = 0.5
iters = 200
tol = 1e-8
```

The manuscript-reported configuration uses:

- strict **no-in-out** KO semantics
- **uniform** restart distribution over non-KO genes
- **raw** differential score (`delta_raw`) for KO ranking

For real datasets, pathway-level interpretation is based on preranked GSEA using the TAKO `delta_raw` ranking.

---

## Manuscript dataset scope

The submitted manuscript reports:

### Synthetic datasets
- SERGIO 100-gene GRN
- SERGIO 400-gene GRN
- SERGIO 1,200-gene GRN

### Real datasets
- Microglia / Trem2
- AT1 epithelial cells / Nkx2-1
- Human airway epithelial cells / STAT1
- Neurons / Mecp2

Further dataset details and accession information are provided in `DATASETS.md`.

---

## License

This project is released under the **MIT License**.

See the `LICENSE` file for details.
