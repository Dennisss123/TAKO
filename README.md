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

## Repository structure

```text
TAKO/
├── tako/
│   ├── __init__.py
│   ├── TAKO.py
│   ├── core.py
│   └── grn.py
├── requirements.txt
├── License
└── README.md
