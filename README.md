# TAKO

**Topology-Adjusted KnockOut (TAKO): interpretable in silico single-gene knockout ranking on wild-type single-cell gene regulatory networks**

This repository provides the reference Python implementation of the core TAKO algorithm used in our manuscript.
TAKO estimates KO impact by comparing stationary propagation profiles on:
1) the WT transition graph and
2) a topology-edited KO graph under strict **no-in-out** semantics.

---

## Method summary

Given WT single-cell expression matrix \(X\) (cells × genes):

1. Build a directed WT graph by PCR-based inference and top-\(p\%\) masking.
2. Row-normalize to transition matrix \(P\).
3. For KO gene \(g\), apply strict topology edit:
   - remove inflow to \(g\) (zero column \(g\)),
   - remove outflow from \(g\) (zero row \(g\), then self-loop at \(g\)),
   - re-normalize rows with fallback self-loops.
4. Run personalized propagation on WT and KO graphs with the same restart vector.
5. Define KO impact as stationary shift: \\
   \(\Delta s_{\mathrm{raw}} = s^{\mathrm{WT}} - s^{(g)}\), with optional `pos` / `abs` variants.

---

## Repository structure

```text
WANG-HAO-main/
├── tako/
│   ├── __init__.py
│   ├── TAKO.py                 # single-KO CLI
│   ├── core.py                 # PPR + KO topology edit + ranking
│   └── grn.py                  # PCR graph construction
├── scripts/
│   └── run_sergio_benchmark.py # multi-KO benchmark runner
├── requirements.txt
├── LICENSE
└── README.md
