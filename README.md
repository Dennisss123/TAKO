# TAKO (Topology-Adjusted KnockOut)

This repository provides the reference implementation of **Topology-Adjusted KnockOut (TAKO)**, a computational framework for *in silico* single-gene perturbation screening using only wild-type (WT) single-cell gene regulatory networks (GRNs).

This codebase contains the core algorithmic modules described in the manuscript, designed to ensure reproducibility of the method:
**X → A → P → P^KO_g → (s^WT_g, s^KO_g) → Δs_g**.

Note that datasets are not included in this repository.

## Repository structure

- `tako/grn.py`
  Constructs a directed interaction matrix (`A`) from WT expression data using PCR/ridge regression. The matrix is then sparsified and normalized into a non-negative, row-stochastic transition matrix (`P`).

- `tako/core.py`
  Implements the **no-in/no-out** knockout strategy on `P`, computes Personalized PageRank (PPR) using fixed-point iteration with uniform restarts, and calculates the perturbation scores (`delta_raw`, `delta_pos`, `delta_abs`).

- `tako/TAKO.py`
  A command-line interface (CLI) for executing the workflow with user-defined parameters.

## Installation

```bash
pip install -r requirements.txt
