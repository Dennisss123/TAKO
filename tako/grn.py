# -*- coding: utf-8 -*-
"""
tako/grn.py — Graph construction (PCR/SVD-Regression) and normalization utils.
"""
from __future__ import annotations
import re
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Name & String Utils ----------
_G_PAT = re.compile(r"^(?:G|GENE)?\s*(\d+)$", re.IGNORECASE)

def norm_gene_label(x) -> str:
    s = str(x).strip().upper()
    m = _G_PAT.match(s)
    if m: return f"G{int(m.group(1))}"
    return s

def norm_list(xs) -> list[str]:
    return [norm_gene_label(x) for x in xs]

def _looks_like_gene_names(names, sample: int = 200) -> float:
    if not names:
        return 0.0
    n = min(sample, len(names))
    idx = random.sample(range(len(names)), n)
    geneish = 0
    pat_gene = re.compile(r"^[A-Za-z0-9\-\.]{2,15}$")
    for i in idx:
        s = str(names[i])
        if ":" in s: continue
        parts = s.split("_")
        if len(parts) > 1 and any(ch.isdigit() for ch in parts[0]):
            continue
        if pat_gene.match(s):
            geneish += 1
    return geneish / float(n)

def row_normalize(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    W = np.where(np.isfinite(W), W, 0.0)
    W = np.maximum(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P = W / rs
    return P

# ---------- Normalization / HVG ----------

def _cp10k_log1p(X: np.ndarray, axis_cells: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="F")
    if axis_cells == 1:
        lib = X.sum(axis=0, keepdims=True)
        lib[lib == 0] = 1.0
        Xn = X / lib * 1e4
    else:
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        Xn = X / lib * 1e4
    return np.log1p(Xn)

def _select_hvg_by_var_after_log(X_log: np.ndarray, gene_names_full: list[str], hvg: int,
                                whitelist: list[str], axis_genes: int) -> tuple[np.ndarray, list[str]]:
    Xg = X_log if axis_genes == 0 else X_log.T
    var = Xg.var(axis=1)
    order = np.argsort(-var)

    keep = set(order[: min(hvg, len(order))].tolist())
    for g in whitelist:
        if g in gene_names_full:
            keep.add(gene_names_full.index(g))

    keep = sorted(list(keep))
    gene_names = [gene_names_full[i] for i in keep]
    X_keep = Xg[keep, :]
    return X_keep, gene_names

# ---------- Main Graph Builders ----------

def build_W_corr_from_counts_with_names(counts_csv: str, cutoff: float = 60.0,
                                        hvg: int = 3000, corr_dtype: str = "float32",
                                        whitelist: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(counts_csv, header=0, index_col=0)
    r, c = df.shape
    
    row_score = _looks_like_gene_names(list(df.index))
    col_score = _looks_like_gene_names(list(df.columns))
    rows_as_genes = (row_score > col_score) or (row_score == col_score and r <= c)

    if rows_as_genes:
        gene_names_full = norm_list(df.index.tolist())
        X_full = df.to_numpy(dtype=np.float64)
        axis_genes, axis_cells = 0, 1
    else:
        gene_names_full = norm_list(df.columns.tolist())
        X_full = df.to_numpy(dtype=np.float64)
        axis_genes, axis_cells = 1, 0

    whitelist = [norm_gene_label(s) for s in (whitelist or []) if str(s).strip()]

    X_log = _cp10k_log1p(X_full, axis_cells=axis_cells)
    Xg, gene_names = _select_hvg_by_var_after_log(X_log, gene_names_full, hvg, whitelist, axis_genes=axis_genes)
    
    X = Xg - Xg.mean(axis=1, keepdims=True)
    denom = np.sqrt((X * X).sum(axis=1, keepdims=True))
    denom[denom == 0] = 1.0
    Xn = X / denom

    C = (Xn @ Xn.T) / max(1, Xn.shape[1] - 1)
    np.fill_diagonal(C, 0.0)

    thr = np.percentile(np.abs(C).ravel(), cutoff)
    W = np.where(np.abs(C) >= thr, np.abs(C), 0.0).astype(corr_dtype, copy=False)
    W = row_normalize(W)
    return W, gene_names

def build_W_pcr_struct(counts_csv: str,
                       hvg: int = 3000,
                       whitelist: list[str] | None = None,
                       pc_d: int = 10,
                       top_pct: float = 8.0,
                       ridge: float = 0.05,
                       binarize: int = 0) -> tuple[np.ndarray, list[str]]:
    """
    Build Directed Graph using Principal Component Regression (SVD-based).
    Technique: Beta = (Z'Z + lambda*I)^-1 Z' X, where Z are PCs of expression.
    """
    df = pd.read_csv(counts_csv, header=0, index_col=0)
    r, c = df.shape
    
    row_score = _looks_like_gene_names(list(df.index))
    col_score = _looks_like_gene_names(list(df.columns))
    rows_as_genes = (row_score > col_score) or (row_score == col_score and r <= c)

    if rows_as_genes:
        gene_names_full = norm_list(df.index.tolist())
        X_full = df.to_numpy(dtype=np.float64)   # genes x cells
        axis_genes, axis_cells = 0, 1
    else:
        gene_names_full = norm_list(df.columns.tolist())
        X_full = df.to_numpy(dtype=np.float64)   # cells x genes
        axis_genes, axis_cells = 1, 0

    whitelist = [norm_gene_label(s) for s in (whitelist or []) if str(s).strip()]

    X_log = _cp10k_log1p(X_full, axis_cells=axis_cells)
    Xg, gene_names = _select_hvg_by_var_after_log(X_log, gene_names_full, hvg, whitelist, axis_genes=axis_genes)
    
    p = Xg.shape[0]
    if p < 10:
        warnings.warn(f"[PCR] Too few genes after HVG: p={p}", RuntimeWarning)

    # PCA/SVD on cells x genes
    X_cg = Xg.T  # cells x genes
    X_cg = X_cg - X_cg.mean(axis=0, keepdims=True)
    
    U, S, Vt = np.linalg.svd(X_cg, full_matrices=False)
    rnk = int(np.sum(S > 1e-12))
    if pc_d > rnk:
        warnings.warn(f"[PCR] pc_d={pc_d} > rank={rnk}; use {rnk}.", RuntimeWarning)
    d = min(int(pc_d), Vt.shape[0])
    
    Z = U[:, :d] * S[:d]

    # Ridge Regression
    ZTZ = Z.T @ Z
    ZTZ.flat[::ZTZ.shape[0] + 1] += ridge
    ZTZ_inv = np.linalg.inv(ZTZ)
    B = ZTZ_inv @ Z.T @ X_cg

    # Reconstruct Adjacency A = V' * B
    A = (Vt[:d, :].T @ B)
    np.fill_diagonal(A, 0.0)

    # Keep top-% edges
    absA = np.abs(A)
    thr = np.percentile(absA.ravel(), 100.0 - top_pct)
    mask = absA >= thr
    A_masked = A * mask

    W = np.abs(A_masked)
    if int(binarize) == 1:
        W = (W > 0).astype(np.float64)

    W = row_normalize(W).astype("float32", copy=False)
    return W, gene_names

def load_W(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npy":
        W = np.load(str(p))
    else:
        W = pd.read_csv(str(p), header=None).to_numpy(dtype=float)
    W = np.asarray(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    W[W < 0] = 0.0
    W = row_normalize(W)
    return W
