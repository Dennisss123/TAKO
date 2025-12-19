# -*- coding: utf-8 -*-
"""
tako/core.py — TAKO core algorithms (PPR, restart vectors, KO semantics).
"""
from __future__ import annotations
import re
import numpy as np

def _renorm_rows_inplace(P: np.ndarray):
    """Normalize rows of P to sum to 1 inplace."""
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P /= rs

def ppr_vector(P: np.ndarray, alpha: float, iters: int, tol: float, v: np.ndarray) -> np.ndarray:
    """
    Compute PageRank vector by power iteration.
    s = (1-alpha)*v + alpha*s*P
    """
    v = np.asarray(v, dtype=np.float64)
    v = v / (v.sum() if v.sum() > 0 else 1.0)
    s = v.copy()
    for _ in range(int(iters)):
        s_new = (1 - alpha) * v + alpha * (s @ P)
        if tol is not None and tol > 0:
            if np.linalg.norm(s_new - s, ord=1) < tol:
                s = s_new
                break
        s = s_new
    
    s = np.maximum(s, 0.0)
    if s.sum() > 0:
        s /= s.sum()
    return s

def make_restart_vector(mode: str, P: np.ndarray, gi: int) -> np.ndarray:
    """
    Construct the personalization vector 'v'.
    """
    n = P.shape[0]
    if mode == "neighbor":
        # Restart proportional to direct neighbors of gi
        v = P[gi, :].astype(np.float64).copy()
        v[gi] = 0.0
        if v.sum() <= 0:
            # Fallback if no neighbors
            v = np.ones(n, dtype=np.float64); v[gi] = 0.0
        v = v / v.sum()
        return v
    elif mode == "uniform":
        v = np.ones(n, dtype=np.float64); v[gi] = 0.0
        v = v / v.sum()
        return v
    elif mode == "onehot":
        v = np.zeros(n, dtype=np.float64); v[gi] = 1.0
        return v
    else:
        raise ValueError(f"Unknown restart mode: {mode}")

def ko_matrix_true(P: np.ndarray, gi: int, mode: str) -> np.ndarray:
    """
    Apply 'True Knockout' structural changes to the graph P.
    """
    Pk = np.array(P, copy=True, dtype=np.float64)
    n = Pk.shape[0]

    if mode == "no-in-out":
        # Remove all edges entering or leaving gi
        Pk[:, gi] = 0.0
        Pk[gi, :] = 0.0
        Pk[gi, gi] = 1.0 # Absorb at self to avoid dead end
        _renorm_rows_inplace(Pk)
        return Pk

    elif mode == "silence-out":
        # Only remove edges leaving gi (stop regulation downstream)
        Pk[gi, :] = 0.0
        Pk[gi, gi] = 1.0
        _renorm_rows_inplace(Pk)
        return Pk

    elif mode == "restart-row":
        # Redistribute gi's outgoing weight uniformly
        Pk[:, gi] = 0.0
        row = np.ones(n, dtype=np.float64)
        row[gi] = 0.0
        row /= row.sum()
        Pk[gi, :] = row
        _renorm_rows_inplace(Pk)
        return Pk

    else:
        raise ValueError(f"Unknown ko-mode: {mode}")

def parse_topk(s: str) -> list[int]:
    """Parse comma-separated top-k string e.g. '10,20'."""
    if s is None:
        return []
    out = []
    for x in re.split(r"[,\s]+", str(s).strip()):
        if not x:
            continue
        try:
            out.append(int(x))
        except Exception:
            pass
    out = [k for k in out if k > 0]
    return sorted(list(dict.fromkeys(out)))

def precision_recall_at_k(y: np.ndarray, score: np.ndarray, ks: list[int]) -> tuple[dict[int,float], dict[int,float]]:
    """Compute Precision and Recall at K."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    
    # Sort by score descending
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    
    pos = int(y.sum())
    P_at, R_at = {}, {}
    
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        kk = min(k, len(y_sorted))
        tp = int(y_sorted[:kk].sum())
        
        P_at[k] = tp / float(kk) if kk > 0 else float("nan")
        R_at[k] = tp / float(pos) if pos > 0 else float("nan")
        
    return P_at, R_at
