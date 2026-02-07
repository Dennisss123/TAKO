from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


@dataclass(frozen=True)
class GraphConfig:
    n_components: int = 50
    ridge_lambda: float = 0.05
    top_p: float = 0.15
    binarize: bool = False
    random_state: int = 0


def _row_normalize_nonneg(W: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize nonnegative sparse matrix; zero rows -> self-loop."""
    rs = np.asarray(W.sum(axis=1)).ravel()
    inv = np.zeros_like(rs, dtype=np.float64)
    nz = rs > 0
    inv[nz] = 1.0 / rs[nz]
    P = sp.diags(inv) @ W

    if np.any(~nz):
        P = P.tolil()
        for i in np.where(~nz)[0]:
            P.rows[i] = [i]
            P.data[i] = [1.0]
        P = P.tocsr()
    return P


def pcr_directed_interaction(
    X_cells_genes: np.ndarray,
    n_components: int,
    ridge_lambda: float,
    random_state: int = 0,
) -> np.ndarray:
    """Build directed interaction matrix A (genes x genes) from X (cells x genes)."""
    if X_cells_genes.ndim != 2:
        raise ValueError("X must be 2D (cells x genes).")
    if n_components <= 0:
        raise ValueError("n_components must be > 0.")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0.")

    Xg = X_cells_genes.T.astype(np.float64)  # genes x cells
    Xg = Xg - Xg.mean(axis=1, keepdims=True)

    D = min(int(n_components), min(Xg.shape[0], Xg.shape[1]))
    if D < 1:
        raise ValueError("Invalid D after clipping.")

    U, S, Vt = randomized_svd(Xg, n_components=D, random_state=random_state)

    Z = Vt.T * S.reshape(1, -1)  # cells x D
    Y = Xg.T                     # cells x genes

    M = Z.T @ Z + float(ridge_lambda) * np.eye(D, dtype=np.float64)
    RHS = Z.T @ Y
    B = np.linalg.solve(M, RHS)  # D x genes
    A = U @ B                    # genes x genes

    np.fill_diagonal(A, 0.0)
    return A


def sparsify_top_p(A: np.ndarray, top_p: float) -> np.ndarray:
    """Keep top-p off-diagonal entries by |A|."""
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0,1].")

    G = A.shape[0]
    absA = np.abs(A).astype(np.float64)
    absA[np.eye(G, dtype=bool)] = 0.0

    vals = absA[absA > 0]
    if vals.size == 0:
        return np.zeros_like(A, dtype=np.float64)

    thr = np.quantile(vals, 1.0 - float(top_p))
    mask = absA >= thr
    mask[np.eye(G, dtype=bool)] = False

    out = np.zeros_like(A, dtype=np.float64)
    out[mask] = A[mask]
    return out


def interaction_to_transition(A: np.ndarray, binarize: bool = False) -> sp.csr_matrix:
    W = np.abs(A).astype(np.float64)
    np.fill_diagonal(W, 0.0)
    if binarize:
        W = (W > 0).astype(np.float64)
    return _row_normalize_nonneg(sp.csr_matrix(W))


def build_transition_from_expression(X_cells_genes: np.ndarray, cfg: GraphConfig) -> Tuple[sp.csr_matrix, np.ndarray]:
    A = pcr_directed_interaction(
        X_cells_genes,
        n_components=cfg.n_components,
        ridge_lambda=cfg.ridge_lambda,
        random_state=cfg.random_state,
    )
    A_masked = sparsify_top_p(A, cfg.top_p)
    P = interaction_to_transition(A_masked, binarize=cfg.binarize)
    return P, A_masked
