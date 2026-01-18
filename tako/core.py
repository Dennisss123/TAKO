from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp


MatrixLike = Union[np.ndarray, sp.spmatrix]


@dataclass(frozen=True)
class PPRConfig:
    alpha: float = 0.5
    tol: float = 1e-8
    max_iter: int = 200


def uniform_restart_non_g(G: int, g: int) -> np.ndarray:
    v = np.full(G, 1.0 / (G - 1), dtype=np.float64)
    v[g] = 0.0
    return v


def ppr_fixed_point(P: MatrixLike, v: np.ndarray, alpha: float, tol: float, max_iter: int) -> np.ndarray:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    Pm = P.tocsr().astype(np.float64) if sp.issparse(P) else np.asarray(P, dtype=np.float64)
    s = v.astype(np.float64, copy=True)
    for _ in range(max_iter):
        sp_term = (s @ Pm) if not sp.issparse(Pm) else (s @ Pm)
        s_next = (1.0 - alpha) * v + alpha * sp_term
        if np.sum(np.abs(s_next - s)) <= tol:
            s = s_next
            break
        s = s_next
    z = s.sum()
    if z > 0:
        s = s / z
    return s


def apply_no_in_out(P: MatrixLike, g: int) -> sp.csr_matrix:
    P_csr = P.tocsr().astype(np.float64) if sp.issparse(P) else sp.csr_matrix(P, dtype=np.float64)
    G = P_csr.shape[0]
    if not (0 <= g < G):
        raise IndexError("g out of range")

    P_ko = P_csr.tolil(copy=True)
    P_ko[:, g] = 0.0
    P_ko[g, :] = 0.0
    P_ko[g, g] = 1.0
    P_ko = P_ko.tocsr()

    rs = np.asarray(P_ko.sum(axis=1)).ravel()
    inv = np.zeros_like(rs, dtype=np.float64)
    nz = rs > 0
    inv[nz] = 1.0 / rs[nz]
    P_ko = sp.diags(inv) @ P_ko

    zero_rows = ~nz
    if np.any(zero_rows):
        P_ko = P_ko.tolil()
        for i in np.where(zero_rows)[0]:
            P_ko.rows[i] = [i]
            P_ko.data[i] = [1.0]
        P_ko = P_ko.tocsr()

    return P_ko


def tako_ko_profile(
    P: MatrixLike,
    g: int,
    cfg: PPRConfig = PPRConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    G = P.shape[0]
    v = uniform_restart_non_g(G, g)

    s_wt = ppr_fixed_point(P, v, cfg.alpha, cfg.tol, cfg.max_iter)
    P_ko = apply_no_in_out(P, g)
    s_ko = ppr_fixed_point(P_ko, v, cfg.alpha, cfg.tol, cfg.max_iter)

    delta_raw = s_wt - s_ko
    delta_pos = np.maximum(delta_raw, 0.0)
    delta_abs = np.abs(delta_raw)
    return s_wt, s_ko, delta_raw, delta_pos, delta_abs


def rank_targets(scores: np.ndarray, exclude_index: Optional[int] = None, descending: bool = True) -> np.ndarray:
    x = scores.copy()
    if exclude_index is not None and 0 <= exclude_index < x.size:
        x[exclude_index] = -np.inf if descending else np.inf
    order = np.argsort(x)
    return order[::-1] if descending else order
