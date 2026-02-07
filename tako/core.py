from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class PPRConfig:
    alpha: float = 0.5
    tol: float = 1e-8
    max_iter: int = 200


def _renorm_rows_with_selfloop_fallback(P: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize; zero-sum rows become self-loops."""
    P = P.tocsr().astype(np.float64)
    rs = np.asarray(P.sum(axis=1)).ravel()
    inv = np.zeros_like(rs, dtype=np.float64)
    nz = rs > 0
    inv[nz] = 1.0 / rs[nz]
    P = sp.diags(inv) @ P

    if np.any(~nz):
        P = P.tolil()
        for i in np.where(~nz)[0]:
            P.rows[i] = [i]
            P.data[i] = [1.0]
        P = P.tocsr()
    return P


def apply_no_in_out_ko(P: sp.csr_matrix, ko_index: int) -> sp.csr_matrix:
    """
    Strict no-in-out KO:
    1) zero KO column (no inflow),
    2) zero KO row then set self-loop 1 (no outflow),
    3) row renormalize with zero-row self-loop fallback.
    """
    n = P.shape[0]
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be square.")
    if not (0 <= ko_index < n):
        raise IndexError("ko_index out of range.")

    Pk = P.tolil(copy=True)
    Pk[:, ko_index] = 0.0
    Pk[ko_index, :] = 0.0
    Pk[ko_index, ko_index] = 1.0
    return _renorm_rows_with_selfloop_fallback(Pk.tocsr())


def ppr_fixed_point(P: sp.csr_matrix, v: np.ndarray, cfg: PPRConfig) -> np.ndarray:
    """
    Solve s = (1-alpha)v + alpha sP by fixed-point iteration.
    s is a row vector.
    """
    if not sp.issparse(P):
        raise ValueError("P must be a scipy sparse matrix.")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be square.")

    n = P.shape[0]
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size != n:
        raise ValueError("v length mismatch.")
    vsum = v.sum()
    if vsum <= 0:
        raise ValueError("restart vector sums to 0.")
    v = v / vsum

    alpha = float(cfg.alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if cfg.max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if cfg.tol <= 0:
        raise ValueError("tol must be > 0.")

    s = v.copy()
    for _ in range(cfg.max_iter):
        s_new = (1.0 - alpha) * v + alpha * np.asarray(s @ P).ravel()
        if np.linalg.norm(s_new - s, ord=1) < cfg.tol:
            s = s_new
            break
        s = s_new
    return s


def make_restart_vector(n: int, ko_index: int, mode: str = "uniform") -> np.ndarray:
    if not (0 <= ko_index < n):
        raise IndexError("ko_index out of range.")

    if mode == "uniform":
        v = np.ones(n, dtype=np.float64)
        v[ko_index] = 0.0
        v /= v.sum()
        return v
    if mode == "onehot":
        v = np.zeros(n, dtype=np.float64)
        v[ko_index] = 1.0
        return v

    raise ValueError(f"Unsupported restart mode: {mode}")


def tako_ko_profile(
    P: sp.csr_matrix,
    ko_index: int,
    cfg: PPRConfig,
    restart: str = "uniform",
):
    """
    Returns:
      s_wt, s_ko, delta_raw, delta_pos, delta_abs
    """
    n = P.shape[0]
    v = make_restart_vector(n, ko_index, mode=restart)

    s_wt = ppr_fixed_point(P, v, cfg)
    P_ko = apply_no_in_out_ko(P, ko_index)
    s_ko = ppr_fixed_point(P_ko, v, cfg)

    delta_raw = s_wt - s_ko
    delta_pos = np.maximum(delta_raw, 0.0)
    delta_abs = np.abs(delta_raw)
    return s_wt, s_ko, delta_raw, delta_pos, delta_abs


def rank_targets(scores: np.ndarray, exclude_index: int | None = None, descending: bool = True) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel().copy()
    if exclude_index is not None and 0 <= exclude_index < s.size:
        s[exclude_index] = -np.inf if descending else np.inf
    order = np.argsort(-s if descending else s)
    if exclude_index is not None:
        order = order[order != exclude_index]
    return order
