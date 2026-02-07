from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

try:
    from tako.grn import GraphConfig, build_transition_from_expression
    from tako.core import PPRConfig, tako_ko_profile, rank_targets
except ModuleNotFoundError:  # fallback for script-style run
    from grn import GraphConfig, build_transition_from_expression  # type: ignore
    from core import PPRConfig, tako_ko_profile, rank_targets  # type: ignore


def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if p.suffix == ".npy":
        X = np.load(p, allow_pickle=False)
    elif p.suffix == ".npz":
        X = sp.load_npz(p).toarray()
    else:
        raise ValueError("Unsupported input. Use .npy or sparse .npz")

    if X.ndim != 2:
        raise ValueError("Input must be 2D (cells x genes).")
    return np.asarray(X, dtype=np.float64)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TAKO core CLI (single KO on WT matrix)")
    ap.add_argument("--x", type=str, required=True, help="WT expression matrix (cells x genes), .npy or sparse .npz")
    ap.add_argument("--ko-index", type=int, required=True, help="KO gene index (0-based)")

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--iters", type=int, default=200)

    ap.add_argument("--pcr-d", type=int, default=50)
    ap.add_argument("--ridge", type=float, default=0.05)
    ap.add_argument("--top-p", type=float, default=0.15)
    ap.add_argument("--binarize", action="store_true")

    ap.add_argument("--restart", type=str, default="uniform", choices=["uniform", "onehot"])
    ap.add_argument("--rank-metric", type=str, default="raw", choices=["raw", "pos", "abs"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", type=str, default="tako_out.npz")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    X = _load_matrix(args.x)
    n_cells, n_genes = X.shape

    if not (0 <= args.ko_index < n_genes):
        raise IndexError(f"--ko-index={args.ko_index} out of range for {n_genes} genes.")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0.")
    if args.topk <= 0:
        raise ValueError("--topk must be > 0.")

    gcfg = GraphConfig(
        n_components=args.pcr_d,
        ridge_lambda=args.ridge,
        top_p=args.top_p,
        binarize=args.binarize,
        random_state=0,
    )
    P, A_masked = build_transition_from_expression(X, gcfg)

    pcfg = PPRConfig(alpha=args.alpha, tol=args.tol, max_iter=args.iters)
    s_wt, s_ko, delta_raw, delta_pos, delta_abs = tako_ko_profile(
        P, args.ko_index, pcfg, restart=args.restart
    )

    if args.rank_metric == "raw":
        scores = delta_raw
    elif args.rank_metric == "pos":
        scores = delta_pos
    else:
        scores = delta_abs

    order = rank_targets(scores, exclude_index=args.ko_index, descending=True)
    top_idx = order[: min(args.topk, order.size)]
    top_scores = scores[top_idx]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        ko_index=int(args.ko_index),
        shape_cells=int(n_cells),
        shape_genes=int(n_genes),
        alpha=float(args.alpha),
        tol=float(args.tol),
        iters=int(args.iters),
        pcr_d=int(args.pcr_d),
        ridge=float(args.ridge),
        top_p=float(args.top_p),
        binarize=int(bool(args.binarize)),
        restart=str(args.restart),
        rank_metric=str(args.rank_metric),
        top_idx=top_idx.astype(np.int64),
        top_scores=top_scores.astype(np.float64),
        s_wt=s_wt.astype(np.float64),
        s_ko=s_ko.astype(np.float64),
        delta_raw=delta_raw.astype(np.float64),
        delta_pos=delta_pos.astype(np.float64),
        delta_abs=delta_abs.astype(np.float64),
        A_masked=A_masked.astype(np.float64),
    )

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
