from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from tako.grn import GraphConfig, build_transition_from_expression
from tako.core import PPRConfig, tako_ko_profile, rank_targets


def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix == ".npz":
        return sp.load_npz(p).toarray()
    raise ValueError("Unsupported input. Use .npy (dense) or .npz (sparse).")


def main():
    ap = argparse.ArgumentParser("TAKO (core logic only)")
    ap.add_argument("--x", type=str, required=True, help="WT expression matrix (cells x genes), .npy or .npz")
    ap.add_argument("--ko-index", type=int, required=True, help="KO gene index g (0-based)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--iters", type=int, default=200)

    ap.add_argument("--pcr-d", type=int, default=50)
    ap.add_argument("--ridge", type=float, default=0.05)
    ap.add_argument("--top-p", type=float, default=0.15)
    ap.add_argument("--binarize", action="store_true")

    ap.add_argument("--rank-metric", type=str, default="raw", choices=["raw", "pos", "abs"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", type=str, default="tako_out.npz")

    args = ap.parse_args()

    X = _load_matrix(args.x).astype(np.float64)

    gcfg = GraphConfig(
        n_components=args.pcr_d,
        ridge_lambda=args.ridge,
        top_p=args.top_p,
        binarize=args.binarize,
        random_state=0,
    )
    P, A_masked = build_transition_from_expression(X, gcfg)

    pcfg = PPRConfig(alpha=args.alpha, tol=args.tol, max_iter=args.iters)
    s_wt, s_ko, delta_raw, delta_pos, delta_abs = tako_ko_profile(P, args.ko_index, pcfg)

    if args.rank_metric == "raw":
        scores = delta_raw
    elif args.rank_metric == "pos":
        scores = delta_pos
    else:
        scores = delta_abs

    order = rank_targets(scores, exclude_index=args.ko_index, descending=True)
    top_idx = order[: args.topk]
    top_scores = scores[top_idx]

    np.savez_compressed(
        args.out,
        ko_index=int(args.ko_index),
        alpha=float(args.alpha),
        tol=float(args.tol),
        iters=int(args.iters),
        pcr_d=int(args.pcr_d),
        ridge=float(args.ridge),
        top_p=float(args.top_p),
        rank_metric=str(args.rank_metric),
        top_idx=top_idx.astype(np.int64),
        top_scores=top_scores.astype(np.float64),
        s_wt=s_wt.astype(np.float64),
        s_ko=s_ko.astype(np.float64),
        delta_raw=delta_raw.astype(np.float64),
        delta_pos=delta_pos.astype(np.float64),
        delta_abs=delta_abs.astype(np.float64),
    )


if __name__ == "__main__":
    main()
