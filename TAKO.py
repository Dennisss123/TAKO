"""
TAKO.py — CLI entrypoint for the TAKO algorithm.
Usage: python TAKO.py --counts data.csv --ko-seq list.txt --outdir results/
"""

from __future__ import annotations
import os
import json
import argparse
import re
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# --- Imports from the clean TAKO package ---
from tako.grn import _G_PAT, norm_gene_label, build_W_corr_from_counts_with_names, build_W_pcr_struct, load_W, row_normalize
from tako.core import ppr_vector, make_restart_vector, ko_matrix_true, parse_topk, precision_recall_at_k

# ==========================================
#  Helper Functions (Hidden here to keep package clean)
# ==========================================

def _load_gt_aligned(gt_arg: str, gene_names: list[str], ko_seq_path: str = None) -> np.ndarray:
    """Loads Ground Truth and aligns it to gene_names order."""
    p = len(gene_names)
    name2idx = {g:i for i,g in enumerate(gene_names)}
    
    # Auto-match logic if directory provided
    final_path = gt_arg
    if Path(gt_arg).is_dir() and ko_seq_path:
        # Simple heuristic to find best matching file in dir
        with open(ko_seq_path, "r") as f: kos = [x.strip() for x in f if x.strip()]
        best_f, best_sc = None, -1
        for cand in list(Path(gt_arg).glob("*.csv")) + list(Path(gt_arg).glob("*.npy")):
            sc = sum(1 for k in kos if k.lower() in cand.name.lower())
            if sc > best_sc: best_f, best_sc = cand, sc
        if best_f: final_path = str(best_f)

    # Load
    path = Path(final_path)
    if path.suffix == ".npy":
        GT = np.load(path)
        GT = (GT > 0).astype(np.uint8)
        np.fill_diagonal(GT, 0)
        return GT
    else:
        # Assume CSV Edge List
        df = pd.read_csv(path, header=None)
        src = df.iloc[:,0].astype(str).map(norm_gene_label).tolist()
        tgt = df.iloc[:,1].astype(str).map(norm_gene_label).tolist()
        GT = np.zeros((p, p), dtype=np.uint8)
        for s, t in zip(src, tgt):
            i, j = name2idx.get(s), name2idx.get(t)
            if i is not None and j is not None and i != j: GT[i, j] = 1
        return GT

def _save_results(outdir: str, glab: str, gene_names: list[str], 
                  delta: np.ndarray, P: np.ndarray, topk: list[int]):
    """Minimal result saver."""
    df = pd.DataFrame({
        "gene": gene_names,
        "score": delta,
        "deg_out": (P > 0).sum(axis=1).astype(int)
    })
    df.sort_values("score", ascending=False, inplace=True)
    df["rank"] = np.arange(1, len(df)+1)
    
    base = Path(outdir) / f"kopanel_{glab}.csv"
    df.to_csv(base, index=False)
    for k in topk:
        df.head(int(k)).to_csv(Path(outdir) / f"kopanel_{glab}_top{k}.csv", index=False)

def run_evaluation(counts_path, gt_arg, ko_seq_path, outdir, args):
    try:
        print(f"\n==== RUN: {Path(counts_path).name} ====")
        os.makedirs(outdir, exist_ok=True)

        # 1. Load KO List
        with open(ko_seq_path, "r") as f:
            ks_all = [norm_gene_label(x) for x in f if x.strip()]
        
        # 2. Build Graph (The Core Model)
        print(f"[Model] Building Graph: {args.graph}...")
        wl = ks_all if not args.force_include else [norm_gene_label(x) for x in re.split(r"[,\s]+", args.force_include)]
        
        if args.W_path:
            W = load_W(args.W_path)
            _, gene_names = build_W_corr_from_counts_with_names(counts_path, hvg=args.hvg, whitelist=wl)
        elif args.graph == "pcr":
            W, gene_names = build_W_pcr_struct(counts_path, hvg=args.hvg, whitelist=wl,
                                             pc_d=args.pcr_pc_d, top_pct=args.pcr_top_pct, 
                                             ridge=args.pcr_ridge, binarize=args.pcr_binarize)
        else:
            W, gene_names = build_W_corr_from_counts_with_names(counts_path, cutoff=args.cutoff, hvg=args.hvg, whitelist=wl)

        p = len(gene_names)
        P = row_normalize(W)
        name2idx = {g:i for i,g in enumerate(gene_names)}
        print(f"[Graph] {p} genes, {np.count_nonzero(P)} edges.")

        GT = None
        if gt_arg:
            try:
                GT = _load_gt_aligned(gt_arg, gene_names, ko_seq_path)
                print(f"[GT] Loaded. Edges: {GT.sum()}")
            except Exception as e:
                print(f"[GT] Warning: Could not load GT ({e}). Skipping metrics.")

        aucs, aps, times = [], [], []
        ks = ks_all[:args.runs] if args.runs else ks_all
        
        for i, glab in enumerate(ks):
            gi = name2idx.get(glab)
            if gi is None: continue

            # Algorithm Run
            t0 = time.perf_counter()
            v = make_restart_vector(args.restart, P, gi)
            wt = ppr_vector(P, args.alpha, args.iters, args.tol, v=v)
            P_ko = ko_matrix_true(P, gi, mode=args.ko_mode)
            ko = ppr_vector(P_ko, args.alpha, args.iters, args.tol, v=v)
            delta = np.maximum(wt - ko, 0.0) # Using pos delta
            dt = (time.perf_counter() - t0) * 1000

            # Scoring
            auc, ap = float("nan"), float("nan")
            if GT is not None:
                y_full = GT[gi, :].astype(int)
                mask = np.ones_like(y_full, dtype=bool); mask[gi] = False
                y = y_full[mask]; s = delta[mask]
                if y.sum() > 0:
                    try: 
                        auc = roc_auc_score(y, s)
                        ap = average_precision_score(y, s)
                    except: pass
            
            aucs.append(auc); aps.append(ap); times.append(dt)
            print(f"[{i+1}/{len(ks)}] {glab}: AUC={auc:.4f}, AP={ap:.4f}, Time={dt:.1f}ms")

            if not args.no_save:
                _save_results(outdir, glab, gene_names, delta, P, parse_topk(args.topk))

        summ = {
            "dataset": str(counts_path),
            "AUC_mean": float(np.nanmean(aucs)),
            "AP_mean": float(np.nanmean(aps)),
            "params": vars(args)
        }
        print("\n==== SUMMARY ====\n" + json.dumps(summ, indent=2, default=str))
        if not args.no_save:
            with open(Path(outdir)/"tako_summary.json", "w") as f: json.dump(summ, f, indent=2, default=str)

    except Exception:
        traceback.print_exc()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--ko-seq", default=None)
    ap.add_argument("--ko", default=None)
    ap.add_argument("--outdir", default="tako_results")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--graph", default="pcr", choices=["pcr","corr"])
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--hvg", type=int, default=3000)
    ap.add_argument("--force-include", default="")
    ap.add_argument("--W-path", default=None)
    ap.add_argument("--pcr-dims", dest="pcr_pc_d", type=int, default=10)
    ap.add_argument("--pcr-top-pct", type=float, default=8.0)
    ap.add_argument("--pcr-ridge", type=float, default=0.05)
    ap.add_argument("--pcr-binarize", type=int, default=0)
    ap.add_argument("--cutoff", type=float, default=60.0)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--ko-mode", default="no-in-out")
    ap.add_argument("--restart", default="onehot")
    ap.add_argument("--topk", default="10,20,50")
    args = ap.parse_args()

    if args.ko and not args.ko_seq:
        os.makedirs(args.outdir, exist_ok=True)
        t = Path(args.outdir)/"_ko_tmp.txt"
        with open(t, "w") as f: f.write("\n".join(re.split(r"[,\s]+", args.ko)))
        args.ko_seq = str(t)
    
    if not args.ko_seq:
        raise ValueError("Must provide --ko-seq or --ko")

    run_evaluation(args.counts, args.gt, args.ko_seq, args.outdir, args)

if __name__ == "__main__":
    main()
