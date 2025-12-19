from __future__ import annotations
import os
import json
import argparse
import re
import time
import traceback
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from tako.grn import norm_gene_label, build_W_corr_from_counts_with_names, build_W_pcr_struct, load_W, row_normalize
from tako.core import ppr_vector, make_restart_vector, ko_matrix_true, parse_topk

def _load_gt_aligned(gt_arg: str, gene_names: list[str], ko_seq_path: str = None) -> np.ndarray:
    """
    Loads Ground Truth and aligns it to gene_names order.
    Supports .npy (adjacency) or .csv (edge list).
    """
    p = len(gene_names)
    name2idx = {g:i for i,g in enumerate(gene_names)}
    
    final_path = gt_arg
    
    # 1. Directory Auto-matching Logic
    if Path(gt_arg).is_dir() and ko_seq_path:
        with open(ko_seq_path, "r", encoding="utf-8") as f: 
            kos = [x.strip() for x in f if x.strip()]
        
        candidates = list(Path(gt_arg).glob("*.csv")) + list(Path(gt_arg).glob("*.npy")) + list(Path(gt_arg).glob("*.tsv"))
        best_f, best_sc = None, -1
        
        for cand in candidates:
            sc = sum(1 for k in kos if k.lower() in cand.name.lower())
            if sc > best_sc: 
                best_f, best_sc = cand, sc
        
        if best_f: 
            final_path = str(best_f)
            print(f"[GT] Auto-matched file: {final_path}")
        else:
            print(f"[GT] Warning: No matching file found in {gt_arg}")
            return None

    # 2. Load File
    path = Path(final_path)
    if not path.is_file():
        return None

    if path.suffix == ".npy":
        GT = np.load(path)
        GT = (GT > 0).astype(np.uint8)
        np.fill_diagonal(GT, 0)
        if GT.shape != (p, p):
            print(f"[GT] Warning: Shape mismatch {GT.shape} vs ({p},{p}). Skipping.")
            return None
        return GT
    else:
        # CSV/TSV Edge List
        try:
            df = pd.read_csv(path, header=None, sep=None, engine='python')
            if df.shape[0] == p and df.shape[1] == p:
                 GT = (df.values > 0).astype(np.uint8)
                 np.fill_diagonal(GT, 0)
                 return GT
            
            src = df.iloc[:,0].astype(str).map(norm_gene_label).tolist()
            tgt = df.iloc[:,1].astype(str).map(norm_gene_label).tolist()
            GT = np.zeros((p, p), dtype=np.uint8)
            miss = 0
            for s, t in zip(src, tgt):
                i, j = name2idx.get(s), name2idx.get(t)
                if i is not None and j is not None and i != j: 
                    GT[i, j] = 1
                else:
                    miss += 1
            if miss > 0: print(f"[GT] {miss} edges skipped (gene not found).")
            return GT
        except Exception as e:
            print(f"[GT] Error loading CSV: {e}")
            return None

def _save_results(outdir: str, glab: str, gene_names: list[str], 
                  delta: np.ndarray, P: np.ndarray, topk: list[int]):
    """Saves the ranking results for a single KO."""
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

        with open(ko_seq_path, "r", encoding="utf-8") as f:
            ks_all = [norm_gene_label(x) for x in f if x.strip()]
        
        # Log clearly which gene we are targeting
        print(f"[Input] Target KO: {', '.join(ks_all)}")
        
        # --- Build Graph ---
        print(f"[Model] Building Graph: {args.graph}...")
        
        wl = ks_all[:]
        if args.force_include:
            wl += [norm_gene_label(x) for x in re.split(r"[,\s]+", args.force_include) if x.strip()]
        
        if args.W_path:
            print(f"[Graph] Loading W from {args.W_path}")
            W = load_W(args.W_path)
            _, gene_names = build_W_corr_from_counts_with_names(counts_path, hvg=args.hvg, whitelist=wl)
        
        elif args.graph == "pcr":
            W, gene_names = build_W_pcr_struct(
                counts_path, 
                hvg=args.hvg, 
                whitelist=wl,
                pc_d=args.pcr_pc_d,       
                top_pct=args.pcr_top_pct, 
                ridge=args.pcr_ridge, 
                binarize=args.pcr_binarize
            )
        else:
            W, gene_names = build_W_corr_from_counts_with_names(
                counts_path, 
                cutoff=args.cutoff, 
                hvg=args.hvg, 
                whitelist=wl
            )

        p = len(gene_names)
        P = row_normalize(W)
        name2idx = {g:i for i,g in enumerate(gene_names)}
        print(f"[Graph] {p} genes, {np.count_nonzero(P)} edges.")

        # --- Load GT ---
        GT = None
        if gt_arg:
            try:
                GT = _load_gt_aligned(gt_arg, gene_names, ko_seq_path)
                if GT is not None:
                    print(f"[GT] Loaded. Total Edges: {GT.sum()}")
            except Exception as e:
                print(f"[GT] Warning: Could not load GT ({e}). Skipping metrics.")

        # --- Loop (Usually just 1 gene) ---
        aucs, aps, times = [], [], []
        ks = ks_all[:args.runs] if args.runs else ks_all
        topk_list = parse_topk(args.topk)
        
        for i, glab in enumerate(ks):
            gi = name2idx.get(glab)
            if gi is None:
                m = re.match(r"^(?:G|GENE)?\s*(\d+)$", glab, re.IGNORECASE)
                if m: gi = name2idx.get(f"G{int(m.group(1))}")
            
            if gi is None:
                print(f"[{i+1}] Skipping {glab} (not in graph)")
                continue

            # Core Algorithm
            t0 = time.perf_counter()
            
            v = make_restart_vector(args.restart, P, gi)
            wt = ppr_vector(P, args.alpha, args.iters, args.tol, v=v)
            
            P_ko = ko_matrix_true(P, gi, mode=args.ko_mode)
            ko = ppr_vector(P_ko, args.alpha, args.iters, args.tol, v=v)
            
            delta = np.maximum(wt - ko, 0.0) 
            
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
            
            gt_info = f"AUC={auc:.4f}, AP={ap:.4f}" if np.isfinite(auc) else "No GT"
            print(f"[{i+1}/{len(ks)}] {glab}: {gt_info}, Time={dt:.1f}ms")

            if not args.no_save:
                _save_results(outdir, glab, gene_names, delta, P, topk_list)

        # --- Summary ---
        summ = {
            "dataset": str(counts_path),
            "AUC_mean": float(np.nanmean(aucs)),
            "AP_mean": float(np.nanmean(aps)),
            "runtime_ms_mean": float(np.nanmean(times)),
            "params": vars(args)
        }
        print("\n==== SUMMARY ====\n" + json.dumps(summ, indent=2, default=str))
        
        if not args.no_save:
            with open(Path(outdir)/"tako_summary.json", "w") as f: 
                json.dump(summ, f, indent=2, default=str)

    except Exception:
        traceback.print_exc()

def main():
    ap = argparse.ArgumentParser(description="TAKO: Targeted Analysis of KnockOuts via ΔPPR")
    ap.add_argument("--counts", required=True, help="Path to scRNA-seq counts CSV/TSV")
    ap.add_argument("--gt", default=None, help="Ground Truth file or directory (optional)")
    ap.add_argument("--ko", default=None, help="Target KO gene (e.g. MECP2)")
    ap.add_argument("--ko-seq", default=None, help="File containing list of KO genes (for batch)")
    ap.add_argument("--outdir", default="tako_results", help="Directory to save results")
    ap.add_argument("--no-save", action="store_true", help="Run without saving files")
    ap.add_argument("--graph", default="pcr", choices=["pcr","corr"], help="Graph construction method")
    ap.add_argument("--hvg", type=int, default=3000, help="Number of Highly Variable Genes")
    ap.add_argument("--force-include", default="", help="Genes to force include in graph")
    ap.add_argument("--W-path", default=None, help="Load pre-computed adjacency matrix")
    ap.add_argument("--pcr-dims", dest="pcr_pc_d", type=int, default=10, help="PCR: Principal Components")
    ap.add_argument("--pcr-top-pct", type=float, default=8.0, help="PCR: Top % edges to keep")
    ap.add_argument("--pcr-ridge", type=float, default=0.05, help="PCR: Ridge penalty")
    ap.add_argument("--pcr-binarize", type=int, default=0, help="PCR: Binarize edges (0 or 1)")
    ap.add_argument("--cutoff", type=float, default=60.0, help="Corr: Percentile cutoff")
    ap.add_argument("--alpha", type=float, default=0.30, help="PPR restart probability")
    ap.add_argument("--iters", type=int, default=200, help="PPR iterations")
    ap.add_argument("--tol", type=float, default=1e-8, help="PPR convergence tolerance")
    ap.add_argument("--ko-mode", default="no-in-out", choices=["no-in-out","silence-out","restart-row"])
    ap.add_argument("--restart", default="onehot", choices=["onehot","neighbor","uniform"])
    ap.add_argument("--runs", type=int, default=None, help="Limit number of KOs to run")
    ap.add_argument("--topk", default="10,20,50", help="Top-K metrics to save")
    
    args = ap.parse_args()

    if args.ko and not args.ko_seq:
        os.makedirs(args.outdir, exist_ok=True)
        t = Path(args.outdir)/"_ko_tmp.txt"
        # Logic still supports comma-split just in case, but help text implies single
        with open(t, "w") as f: f.write("\n".join(re.split(r"[,\s]+", args.ko)))
        args.ko_seq = str(t)
    
    if not args.ko_seq:
        # Error message updated to reflect single gene expectation
        ap.error("Must provide either --ko <gene_name> or --ko-seq <file>")

    run_evaluation(args.counts, args.gt, args.ko_seq, args.outdir, args)

if __name__ == "__main__":
    main()
