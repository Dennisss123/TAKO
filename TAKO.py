# -*- coding: utf-8 -*-
"""
TAKO.py — CLI entrypoint for the TAKO algorithm.

This script handles argument parsing, data loading, and the main evaluation loop.
It delegates core algorithms to the 'tako' package.
"""

from __future__ import annotations
import os
import json
import argparse
import re
import time
import warnings
import random
import traceback
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# --- Imports from the TAKO package ---
# Ensure you have a folder named 'tako' containing __init__.py, grn.py, gt.py, core.py, panel.py
from tako.grn import _G_PAT, norm_gene_label, build_W_corr_from_counts_with_names, build_W_GENKI_pcr, load_W, row_normalize
from tako.gt import resolve_gt_inputs, load_gt_aligned
from tako.core import ppr_vector, make_restart_vector, ko_matrix_true, parse_topk, precision_recall_at_k
from tako.panel import _save_kopanel

def run_evaluation(counts_path, gt_arg, ko_seq_path, outdir, args):
    try:
        print(f"\n==== RUN START: counts={counts_path} | gt={gt_arg} | ko_seq={ko_seq_path} ====")

        # 1) Read KO list (target genes)
        #    Also serves as the 'whitelist' to keep during HVG selection if needed
        with open(ko_seq_path, "r", encoding="utf-8") as f:
            ks_all = [norm_gene_label(x) for x in f if x.strip()]

        # 2) Build or Load the Weight Matrix (W)
        #    Merge --force-include genes into the whitelist
        wl = []
        if getattr(args, "force_include", None):
            wl = [norm_gene_label(x) for x in re.split(r"[,\s]+", args.force_include) if x.strip()]
        else:
            wl = ks_all[:]  # default: keep KOs in the graph

        if args.W_path:
            print(f"[PPR] loading W from {args.W_path}")
            W = load_W(args.W_path)
            # We still need gene names from the counts file to map indices
            # Using a loose cutoff/hvg here just to get names, actual W is loaded
            _, gene_names = build_W_corr_from_counts_with_names(
                counts_path, cutoff=args.cutoff, hvg=args.hvg, corr_dtype=args.corr_dtype, whitelist=wl
            )
        else:
            if args.graph.lower() == "pcr":
                print(f"[PPR] GENKI-PCR (hvg={args.hvg}, pc_d={args.genki_pc_d}, "
                      f"top_pct={args.genki_top_pct}, ridge={args.genki_ridge}, binarize={args.genki_binarize})")
                W, gene_names = build_W_GENKI_pcr(
                    counts_csv=counts_path,
                    hvg=args.hvg,
                    whitelist=wl,
                    pc_d=args.genki_pc_d,
                    top_pct=args.genki_top_pct,
                    ridge=args.genki_ridge,
                    binarize=args.genki_binarize
                )
            else:
                print(f"[PPR] CORR graph (cutoff={args.cutoff}, hvg={args.hvg}, corr_dtype={args.corr_dtype})")
                W, gene_names = build_W_corr_from_counts_with_names(
                    counts_path, cutoff=args.cutoff, hvg=args.hvg, corr_dtype=args.corr_dtype, whitelist=wl
                )

        p = len(gene_names)
        assert W.shape == (p, p), f"W shape {W.shape} != ({p},{p})"
        
        # Row-normalize W to get Transition Matrix P
        P = row_normalize(W)
        nnzP = int(np.count_nonzero(P))
        print(f"[W] shape={W.shape}, nnz(P)={nnzP}, genes[0:6]={gene_names[:6]}")

        # 3) Load Ground Truth (GT) - Optional
        GT = None
        if gt_arg:
            gt_csv, gt_adj = None, args.gt_adj
            # auto-resolve if gt_arg is a directory
            gt_csv_auto, auto_adj = resolve_gt_inputs(gt_arg, ko_seq_path)
            if auto_adj is not None:
                gt_adj = auto_adj
            if gt_csv_auto is not None:
                gt_csv = gt_csv_auto
            
            GT = load_gt_aligned(gt_csv if gt_csv else gt_arg, gt_adj, gene_names)
            assert GT.shape == (p, p)
            total_edges = int(GT.sum())
            genes_with_out = int((GT.sum(axis=1) > 0).sum())
            print(f"[GT] edges={total_edges} | genes_with_outdeg>0={genes_with_out}/{p}")
        else:
            print("[GT] not provided; will skip AUC/AP and only export panels.")

        # 4) Debug Checks (Optional)
        if args.debug and GT is not None and args.graph.lower() == "pcr":
            try:
                # Quick orientation check
                gi_dbg = None
                for g in ks_all:
                    if g in gene_names:
                        gi_dbg = gene_names.index(g); break
                if gi_dbg is not None:
                    out_corr = np.corrcoef(W[gi_dbg,:], GT[gi_dbg,:])[0,1]
                    in_corr  = np.corrcoef(W[:,gi_dbg], GT[:,gi_dbg])[0,1]
                    dens = (W>0).sum() / (p*p)
                    print(f"[DEBUG] g={gene_names[gi_dbg]} | row-vs-GT_out={out_corr:.4f} | col-vs-GT_in={in_corr:.4f} | dens={dens:.5f}")
            except Exception as e:
                print(f"[DEBUG] orientation check failed: {e}")

        # 5) Prepare for Evaluation Loop
        ks = ks_all if args.runs is None else ks_all[: int(args.runs)]
        print(f"[ko_seq] N={len(ks)} | head: {', '.join(ks[:6]) if ks else '(empty)'}")
        name2idx = {g:i for i,g in enumerate(gene_names)}

        topk_list = parse_topk(args.topk)

        # Init Summary CSV
        if not args.no_save:
            os.makedirs(outdir, exist_ok=True)
            per_csv = Path(outdir) / "tako_perko.csv"
            with open(per_csv, "w", encoding="utf-8") as f:
                cols = ["gene","ko_mode","restart","delta_mode","pos","neg","AUC","AP"]
                for k in topk_list:
                    cols += [f"P@{k}", f"R@{k}"]
                cols += ["runtime_ms"]
                f.write(",".join(cols) + "\n")
        else:
            per_csv = None

        aucs, aps, times_ms = [], [], []
        Pks, Rks = {k: [] for k in topk_list}, {k: [] for k in topk_list}
        n_total = len(ks)

        # === MAIN LOOP: Iterate over each Knockout Gene ===
        for i, glab in enumerate(ks, 1):
            gi = name2idx.get(glab)
            # Fallback for G123 format if needed
            if gi is None:
                m = _G_PAT.match(glab)
                if m: gi = name2idx.get(f"G{int(m.group(1))}")
            
            if gi is None:
                print(f"[warn] skip unmapped KO: {glab}")
                continue

            # A. Make Restart Vector (v)
            v = make_restart_vector(args.restart, P, gi)
            if args.restart == "neighbor" and v[gi] > 0.95:
                warnings.warn(f"[v] restart places >0.95 mass on...degenerate. Consider --restart onehot/uniform.", RuntimeWarning)

            t0 = time.perf_counter()

            # B. Base PPR (WT state)
            wt = ppr_vector(P, args.alpha, args.iters, args.tol, v=v)

            # C. True KO PPR (Mutant state)
            #    This is the core innovation: modify the graph structure P -> P_ko
            P_ko = ko_matrix_true(P, gi, mode=args.ko_mode)
            ko = ppr_vector(P_ko, args.alpha, args.iters, args.tol, v=v)

            dt_ms = (time.perf_counter() - t0) * 1000.0

            # D. Calculate Delta (Impact Score)
            delta_raw = wt - ko
            delta_pos = np.maximum(delta_raw, 0.0) # Downregulated targets
            delta_abs = np.abs(delta_raw)          # All dysregulated targets

            if args.delta_mode == "pos":
                delta = delta_pos
            elif args.delta_mode == "abs":
                delta = delta_abs
            else:
                delta = delta_raw

            # E. Save Individual KO Panel
            if not args.no_save:
                _save_kopanel(Path(outdir), gene_names[gi], gene_names, delta_pos, delta_raw, delta_abs, P, topk_list)

            # F. Compute Metrics (if GT exists)
            pos = neg = 0
            auc = ap = float("nan")
            P_at, R_at = {}, {}
            
            if GT is not None:
                y_full = GT[gi, :].astype(int)
                # Remove self from evaluation
                mask = np.ones_like(y_full, dtype=bool); mask[gi] = False
                y = y_full[mask]
                s = delta[mask]
                
                pos = int(y.sum())
                neg = int(len(y) - pos)

                if pos > 0 and pos < len(y):
                    try:   auc = float(roc_auc_score(y, s))
                    except Exception: pass
                if pos > 0:
                    try:   ap  = float(average_precision_score(y, s))
                    except Exception: pass

                if len(topk_list) > 0 and len(y) > 0:
                    P_at, R_at = precision_recall_at_k(y, s, topk_list)

            aucs.append(auc); aps.append(ap); times_ms.append(dt_ms)
            for k in topk_list:
                Pks[k].append(P_at.get(k, np.nan))
                Rks[k].append(R_at.get(k, np.nan))

            # Log Progress
            pr = " ".join([f"P@{k}={P_at.get(k, np.nan):.3f}" if k in P_at else f"P@{k}=NA" for k in topk_list])
            rr = " ".join([f"R@{k}={R_at.get(k, np.nan):.3f}" if k in R_at else f"R@{k}=NA" for k in topk_list])
            print(f"[{i:02d}/{n_total}] g={gi:4d} ({gene_names[gi]}) | pos={pos:4d} neg={neg:4d} | "
                  f"AUC={auc:.4f} AP={ap:.4f} | {args.ko_mode}/{args.restart} | {dt_ms:.1f} ms | {pr} | {rr}")

            # Append to CSV
            if per_csv is not None:
                with open(per_csv, "a", encoding="utf-8") as f:
                    row = [gene_names[gi], args.ko_mode, args.restart, args.delta_mode,
                           str(pos), str(neg),
                           f"{auc:.6f}" if np.isfinite(auc) else "",
                           f"{ap:.6f}"  if np.isfinite(ap)  else ""]
                    for k in topk_list:
                        pk = P_at.get(k, np.nan); rk = R_at.get(k, np.nan)
                        row += [f"{pk:.6f}" if np.isfinite(pk) else "",
                                f"{rk:.6f}" if np.isfinite(rk) else ""]
                    row += [f"{dt_ms:.1f}"]
                    f.write(",".join(row) + "\n")

        # 6) Final Summary Statistics
        def _nanstd(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            return float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        def _nanmean(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            return float(np.mean(x)) if x.size > 0 else float("nan")
        def _nanmedian(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            return float(np.median(x)) if x.size > 0 else float("nan")
        def _nan_iqr(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            return float(np.percentile(x, 75) - np.percentile(x, 25)) if x.size > 0 else float("nan")

        runs_done = len(aucs)
        summ = {
            "dataset": str(Path(counts_path).parent),
            "counts": counts_path,
            "gt": gt_arg or "(none)",
            "ko_seq": str(ko_seq_path),
            "runs": runs_done,
            "alpha": args.alpha,
            "iters": args.iters,
            "tol": args.tol,
            "graph": args.graph,
            "cutoff": args.cutoff if (args.W_path is None and args.graph=="corr") else None,
            "W_path": args.W_path or "(built from counts)",
            "genki_pc_d": args.genki_pc_d,
            "genki_top_pct": args.genki_top_pct,
            "genki_binarize": args.genki_binarize,
            "genki_ridge": args.genki_ridge,
            "ko_mode": args.ko_mode,
            "restart": args.restart,
            "delta_mode": args.delta_mode,
            "p": int(p),
            "nnzP": nnzP,
            "AUC_mean": _nanmean(aucs),
            "AUC_sd": _nanstd(aucs),
            "AUC_median": _nanmedian(aucs),
            "AUC_iqr": _nan_iqr(aucs),
            "AP_mean": _nanmean(aps),
            "AP_sd": _nanstd(aps),
            "AP_median": _nanmedian(aps),
            "AP_iqr": _nan_iqr(aps),
            "runtime_median_ms": _nanmedian(times_ms),
            "runtime_iqr_ms": _nan_iqr(times_ms),
        }

        print("\n==== Summary (print+save) ====\n" + json.dumps(summ, indent=2, ensure_ascii=False))

        if not args.no_save:
            out_json = Path(outdir) / "tako_summary.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2, ensure_ascii=False)
            print(f"[save] {out_json}")
            if 'per_csv' in locals() and per_csv is not None:
                print(f"[save] {per_csv}")

    except Exception as e:
        print(f"[ERROR] run failed for counts={counts_path} : {e}")
        traceback.print_exc()

def main():
    ap = argparse.ArgumentParser(description="TAKO: True KO Analysis Algorithm")
    
    # Input Data
    ap.add_argument("--counts", required=False, help="Path to scRNA-seq counts CSV")
    ap.add_argument("--gt", required=False, help="Ground Truth file or dir (empty -> skip AUC/AP)")
    ap.add_argument("--gt-adj", default=None, help="p×p adjacency .npy aligned to counts order (overrides auto-match)")
    
    # KO Targets
    ap.add_argument("--ko-seq", required=False, help="File with one KO gene per line")
    ap.add_argument("--ko", default=None, help="Single/Multi KO names (comma/space separated)")
    
    # Output
    ap.add_argument("--outdir", required=False, help="Directory to save results")
    ap.add_argument("--no-save", action="store_true", help="Dry run: print only, no files saved")

    # Algorithm Params
    ap.add_argument("--runs", type=int, default=None, help="Limit number of KOs to run (for testing)")
    ap.add_argument("--alpha", type=float, default=0.30, help="Restart probability (1-damping)")
    ap.add_argument("--iters", type=int, default=200, help="Max PPR iterations")
    ap.add_argument("--tol", type=float, default=1e-8, help="PPR convergence tolerance")
    
    # Graph Construction
    ap.add_argument("--graph", choices=["pcr","corr"], default="pcr", help="Graph type: 'pcr' (GenKI-style) or 'corr'")
    ap.add_argument("--hvg", type=int, default=3000, help="Number of Highly Variable Genes")
    ap.add_argument("--force-include", default="", help="Force-keep these genes during HVG (comma/space separated)")
    ap.add_argument("--W-path", dest="W_path", default=None, help="Load pre-computed W matrix (.npy)")

    # PCR / GenKI Params
    ap.add_argument("--genki-pc-d", type=int, default=10, help="Number of PCs for PCR")
    ap.add_argument("--genki-top-pct", type=float, default=8.0, help="Top % edges to keep")
    ap.add_argument("--genki-binarize", type=int, default=0, help="1=Binarize weights, 0=Weighted")
    ap.add_argument("--genki-ridge", type=float, default=0.05, help="Ridge penalty")

    # Correlation Params
    ap.add_argument("--cutoff", type=float, default=60.0, help="Percentile cutoff for CORR graph")
    ap.add_argument("--corr-dtype", default="float32", choices=["float32","float64"])

    # Semantics
    ap.add_argument("--ko-mode", default="no-in-out", choices=["no-in-out","silence-out","restart-row"], help="True KO implementation mode")
    ap.add_argument("--restart", default="onehot", choices=["neighbor","uniform","onehot"], help="Restart vector strategy")
    ap.add_argument("--delta-mode", default="pos", choices=["pos","abs","raw"], help="Score metric: 'pos' (down-reg), 'abs' (dys-reg)")
    ap.add_argument("--topk", default="10,20,50", help="Comma-separated K for Precision@K")

    ap.add_argument("--debug", action="store_true", help="Print debug info")
    ap.add_argument("--batch", default=None, help="Batch CSV file: counts,gt,ko_seq,outdir")

    args = ap.parse_args()

    # Convenience: Allow --ko MECP2 instead of requiring a file
    if args.ko and args.ko_seq:
        raise ValueError("Use either --ko or --ko-seq, not both.")

    # Batch Mode
    if args.batch:
        print(f"[BATCH] loading {args.batch}")
        with open(args.batch, "r", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            rows = [row for row in rdr if row and any(cell.strip() for cell in row)]
        if not rows:
            raise FileNotFoundError(f"[BATCH] no rows found in {args.batch}")
        for row in rows:
            try:
                counts_path = row[0].strip()
                gt_arg = row[1].strip() or None
                ko_seq_path = row[2].strip()
                outdir = row[3].strip()
            except Exception:
                print(f"[BATCH] malformed row, skipping: {row}")
                continue
            run_evaluation(counts_path, gt_arg, ko_seq_path, outdir, args)
    else:
        # Single Run Mode
        required = ["counts","outdir"]
        missing = [p for p in required if getattr(args, p) is None]
        
        # Handle the case where user provides --ko string but not file
        if args.ko and not args.ko_seq:
            if not args.outdir:
                raise ValueError("Must provide --outdir when using --ko")
            os.makedirs(args.outdir, exist_ok=True)
            kos = [norm_gene_label(k) for k in re.split(r"[,\s]+", args.ko) if k.strip()]
            if not kos:
                raise ValueError("--ko is empty")
            _ko_tmp = Path(args.outdir) / "_ko_tmp.txt"
            with open(_ko_tmp, "w", encoding="utf-8") as fh:
                fh.write("\n".join(kos) + "\n")
            args.ko_seq = str(_ko_tmp)

        if missing:
            raise ValueError(f"Missing required args: {missing} (or use --batch)")
            
        os.makedirs(args.outdir, exist_ok=True)
        run_evaluation(args.counts, args.gt, args.ko_seq, args.outdir, args)

if __name__ == "__main__":
    main()
