"""Summaries of the screening runs (primary and supplementary tables).

Usage: python3 summarise.py <runs_dir> <out_dir>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["Ridge", "KRR-RBF", "GP", "GBT", "KRR-RBF-PCA6", "GP-PCA6", "Q-Fidelity", "Q-Projected"]


def ci(x):
    x = np.asarray(x, float)
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def summarise(r):
    g = r.groupby(["subset", "design", "model", "rule", "delta", "alpha"])
    out = g.agg(reps=("rep", "nunique"), hits_mean=("true_hits", "mean"),
                hits_median=("true_hits", "median"), sel_mean=("n_selected", "mean"),
                fdr=("fdp", "mean"), fdp_q975=("fdp", lambda x: np.percentile(x, 97.5)),
                empty=("n_selected", lambda x: np.mean(np.asarray(x) == 0)),
                mae=("mae", "mean"), pool=("n_candidates", "first"),
                true_in_pool=("n_true_in_pool", "mean"), fit_time=("fit_time_s", "mean")).reset_index()
    sel_nonempty = r[r.n_selected > 0].groupby(["subset", "design", "model", "rule", "delta", "alpha"])
    calls = sel_nonempty.apply(lambda d: (d.n_selected.sum() / max(1, d.true_hits.sum()))).rename("calcs_per_hit")
    out = out.merge(calls.reset_index(), how="left")
    out["fdr_se"] = r.groupby(["subset", "design", "model", "rule", "delta", "alpha"])["fdp"].sem().to_numpy()
    return out


def main(runs, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in glob.glob(f"{runs}/results_*.csv"):
        subset = Path(f).stem.split("_")[1]
        d = pd.read_csv(f); d["subset"] = subset; frames.append(d)
    r = pd.concat(frames, ignore_index=True)
    s = summarise(r)
    s.to_csv(out / "summary_all_settings.csv", index=False)
    meta = []
    for f in glob.glob(f"{runs}/meta_*.jsonl"):
        subset = Path(f).stem.split("_")[1]
        for line in open(f):
            m = json.loads(line); m["subset"] = subset; meta.append(m)
    meta = pd.DataFrame(meta)
    meta.to_csv(out / "meta_all.csv", index=False)
    prim = s[(s.delta == 0.2) & (s.alpha == 0.2)]
    prim.to_csv(out / "summary_primary.csv", index=False)
    md = meta.groupby(["subset", "design"]).agg(
        reps=("rep", "nunique"), gd_fid_med=("gd_fidelity", "median"), gd_proj_med=("gd_projected", "median"),
        **({"frac_w_clipped": ("frac_w_clipped", "mean")} if "frac_w_clipped" in meta else {})).reset_index()
    md.to_csv(out / "meta_summary.csv", index=False)
    print(md.to_string(index=False))
    pd.set_option("display.width", 250)
    for subset in sorted(prim.subset.unique()):
        for design in ("reference", "transfer_L10", "transfer_L12"):
            p = prim[(prim.subset == subset) & (prim.design == design)]
            if p.empty:
                continue
            t = p.pivot_table(index="model", columns="rule", values=["hits_mean", "sel_mean", "fdr", "empty"]).round(3)
            print(f"\n=== {subset} / {design}  reps={int(p.reps.max())}  pool={int(p.pool.max())}  "
                  f"true in pool={p.true_in_pool.max():.0f}")
            print(t.reindex([m for m in MODELS if m in t.index]).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
