"""Exploratory no-tumor test: false alarms on normal slices put in context with the tumor test images.

For each cross-validation model (configuration x test fold), the highest detection score of every image
(any class) is compared between its own Figshare test fold (all slices contain a tumor) and the external
no-tumor slices. Reported per threshold: alarm rate on tumor slices, correct-detection rate on tumor slices
(right class, IoU >= 0.5), false-alarm rate on no-tumor slices; and the image-level AUC (tumor vs no tumor)
of the highest score. Mean ± SD across the 5 folds, and per-fold differences between configurations.

Usage: python3 negatives_analysis.py --cv runs --results notumor_fp_test_results.zip --out out
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def auc(pos, neg):
    """Mann-Whitney AUC: probability that a tumor slice scores higher than a no-tumor slice (ties count 1/2)."""
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", default=["adopted", "yolov12n"])
    args = ap.parse_args()
    sys.path.insert(0, str(args.cv / "analysis"))
    import cv_analysis as cva

    args.out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.results)
    neg_imgs = pd.read_csv(io.BytesIO(zf.read("notumor_images.csv")))
    neg_scores = pd.read_csv(io.BytesIO(zf.read("notumor_max_scores.csv")))
    subsets = {"main": set(neg_imgs["path"]), "dhash_unique": set(neg_imgs.loc[neg_imgs["dhash_first"], "path"])}
    gt = cva.ground_truth(args.cv / "analysis" / "split_metadata.csv")

    rows = []
    for config in args.configs:
        runs = cva.load_runs(args.cv, config)
        for k in sorted(runs):
            op = float(runs[k]["summary"]["test"]["operating_conf"])
            rec, tp, g = cva.build_records(runs[k], gt)
            rec["tp50"] = tp[:, 0] & (rec["cls"].to_numpy() == g.set_index("name").loc[rec["name"], "cls"].to_numpy())
            tumor_max = rec.groupby("name")["conf"].max().reindex(g["name"]).fillna(0.0).to_numpy()
            for subset, keep in subsets.items():
                ns = neg_scores[(neg_scores.config == config) & (neg_scores.fold == k) & neg_scores.path.isin(keep)]
                nmax = ns["max_conf"].to_numpy()
                row = {"subset": subset, "config": config, "fold": k, "operating_conf": op,
                       "n_tumor": len(g), "n_notumor": len(ns), "auc_tumor_vs_notumor": auc(tumor_max, nmax)}
                for name, t in (("0.25", 0.25), ("op", op), ("0.50", 0.50)):
                    correct = rec[(rec.conf >= t) & rec.tp50]["name"].nunique() / len(g)
                    row[f"tumor_alarm@{name}"] = float((tumor_max >= t).mean())
                    row[f"tumor_correct@{name}"] = correct
                    row[f"notumor_false_alarm@{name}"] = float((nmax >= t).mean())
                rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(args.out / "notumor_vs_tumor_by_fold.csv", index=False)

    lines = []
    metrics = ["tumor_correct@0.25", "tumor_alarm@0.25", "notumor_false_alarm@0.25",
               "tumor_correct@op", "notumor_false_alarm@op", "notumor_false_alarm@0.50", "auc_tumor_vs_notumor"]
    for subset, gs in res.groupby("subset", sort=False):
        lines.append(f"===== {subset}: {gs['n_notumor'].iloc[0]} no-tumor slices; tumor slices = each model's own test fold")
        lines.append(gs[["config", "fold"] + metrics].round(3).to_string(index=False))
        for config, g in gs.groupby("config", sort=False):
            lines.append(f"{config:9s} " + ", ".join(f"{m} {g[m].mean():.3f} ± {g[m].std(ddof=1):.3f}" for m in metrics))
        if len(args.configs) >= 2:
            a, b = (gs[gs.config == c].set_index("fold") for c in args.configs[:2])
            for m in metrics:
                d = (b[m] - a[m]).dropna()
                lines.append(f"  {args.configs[1]} − {args.configs[0]} {m}: {' '.join(f'{x:+.3f}' for x in d)} "
                             f"(mean {d.mean():+.3f}, {int((d > 0).sum())}/{len(d)} folds higher)")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (args.out / "notumor_vs_tumor_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
