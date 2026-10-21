"""Training-seed variability of the patient-level cross-validation.

Reads the run archives cv_runs_<config>_seed<s>_*.zip of the folder given by --cv for the
training seeds --seeds (default 42 0 1) and the test folds --folds (default 1 2). The loading, matching,
AP and bootstrap functions are imported from cv_analysis.py (same folder).
  1. per (configuration, seed, test fold): test metrics computed by Ultralytics (summary.json), with the
     same check as cv_analysis.py (AP recomputed here from predictions.json);
  2. seed variability: per configuration and fold, mean, SD and range across seeds; the seed SD pooled
     within folds is set against the between-fold SD of seed 42 over the 5 folds;
  3. paired comparison of two configurations (second − first, default yolov12n − adopted) on the same
     test patients: for every seed, the difference of the mean over the folds, and the difference of the
     seed-averaged means, each with a 95% CI from the fold-stratified patient bootstrap of cv_analysis.py
     (in every resample the same patients are used for all configurations, seeds and folds).
Outputs (--out): seed_fold_metrics.csv, seed_variability.csv, seed_paired_differences.csv,
seed_statistics_report.txt.

Usage: python3 seed_analysis.py --cv runs --meta split_metadata.csv --out out_seeds
       [--configs adopted yolov12n] [--seeds 42 0 1] [--folds 1 2] [--boot 2000] [--boot-seed 2026]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_analysis import CLASSES, Pooled, build_records, ground_truth, percentile_ci, summarize_ap  # noqa: E402

AP_METRICS = ["mAP50", "mAP50_95"] + [f"AP50_{c}" for c in CLASSES]
VAR_METRICS = ["P", "R"] + AP_METRICS


def load_runs_seed(cv_dir: Path, config: str, seed: int, folds=None, predictions=True) -> dict[int, dict]:
    """Same as cv_analysis.load_runs, for any training seed; later archives overwrite earlier ones."""
    runs: dict[int, dict] = {}
    zips = sorted(cv_dir.glob(f"cv_runs_{config}_seed{seed}_*.zip"), key=lambda p: p.stat().st_mtime)
    for zpath in zips:
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                m = re.fullmatch(rf"{config}/test(\d)_seed{seed}/summary\.json", name)
                if not m or (folds and int(m.group(1)) not in folds):
                    continue
                k, base = int(m.group(1)), name.rsplit("/", 1)[0]
                summary = json.loads(z.read(name))
                assert int(summary["seed"]) == seed and int(summary["test_fold"]) == k, name
                run = {"summary": summary, "archive": zpath.name}
                if predictions:
                    run["predictions"] = json.loads(z.read(f"{base}/test_eval/predictions.json"))
                    run["manifest"] = pd.read_csv(io.BytesIO(z.read(f"{base}/fold_manifest.csv")), dtype={"pid": str})
                runs[k] = run
    return runs


def fold_row(config: str, seed: int, k: int, run: dict) -> dict:
    s = run["summary"]
    t, sel = s["test"], s["selection"]
    row = {"config": config, "seed": seed, "test_fold": k, "best_epoch": sel["best_epoch"],
           "epochs_run": sel["epochs_completed"], "P": t["precision"], "R": t["recall"],
           "mAP50": t["mAP50"], "mAP50_95": t["mAP50_95"]}
    for c in t["per_class"]:
        row[f"AP50_{c['class']}"] = c["AP50"]
        row[f"AP50_95_{c['class']}"] = c["AP50_95"]
    row["archive"] = run["archive"]
    return row


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cv", required=True, type=Path)
    ap_.add_argument("--meta", required=True, type=Path)
    ap_.add_argument("--out", required=True, type=Path)
    ap_.add_argument("--configs", nargs="+", default=["adopted", "yolov12n"])
    ap_.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    ap_.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap_.add_argument("--boot", type=int, default=2000)
    ap_.add_argument("--boot-seed", type=int, default=2026)
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gt = ground_truth(args.meta)
    pids = sorted(gt["pid"].unique())
    report: list[str] = []

    def say(s=""):
        print(s)
        report.append(s)

    # 1. per (config, seed, fold) metrics and recomputation check
    rows, sets, check, missing = [], {}, [], []
    for config in args.configs:
        for seed in args.seeds:
            runs = load_runs_seed(args.cv, config, seed, args.folds)
            for k in args.folds:
                if k not in runs:
                    missing.append(f"{config} seed {seed} fold {k}")
                    continue
                t = runs[k]["summary"]["test"]
                rows.append(fold_row(config, seed, k, runs[k]))
                r, tp, g = build_records(runs[k], gt)
                sets[(config, seed, k)] = Pooled(r, tp, g, pids)
                mine = summarize_ap(sets[(config, seed, k)].ap())
                check.append(max(abs(mine["mAP50"] - t["mAP50"]), abs(mine["mAP50_95"] - t["mAP50_95"]),
                                 *(abs(mine[f"AP50_{c['class']}"] - c["AP50"]) for c in t["per_class"])))
    folds = pd.DataFrame(rows)
    folds.to_csv(args.out / "seed_fold_metrics.csv", index=False)

    say(f"TRAINING-SEED VARIABILITY — test folds {args.folds}, seeds {args.seeds} "
        "(metrics computed by Ultralytics on the test fold; same splits for every seed)")
    if missing:
        say("Not available yet (skipped): " + "; ".join(missing))
    if folds.empty:
        (args.out / "seed_statistics_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        return
    say(f"Check: AP recomputed here from predictions.json vs Ultralytics, max |difference| = {max(check):.4f}")

    # 2. seed variability
    var_rows = []
    for config, g in folds.groupby("config", sort=False):
        say(f"\n{config}")
        say(g[["test_fold", "seed", "best_epoch", "epochs_run"] + VAR_METRICS]
            .sort_values(["test_fold", "seed"]).round(3).to_string(index=False))
        ref = pd.DataFrame([fold_row(config, 42, k, r) for k, r in
                            sorted(load_runs_seed(args.cv, config, 42, predictions=False).items())])
        seeds_avail = sorted(g["seed"].unique().tolist())
        for m in VAR_METRICS:
            per_fold = g.groupby("test_fold")[m]
            n_seeds = per_fold.count()
            pooled_var = (per_fold.var(ddof=1) * (n_seeds - 1)).sum() / max((n_seeds - 1).sum(), 1)
            fold_means = g.groupby("seed")[m].mean()  # mean over the analysed folds, per seed
            row = {"config": config, "metric": m, "seeds": " ".join(map(str, seeds_avail)),
                   "seed_sd_pooled_within_folds": float(np.sqrt(pooled_var)) if (n_seeds > 1).any() else np.nan,
                   "df_seed_sd": int((n_seeds - 1).sum()),
                   "sd_across_seeds_of_folds_mean": float(fold_means.std(ddof=1)) if len(fold_means) > 1 else np.nan,
                   "between_fold_sd_seed42": float(ref[m].std(ddof=1)) if len(ref) > 1 else np.nan,
                   "n_folds_seed42": len(ref)}
            for k, v in per_fold:
                row[f"fold{k}_mean"], row[f"fold{k}_sd"] = v.mean(), v.std(ddof=1)
                row[f"fold{k}_min"], row[f"fold{k}_max"] = v.min(), v.max()
            var_rows.append(row)
        sub = pd.DataFrame([r for r in var_rows if r["config"] == config]).set_index("metric")
        for k in sorted(g["test_fold"].unique()):
            say(f"  fold {k}, mean ± SD across seeds (range): " + ", ".join(
                f"{m} {sub.loc[m, f'fold{k}_mean']:.3f} ± {sub.loc[m, f'fold{k}_sd']:.3f} "
                f"({sub.loc[m, f'fold{k}_min']:.3f}–{sub.loc[m, f'fold{k}_max']:.3f})" for m in AP_METRICS[:2]))
        say(f"  seed SD pooled within folds (df {int(sub['df_seed_sd'].iloc[0])}) vs between-fold SD of seed 42 "
            f"over {int(sub['n_folds_seed42'].iloc[0])} folds:")
        for m in VAR_METRICS:
            say(f"    {m:22s} seed SD {sub.loc[m, 'seed_sd_pooled_within_folds']:.3f} | fold SD "
                f"{sub.loc[m, 'between_fold_sd_seed42']:.3f} | SD across seeds of the folds-{args.folds} mean "
                f"{sub.loc[m, 'sd_across_seeds_of_folds_mean']:.3f}")
    pd.DataFrame(var_rows).to_csv(args.out / "seed_variability.csv", index=False)

    # 3. paired comparison on the same patients, seeds and folds complete for both configurations
    configs = [c for c in args.configs if c in set(folds["config"])]
    if len(configs) < 2:
        say("\nPaired comparison: needs two configurations.")
        (args.out / "seed_statistics_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        return
    a, b = configs[0], configs[1]
    seeds = [s for s in args.seeds if all((c, s, k) in sets for c in (a, b) for k in args.folds)]
    if not seeds:
        say(f"\nPaired comparison {b} − {a}: no seed with folds {args.folds} complete for both yet.")
        (args.out / "seed_statistics_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        return
    fold_pat = {k: np.unique(sets[(a, seeds[0], k)].gt_pat) for k in args.folds}
    rng = np.random.default_rng(args.boot_seed)
    resamples = []
    for _ in range(args.boot):
        w = np.zeros(len(pids), dtype=int)
        for k in args.folds:
            np.add.at(w, rng.choice(fold_pat[k], size=len(fold_pat[k]), replace=True), 1)
        resamples.append(w)

    def values(w=None) -> dict:  # metric per (config, seed): mean over the folds
        out = {}
        for c in (a, b):
            for s in seeds:
                vals = [summarize_ap(sets[(c, s, k)].ap(w)) for k in args.folds]
                out[(c, s)] = np.array([np.mean([v[m] for v in vals]) for m in AP_METRICS])
        return out

    point = values()
    draws = [values(w) for w in resamples]
    say(f"\nPAIRED COMPARISON {b} − {a} (positive = {b} higher), mean over test folds {args.folds}, "
        f"seeds {seeds}; bootstrap {args.boot} resamples of the test patients within each fold "
        f"(seed {args.boot_seed})")
    pair_rows = []
    for label, ss in [(f"seed {s}", [s]) for s in seeds] + ([("mean of seeds", seeds)] if len(seeds) > 1 else []):
        diff_pt = np.mean([point[(b, s)] for s in ss], 0) - np.mean([point[(a, s)] for s in ss], 0)
        diff_bt = np.array([np.mean([d[(b, s)] for s in ss], 0) - np.mean([d[(a, s)] for s in ss], 0) for d in draws])
        parts = []
        for i, m in enumerate(AP_METRICS):
            lo, hi = percentile_ci(diff_bt[:, i])
            pair_rows.append({"comparison": f"{b} - {a}", "seeds": label, "folds": " ".join(map(str, args.folds)),
                              "metric": m, f"{a}": np.mean([point[(a, s)][i] for s in ss]),
                              f"{b}": np.mean([point[(b, s)][i] for s in ss]), "difference": diff_pt[i],
                              "ci95_low": lo, "ci95_high": hi, "share_b_higher": float((diff_bt[:, i] > 0).mean())})
            parts.append(f"{m} {diff_pt[i]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        say(f"  {label:14s} " + ", ".join(parts))
    # per fold: how many of the seed combinations put b above a
    say("  per fold, seed-mean of each configuration and number of (seed of a, seed of b) combinations with b higher:")
    for k in args.folds:
        fa, fb = folds[(folds.config == a) & (folds.test_fold == k)], folds[(folds.config == b) & (folds.test_fold == k)]
        fa, fb = fa[fa.seed.isin(seeds)], fb[fb.seed.isin(seeds)]
        say(f"    fold {k}: " + ", ".join(
            f"{m} {fa[m].mean():.3f} vs {fb[m].mean():.3f} "
            f"({int((fb[m].to_numpy()[None, :] > fa[m].to_numpy()[:, None]).sum())}/{len(fa) * len(fb)})"
            for m in AP_METRICS[:2] + ["AP50_glioma"]))
    pd.DataFrame(pair_rows).to_csv(args.out / "seed_paired_differences.csv", index=False)
    say("  A difference whose 95% CI excludes 0 is considered significant. The CI reflects the sampling of test "
        "patients for the given seed(s); the spread between seeds is reported above.")
    (args.out / "seed_statistics_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
