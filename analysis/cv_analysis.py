"""Statistical analysis of the patient-level 5-fold cross-validation.

Reads the run archives cv_runs_<config>_seed42_*.zip of the folder given by --cv:
  1. per-fold test metrics as computed by Ultralytics (summary.json), mean and SD across folds;
  2. check: the same metrics recomputed here from the saved test predictions (predictions.json)
     and the ground truth (split_metadata.csv), with Ultralytics' matching and AP rules;
  3. 95% confidence intervals of the cross-validated mean (mean of the 5 test-fold metrics) by a
     patient-level bootstrap stratified by fold: in each resample the test patients of every fold are
     drawn with replacement, every fold is re-evaluated and the 5 values are averaged;
  4. paired comparison of two configurations on the same patients: per-fold differences, and the
     bootstrap CI of the difference of the cross-validated means (the same resampled patients for
     both models in every resample);
  5. sensitivity analysis: the same for the pooled out-of-fold evaluation (all test images of the
     5 folds evaluated together; scores of 5 different models are then ranked together).
Outputs (folder given by --out): cv_fold_metrics.csv, cv_bootstrap_ci.csv,
cv_paired_differences.csv, cv_statistics_report.txt.

Usage: python3 cv_analysis.py --cv runs --meta split_metadata.csv --out out
       [--configs adopted yolov12n] [--boot 2000] [--seed 2026]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["meningioma", "glioma", "pituitary_tumor"]
IOUV = np.linspace(0.5, 0.95, 10)
IMG_SIZE = 640  # all slices are stored as 640 x 640 (split_metadata: saved_height/width)


# ----------------------------------------------------------------------------- loading
def load_runs(cv_dir: Path, config: str) -> dict[int, dict]:
    """Per test fold: summary, predictions and manifest, from the most recent archive holding it."""
    runs: dict[int, dict] = {}
    zips = sorted(cv_dir.glob(f"cv_runs_{config}_seed42_*.zip"), key=lambda p: p.stat().st_mtime)
    for zpath in zips:  # later archives overwrite earlier ones
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                m = re.fullmatch(rf"{config}/test(\d)_seed42/summary\.json", name)
                if not m:
                    continue
                k = int(m.group(1))
                base = name.rsplit("/", 1)[0]
                runs[k] = {
                    "summary": json.loads(z.read(name)),
                    "predictions": json.loads(z.read(f"{base}/test_eval/predictions.json")),
                    "manifest": pd.read_csv(io.BytesIO(z.read(f"{base}/fold_manifest.csv")), dtype={"pid": str}),
                    "archive": zpath.name,
                }
    return runs


def ground_truth(meta_csv: Path) -> pd.DataFrame:
    md = pd.read_csv(meta_csv, dtype={"pid": str})
    assert (md["saved_height"] == IMG_SIZE).all() and (md["saved_width"] == IMG_SIZE).all()
    md["name"] = md["image"].map(lambda s: Path(s).name)
    cx, cy, w, h = (md[c].to_numpy() * IMG_SIZE for c in ("x_center", "y_center", "width", "height"))
    md["x1"], md["y1"], md["x2"], md["y2"] = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return md.set_index("name")


# ----------------------------------------------------------------------------- matching (Ultralytics)
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-7)


def match_predictions(pred_cls, gt_cls, iou) -> np.ndarray:
    """Same rule as ultralytics BaseValidator.match_predictions (iou: [n_gt, n_pred])."""
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= t)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def build_records(run: dict, gt: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Predictions with TP flags at the 10 IoU thresholds, and the ground truth of the test images."""
    test = run["manifest"][run["manifest"]["role"] == "test"]
    names = set(test["image"])
    preds = pd.DataFrame(run["predictions"])
    preds["name"] = preds["image_id"].map(lambda i: f"{int(i):04d}.png")
    assert set(preds["name"]) <= names, "prediction on a non-test image"
    rows, tps = [], []
    for name, g in preds.groupby("name", sort=False):
        b = np.array(g["bbox"].tolist(), dtype=float)
        pb = np.c_[b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]]
        r = gt.loc[name]
        gb = np.array([[r.x1, r.y1, r.x2, r.y2]])
        pc = g["category_id"].to_numpy(int) - 1  # Ultralytics writes category_id = class index + 1
        tps.append(match_predictions(pc, np.array([int(r.class_id)]), box_iou(gb, pb)))
        rows.append(pd.DataFrame({"name": name, "pid": r.pid, "cls": pc, "conf": g["score"].to_numpy(float)}))
    records = pd.concat(rows, ignore_index=True)
    gtt = gt.loc[sorted(names), ["pid", "class_id"]].rename(columns={"class_id": "cls"}).reset_index()
    return records, np.concatenate(tps), gtt


# ----------------------------------------------------------------------------- AP (Ultralytics)
def compute_ap(recall, precision) -> float:
    # Ultralytics 8.4.104: the precision drops to 0 right after the last recall reached
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def ap_table(tp, conf, pred_cls, gt_cls) -> np.ndarray:
    """AP [n_classes, 10] as in ultralytics ap_per_class (classes absent from the GT -> NaN)."""
    order = np.argsort(-conf, kind="stable")
    tp, pred_cls = tp[order], pred_cls[order]
    ap = np.full((len(CLASSES), len(IOUV)), np.nan)
    for c in range(len(CLASSES)):
        n_l = int((gt_cls == c).sum())
        if n_l == 0:
            continue
        m = pred_cls == c
        if not m.any():
            ap[c] = 0.0
            continue
        tpc = tp[m].cumsum(0)
        fpc = (1 - tp[m]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc)
        ap[c] = [compute_ap(recall[:, j], precision[:, j]) for j in range(len(IOUV))]
    return ap


def summarize_ap(ap: np.ndarray) -> dict:
    out = {"mAP50": np.nanmean(ap[:, 0]), "mAP50_95": np.nanmean(ap)}
    for c, name in enumerate(CLASSES):
        out[f"AP50_{name}"] = ap[c, 0]
        out[f"AP50_95_{name}"] = ap[c].mean()
    return out


# ----------------------------------------------------------------------------- bootstrap
class Pooled:
    """Out-of-fold predictions of one configuration, sorted once by confidence (stable)."""

    def __init__(self, records: pd.DataFrame, tp: np.ndarray, gtt: pd.DataFrame, pids: list[str]):
        index = {p: i for i, p in enumerate(pids)}
        order = np.argsort(-records["conf"].to_numpy(), kind="stable")
        self.tp = tp[order].astype(np.float64)
        self.conf = records["conf"].to_numpy(float)[order]
        self.cls = records["cls"].to_numpy(int)[order]
        self.pat = records["pid"].map(index).to_numpy(int)[order]
        self.gt_cls = gtt["cls"].to_numpy(int)
        self.gt_pat = gtt["pid"].map(index).to_numpy(int)

    def ap(self, weights: np.ndarray | None = None) -> np.ndarray:
        # predictions are kept in confidence order: their rank is passed as the score
        if weights is None:
            return ap_table(self.tp, -np.arange(len(self.cls), dtype=float), self.cls, self.gt_cls)
        rep = weights[self.pat]
        keep = np.repeat(np.arange(len(self.cls)), rep)  # each patient's predictions repeated, order kept
        gt_keep = np.repeat(np.arange(len(self.gt_cls)), weights[self.gt_pat])
        return ap_table(self.tp[keep], -keep.astype(float), self.cls[keep], self.gt_cls[gt_keep])


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cv", required=True, type=Path)
    ap_.add_argument("--meta", required=True, type=Path)
    ap_.add_argument("--out", required=True, type=Path)
    ap_.add_argument("--configs", nargs="+", default=["adopted", "yolov12n"])
    ap_.add_argument("--boot", type=int, default=2000)
    ap_.add_argument("--seed", type=int, default=2026)
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gt = ground_truth(args.meta)
    pids = sorted(gt["pid"].unique())
    report: list[str] = []

    def say(s=""):
        print(s)
        report.append(s)

    # 1-2. per-fold metrics (Ultralytics) and recomputation check
    fold_rows, sets, check = [], {}, []
    for config in args.configs:
        runs = load_runs(args.cv, config)
        sets[config] = {}
        for k in sorted(runs):
            s = runs[k]["summary"]
            t, sel = s["test"], s["selection"]
            row = {"config": config, "test_fold": k, "val_fold": s["val_fold"], "n_test_images": s["n_test_images"],
                   "n_test_patients": s["n_test_patients"], "best_epoch": sel["best_epoch"],
                   "epochs_run": sel["epochs_completed"], "P": t["precision"], "R": t["recall"],
                   "mAP50": t["mAP50"], "mAP50_95": t["mAP50_95"], "archive": runs[k]["archive"]}
            for c in t["per_class"]:
                row[f"AP50_{c['class']}"] = c["AP50"]
                row[f"AP50_95_{c['class']}"] = c["AP50_95"]
                row[f"R_{c['class']}"] = c["recall"]
            fold_rows.append(row)
            r, tp, g = build_records(runs[k], gt)
            sets[config][k] = Pooled(r, tp, g, pids)
            mine = summarize_ap(sets[config][k].ap())
            check.append(max(abs(mine["mAP50"] - t["mAP50"]), abs(mine["mAP50_95"] - t["mAP50_95"]),
                             *(abs(mine[f"AP50_{c['class']}"] - c["AP50"]) for c in t["per_class"])))
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(args.out / "cv_fold_metrics.csv", index=False)

    metrics = ["P", "R", "mAP50", "mAP50_95"] + [f"AP50_{c}" for c in CLASSES]
    ap_metrics = ["mAP50", "mAP50_95"] + [f"AP50_{c}" for c in CLASSES] + [f"AP50_95_{c}" for c in CLASSES]
    say("PATIENT-LEVEL 5-FOLD CROSS-VALIDATION — test-fold metrics computed by Ultralytics")
    say("(test = fold k, validation = fold k+1 for checkpoint selection, training = the other three folds)")
    for config, g in folds.groupby("config", sort=False):
        say(f"\n{config}: folds {g['test_fold'].tolist()}")
        say(g[["test_fold", "best_epoch", "epochs_run"] + metrics].round(3).to_string(index=False))
        say("mean ± SD across folds: " + ", ".join(f"{m} {g[m].mean():.3f} ± {g[m].std(ddof=1):.3f}" for m in metrics))
    say(f"\nCheck: AP recomputed here from predictions.json vs Ultralytics, max |difference| over all folds "
        f"and metrics = {max(check):.4f}")

    # 3-5. bootstrap (fold-stratified patient resampling, shared by all configurations)
    common = sorted(set.intersection(*(set(v) for v in sets.values() if v))) if any(sets.values()) else []
    configs = [c for c in args.configs if sets.get(c)]
    if not common:
        (args.out / "cv_statistics_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        return
    fold_pat = {k: np.unique(sets[configs[0]][k].gt_pat) for k in common}
    rng = np.random.default_rng(args.seed)
    resamples = []
    for _ in range(args.boot):
        w = np.zeros(len(pids), dtype=int)
        for k in common:
            np.add.at(w, rng.choice(fold_pat[k], size=len(fold_pat[k]), replace=True), 1)
        resamples.append(w)

    def cv_mean(config, w=None):  # mean over the common folds of the per-fold AP metrics
        vals = [summarize_ap(sets[config][k].ap(w)) for k in common]
        return {m: float(np.mean([v[m] for v in vals])) for m in ap_metrics}

    def pooled(config, w=None):  # all test images of the common folds evaluated together
        r = [sets[config][k] for k in common]
        merged = Pooled.__new__(Pooled)
        conf_order = np.argsort(-np.concatenate([x.conf for x in r]), kind="stable")
        merged.tp = np.concatenate([x.tp for x in r])[conf_order]
        merged.cls = np.concatenate([x.cls for x in r])[conf_order]
        merged.pat = np.concatenate([x.pat for x in r])[conf_order]
        merged.conf = np.concatenate([x.conf for x in r])[conf_order]
        merged.gt_cls = np.concatenate([x.gt_cls for x in r])
        merged.gt_pat = np.concatenate([x.gt_pat for x in r])
        return summarize_ap(merged.ap(w))

    n_pat = sum(len(v) for v in fold_pat.values())
    say(f"\nBOOTSTRAP: {args.boot} resamples of the test patients within each fold (seed {args.seed}); "
        f"folds {common}, {n_pat} patients")
    rows, draws = [], {}
    for config in configs:
        for kind, fn in (("cv_mean", cv_mean), ("pooled", pooled)):
            point = fn(config)
            samples = pd.DataFrame([fn(config, w) for w in resamples])
            draws[(config, kind)] = samples
            for m in ap_metrics:
                lo, hi = percentile_ci(samples[m].to_numpy())
                rows.append({"config": config, "estimate": kind, "metric": m, "value": point[m],
                             "ci95_low": lo, "ci95_high": hi})
    ci = pd.DataFrame(rows)
    ci.to_csv(args.out / "cv_bootstrap_ci.csv", index=False)
    for kind, title in (("cv_mean", "Cross-validated mean (mean of the test folds) with 95% CI"),
                        ("pooled", "Sensitivity: pooled out-of-fold evaluation with 95% CI")):
        say(f"\n{title}")
        for config in configs:
            sub = ci[(ci.config == config) & (ci.estimate == kind)].set_index("metric")
            say(f"  {config:9s} " + ", ".join(f"{m} {sub.loc[m, 'value']:.3f} [{sub.loc[m, 'ci95_low']:.3f}, "
                                              f"{sub.loc[m, 'ci95_high']:.3f}]" for m in ap_metrics[:5]))

    if len(configs) >= 2:
        a, b = configs[0], configs[1]
        say(f"\nPAIRED COMPARISON {b} − {a} on the same test patients (positive = {b} higher)")
        fa = folds[(folds.config == a) & folds.test_fold.isin(common)].set_index("test_fold")
        fb = folds[(folds.config == b) & folds.test_fold.isin(common)].set_index("test_fold")
        pair_rows = []
        for m in metrics + [f"AP50_95_{c}" for c in CLASSES]:
            d = (fb[m] - fa[m]).loc[common]
            row = {"metric": m, "per_fold_differences": " ".join(f"{x:+.3f}" for x in d),
                   "mean_difference": d.mean(), "folds_b_higher": int((d > 0).sum()), "n_folds": len(d)}
            extra = ""
            if m in ap_metrics:
                for kind in ("cv_mean", "pooled"):
                    diff = draws[(b, kind)][m].to_numpy() - draws[(a, kind)][m].to_numpy()
                    lo, hi = percentile_ci(diff)
                    est = (cv_mean if kind == "cv_mean" else pooled)
                    row.update({f"{kind}_difference": est(b)[m] - est(a)[m], f"{kind}_ci95_low": lo,
                                f"{kind}_ci95_high": hi, f"{kind}_share_b_higher": float((diff > 0).mean())})
                extra = (f" | CV mean {row['cv_mean_difference']:+.3f} [{row['cv_mean_ci95_low']:+.3f}, "
                         f"{row['cv_mean_ci95_high']:+.3f}] | pooled {row['pooled_difference']:+.3f} "
                         f"[{row['pooled_ci95_low']:+.3f}, {row['pooled_ci95_high']:+.3f}]")
            pair_rows.append(row)
            say(f"  {m:22s} folds {row['per_fold_differences']} ({row['folds_b_higher']}/{row['n_folds']} higher){extra}")
        pd.DataFrame(pair_rows).to_csv(args.out / "cv_paired_differences.csv", index=False)
        say("  A difference whose 95% CI excludes 0 is considered significant. P and R depend on a per-fold "
            "confidence threshold and are reported per fold only.")
    (args.out / "cv_statistics_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
