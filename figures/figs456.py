import sys, json, csv
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cmlib import load, cm

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9.5, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
                     "axes.labelcolor": "#111111", "grid.color": "#e2e2e2", "grid.linewidth": 0.6})
COL = {"adopted": "#2a78d6", "yolov12n": "#eb6834"}
LAB = {"adopted": "Adopted model", "yolov12n": "Native YOLOv12n"}
CFGS = ("adopted", "yolov12n")
K = range(1, 6)

def summ(cfg, k):
    return json.load(open(f"data/{cfg}/test{k}_seed42/summary.json"))

def results(cfg, k):
    rows = list(csv.DictReader(open(f"data/{cfg}/test{k}_seed42/train/results.csv")))
    return {c.strip(): np.array([float(r[c]) for r in rows]) for c in rows[0]}

def style(ax):
    ax.grid(True); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------------- Fig. 4
fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4))
cols = [("train/box_loss", "(a) Training box loss", "Box loss"), ("val/box_loss", "(b) Validation box loss", "Box loss"),
        ("metrics/mAP50(B)", "(c) Validation mAP50", "mAP50"), ("metrics/mAP50-95(B)", "(d) Validation mAP50-95", "mAP50-95")]
for ax, (c, t, yl) in zip(axes.ravel(), cols):
    for cfg in CFGS:
        for k in K:
            r = results(cfg, k); be = summ(cfg, k)["selection"]["best_epoch"]
            ax.plot(r["epoch"], r[c], color=COL[cfg], lw=0.9, alpha=0.8)
            i = int(np.where(r["epoch"] == be)[0][0])
            ax.plot(be, r[c][i], "o", ms=4.5, color=COL[cfg], mec="white", mew=0.8, zorder=5)
    ax.set_title(t, fontsize=10, weight="bold", loc="left")
    ax.set_xlabel("Epoch"); ax.set_ylabel(yl); style(ax); ax.set_xlim(0, 345)
    if "mAP" in c: ax.set_ylim(0, 1)
    if c == "val/box_loss": ax.set_ylim(1.0, 2.4)
    if c == "train/box_loss": ax.set_ylim(0.6, 2.6)
h = [Line2D([], [], color=COL[c], lw=1.6) for c in CFGS] + [Line2D([], [], ls="", marker="o", color="#555555", ms=5)]
fig.legend(h, [LAB[c] for c in CFGS] + ["Selected checkpoint"], loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=1.5)
fig.savefig("Figure_4.png", dpi=600, facecolor="white"); fig.savefig("Figure_4_preview.png", dpi=110, facecolor="white")
plt.close(fig)

# ---------------- Fig. 5
fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.3))
names = ["Meningioma", "Glioma", "Pituitary tumor"]
for c, ax in enumerate(axes.ravel()[:3]):
    for cfg in CFGS:
        cur = []
        for k in K:
            d = np.load(f"data/{cfg}/test{k}_seed42/test_curves.npz")
            cur.append(d["precision_recall"][c]); px = d["px"]
        cur = np.array(cur)
        ap = np.mean([summ(cfg, k)["test"]["per_class"][c]["AP50"] for k in K])
        ax.fill_between(px, cur.min(0), cur.max(0), color=COL[cfg], alpha=0.15, lw=0)
        ax.plot(px, cur.mean(0), color=COL[cfg], lw=1.6, label=f"{LAB[cfg]} (AP50 {ap:.3f})")
    ax.set_title(f"({'abc'[c]}) {names[c]}", fontsize=10, weight="bold", loc="left")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); style(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=8.8)
ax = axes[1, 1]
for cfg in CFGS:
    cur = []
    for k in K:
        d = np.load(f"data/{cfg}/test{k}_seed42/test_curves.npz")
        cur.append(d["f1_confidence"].mean(0)); px = d["px"]
    cur = np.array(cur); m = cur.mean(0)
    ax.fill_between(px, cur.min(0), cur.max(0), color=COL[cfg], alpha=0.15, lw=0)
    ax.plot(px, m, color=COL[cfg], lw=1.6, label=f"{LAB[cfg]} (max F1 {m.max():.2f})")
ax.set_title("(d) All classes", fontsize=10, weight="bold", loc="left")
ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1 score"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); style(ax)
ax.legend(loc="lower left", frameon=False, fontsize=8.8)
fig.tight_layout(h_pad=1.4, w_pad=1.5)
fig.savefig("Figure_5.png", dpi=600, facecolor="white"); fig.savefig("Figure_5_preview.png", dpi=110, facecolor="white")
plt.close(fig)

# ---------------- Fig. 6 (rotation 1, fold-specific operating threshold)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
labels_t = ["Meningioma", "Glioma", "Pituitary\ntumor"]
labels_p = ["Meningioma", "Glioma", "Pituitary\ntumor", "Missed"]
info = {}
for ax, cfg in zip(axes, CFGS):
    G, by = load(cfg); op = summ(cfg, 1)["test"]["operating_conf"]
    M = cm(G, by, op)                      # rows predicted (+ missed), cols true (+ background)
    T = M[:, :3].T                         # rows true, cols predicted + missed
    F = T / T.sum(1, keepdims=True)
    info[cfg] = dict(op=op, T=T.tolist(), extra=M[:3, 3].tolist())
    ax.imshow(F, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{100 * F[i, j]:.1f}%\n({T[i, j]})", ha="center", va="center", fontsize=8.5,
                    color="white" if F[i, j] > 0.55 else "#111111")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels_p, fontsize=8.8)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels_t, fontsize=8.8)
    ax.set_xlabel("Prediction"); ax.set_ylabel("True class")
    ax.set_title(f"({'ab'[CFGS.index(cfg)]}) {LAB[cfg]}", fontsize=10, weight="bold", loc="left")
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(w_pad=2.0)
fig.savefig("Figure_6.png", dpi=600, facecolor="white"); fig.savefig("Figure_6_preview.png", dpi=110, facecolor="white")
json.dump(info, open("fig6_info.json", "w"), indent=1)
print(json.dumps(info))
