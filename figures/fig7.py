import sys, json
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
from cmlib import load

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9})
import os
IMG = os.environ.get("BENCHMARK_IMAGES", "benchmark_images") + "/"  # folder with the 100 fold-1 PNG slices
NAMES = ["Meningioma", "Glioma", "Pituitary tumor"]
GT_C, PR_C = "#e34948", "#2a9df4"
data = {}
for cfg in ("adopted", "yolov12n"):
    G, by = load(cfg)
    op = json.load(open(f"data/{cfg}/test1_seed42/summary.json"))["test"]["operating_conf"]
    data[cfg] = (G, by, op)

def panel(ax, f, cfg):
    G, by, op = data[cfg]
    im = Image.open(IMG + f).convert("L")
    assert im.size == (640, 640), im.size
    ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
    g = G[f][0]
    ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec=GT_C, lw=1.6, ls=(0, (4, 2))))
    for p in by.get(f, []):
        if p[5] > op:
            ax.add_patch(Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1], fill=False, ec=PR_C, lw=1.6))
            ty = p[1] - 6 if p[1] > 40 else p[3] + 30
            ax.text(p[0], ty, f"{NAMES[int(p[4])]} {p[5]:.2f}", color="white", fontsize=7.5, va="bottom" if p[1] > 40 else "top",
                    bbox=dict(boxstyle="square,pad=0.15", fc=PR_C, ec="none"))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def figure(cases, out, dpi, crop=40):
    n = len(cases)
    W = 7.1; cw = (W - 0.3) / n; H = 2 * cw + 0.75
    fig, axes = plt.subplots(2, n, figsize=(W, H), gridspec_kw=dict(wspace=0.03, hspace=0.03))
    for j, (f, lab) in enumerate(cases):
        for i, cfg in enumerate(("adopted", "yolov12n")):
            ax = axes[i, j]
            panel(ax, f, cfg)
            ax.set_xlim(crop, 640 - crop); ax.set_ylim(640 - crop, crop)
        axes[0, j].set_title(lab, fontsize=9.5)
    axes[0, 0].set_ylabel("Adopted model", fontsize=9.5, weight="bold")
    axes[1, 0].set_ylabel("Native YOLOv12n", fontsize=9.5, weight="bold")
    h = [Line2D([], [], color=GT_C, lw=1.8, ls=(0, (4, 2))), Line2D([], [], color=PR_C, lw=1.8)]
    fig.legend(h, ["Ground truth", "Prediction"], loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.52, 0.0))
    fig.subplots_adjust(left=0.3 / W, right=0.998, top=1 - 0.28 / H, bottom=0.42 / H)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    cases = [("0132.png", "(a) Meningioma"), ("2315.png", "(b) Glioma"), ("2244.png", "(c) Glioma"), ("1922.png", "(d) Glioma")]
    figure(cases, "Figure_7.png", 600)
    figure(cases, "Figure_7_preview.png", 150)
