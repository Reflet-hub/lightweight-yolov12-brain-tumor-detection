import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9})
INK, MUTED = "#111111", "#444444"
TEST, VAL, TRAIN = "#1f5fa8", "#8fb8e6", "#e9e9e6"

fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 72); ax.set_ylim(0, 50); ax.axis("off")

def rbox(x, y, w, h, lw=1.2, r=0.8, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, ec=INK, fc=fc))

def box(x, y, w, h, title=None, lines=(), fs=9, tfs=9.5, tdy=2.0, ldy=1.75, lw=1.2):
    rbox(x, y, w, h, lw=lw)
    yy = y + h - tdy
    if title:
        ax.text(x + w / 2, yy, title, ha="center", va="center", fontsize=tfs, weight="bold", color=INK)
        yy -= ldy + 0.35
    for l in lines:
        ax.text(x + w / 2, yy, l, ha="center", va="center", fontsize=fs, color=INK)
        yy -= ldy

def arrow(x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=INK,
                                 shrinkA=0, shrinkB=0, linestyle=ls))

# dataset
box(0.5, 30.5, 12.0, 14.0, "Figshare dataset", ["3,064 MRI slices", "233 patients", "3 tumor classes", "5 official", "patient-level folds"])
arrow(12.5, 37.5, 14.5, 37.5)

# cross-validation grid
rbox(14.5, 25.0, 25.0, 24.5)
ax.text(27.0, 47.4, "Five-fold patient-level", ha="center", va="center", fontsize=9.5, weight="bold")
ax.text(27.0, 45.6, "cross-validation", ha="center", va="center", fontsize=9.5, weight="bold")
gx, gy, cw, ch = 21.4, 30.0, 3.4, 2.3
ax.text(gx - 0.5, gy + 5 * ch + 1.0, "Rotation", ha="right", va="center", fontsize=8, color=MUTED)
for j in range(5):
    ax.text(gx + j * cw + cw / 2, gy + 5 * ch + 1.0, f"Fold {j + 1}", ha="center", va="center", fontsize=7.5, color=MUTED)
for i in range(5):
    k = i + 1
    yy = gy + (4 - i) * ch
    ax.text(gx - 2.4, yy + ch / 2, f"{k}", ha="center", va="center", fontsize=8.5, color=INK)
    for j in range(5):
        f = j + 1
        role = "Test" if f == k else ("Val" if f == (k % 5) + 1 else "Train")
        fc = {"Test": TEST, "Val": VAL, "Train": TRAIN}[role]
        ax.add_patch(Rectangle((gx + j * cw, yy), cw, ch, fc=fc, ec="white", lw=1.5))
        ax.text(gx + j * cw + cw / 2, yy + ch / 2, role, ha="center", va="center", fontsize=7.5,
                color="white" if role == "Test" else INK)
ax.text(27.0, 28.0, "Test: fold k.  Validation (Val): fold k + 1.", ha="center", va="center", fontsize=7.8, color=MUTED)
ax.text(27.0, 26.4, "Training: the three other folds.", ha="center", va="center", fontsize=7.8, color=MUTED)
arrow(39.5, 37.5, 41.5, 37.5)

# training runs
rbox(41.5, 25.0, 30.0, 24.5)
ax.text(56.5, 47.4, "Training runs", ha="center", va="center", fontsize=9.5, weight="bold")
ax.text(56.5, 45.4, "All five rotations, seed 42", ha="center", va="center", fontsize=8.5, style="italic", color=MUTED)
for n, (t, s) in enumerate([("Adopted model", "Full fine-tuning, 746,015 parameters"),
                            ("Frozen backbone", "463,625 trainable parameters"),
                            ("Native YOLOv12n", "COCO-pretrained, 2,568,633 parameters")]):
    y0 = 40.0 - n * 4.4
    rbox(42.7, y0, 27.6, 3.9, lw=0.9, r=0.5)
    ax.text(56.5, y0 + 2.75, t, ha="center", va="center", fontsize=8.8, weight="bold")
    ax.text(56.5, y0 + 1.15, s, ha="center", va="center", fontsize=7.8, color=MUTED)
ax.text(56.5, 29.6, "Rotations 1 and 2 only", ha="center", va="center", fontsize=8.5, style="italic", color=MUTED)
ax.text(56.5, 27.8, "Seeds 0 and 1: adopted model and YOLOv12n", ha="center", va="center", fontsize=8.2)
ax.text(56.5, 26.2, "Three single-component ablations", ha="center", va="center", fontsize=8.2)

# evaluation
arrow(56.5, 25.0, 56.5, 22.5)
box(41.5, 11.5, 30.0, 11.0, "Test-fold evaluation", ["Precision, recall, mAP50, mAP50-95", "Class-wise AP and recall", "Patient-level bootstrap 95% CIs"])

# false-positive test
box(0.5, 11.5, 17.5, 11.0, "Tumor-free images", ["1,681 images from", "another public", "collection"])
arrow(18.0, 17.0, 21.0, 17.0)
box(21.0, 11.5, 18.5, 11.0, "Exploratory test", ["False alarms of the", "test-fold models of the", "adopted model and YOLOv12n"], fs=8.4)
arrow(27.0, 25.0, 30.25, 22.5, ls=(0, (3, 2)))

# deployment
rbox(0.5, 0.5, 71.0, 9.0, lw=1.0)
ax.text(36.0, 7.6, "Deployment (adopted model and YOLOv12n trained before cross-validation)", ha="center", va="center",
        fontsize=9.3, weight="bold")
steps = [("Training: folds 2–5", "Selection: fold 1"), ("FP32 TensorFlow Lite", "Export fidelity on fold 1"),
         ("Kaggle CPU and Redmi 14C smartphone", "Latency and memory")]
xs = [1.5, 22.0, 42.5]; ws = [18.0, 18.0, 28.0]
for (a, b), x, w in zip(steps, xs, ws):
    rbox(x, 1.3, w, 4.6, lw=0.9, r=0.5)
    ax.text(x + w / 2, 4.3, a, ha="center", va="center", fontsize=8.4, weight="bold")
    ax.text(x + w / 2, 2.6, b, ha="center", va="center", fontsize=7.8, color=MUTED)
arrow(19.5, 3.6, 22.0, 3.6); arrow(40.0, 3.6, 42.5, 3.6)
fig.savefig("Figure_1.png", dpi=600, facecolor="white")
fig.savefig("Figure_1_preview.png", dpi=150, facecolor="white")
