import json, numpy as np
from collections import defaultdict

def iou(a, B):
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1]); x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)

def load(cfg, root="data"):
    gt = json.load(open(f"{root}/gt_fold1.json"))
    P = json.load(open(f"{root}/{cfg}/test1_seed42/test_eval/predictions.json"))
    by = defaultdict(list)
    for p in P:
        x, y, w, h = p["bbox"]
        by[p["file_name"]].append((x, y, x + w, y + h, p["category_id"] - 1, p["score"]))
    G = {f: [(g["x_min"] * 640, g["y_min"] * 640, g["x_max"] * 640, g["y_max"] * 640, g["class_id"]) for g in v] for f, v in gt.items()}
    return G, by

def cm(G, by, conf, thr=0.45):
    """Ultralytics rule for images with one GT box: the GT is matched to the detection (conf > conf) with the
    highest IoU above thr; unmatched detections are FP (background column)."""
    M = np.zeros((4, 4), int)
    for f, gts in G.items():
        d = np.array([p for p in by.get(f, []) if p[5] > conf]).reshape(-1, 6)
        g = gts[0]
        if len(d) == 0:
            M[3, g[4]] += 1; continue
        io = iou(np.array(g[:4]), d[:, :4])
        ok = np.where(io > thr)[0]
        if len(ok):
            j = ok[np.argmax(io[ok])]
            M[int(d[j, 4]), g[4]] += 1
            rest = [k for k in range(len(d)) if k != j]
        else:
            M[3, g[4]] += 1; rest = range(len(d))
        for k in rest:
            M[int(d[k, 4]), 3] += 1
    return M
