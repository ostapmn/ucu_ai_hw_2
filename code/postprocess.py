"""Стратегії інференсу та постобробки: тайли (SAHI-подібно), злиття (NMS / WBF),
ансамбль і темпоральне перерахування впевненості (бонус "Data Temporality").

Формат детекцій одного кадру: (boxes xyxy float32 [N,4] в оригінальних пікселях,
scores float32 [N], cls int64 [N], класи 0..7).
"""
import cv2
import numpy as np
import torch
import torchvision
from scipy.optimize import linear_sum_assignment

import common as c

EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


def cat(dets_list):
    dets_list = [d for d in dets_list if len(d[0])]
    if not dets_list:
        return EMPTY
    return tuple(np.concatenate([d[i] for d in dets_list]) for i in range(3))


def nms(dets, iou=0.6, max_det=c.MAX_DETS):
    b, s, k = dets
    if len(b) == 0:
        return EMPTY
    keep = torchvision.ops.batched_nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(),
                                       torch.from_numpy(k), iou).numpy()[:max_det]
    return b[keep], s[keep], k[keep]


def wbf(dets_list, width, height, iou=0.55, skip=0.001, weights=None, conf_type="avg", max_det=c.MAX_DETS):
    """Weighted Boxes Fusion (Solovyev et al., 2019) через пакет ensemble-boxes."""
    from ensemble_boxes import weighted_boxes_fusion
    scale = np.array([width, height, width, height], np.float32)
    boxes = [(d[0] / scale).clip(0, 1).tolist() for d in dets_list]
    scores = [d[1].tolist() for d in dets_list]
    labels = [d[2].tolist() for d in dets_list]
    b, s, k = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou,
                                    skip_box_thr=skip, conf_type=conf_type)
    order = np.argsort(-s)[:max_det]
    return (b[order] * scale).astype(np.float32), s[order].astype(np.float32), k[order].astype(np.int64)


def tile_grid(w, h, tile, overlap):
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in ys for x in xs]


def unscale(b, scale):
    """Бокси з масштабованого кадру назад в оригінальні пікселі; scale = (sx, sy) з common.load_resized."""
    sx, sy = scale
    return (b / np.array([sx, sy, sx, sy], np.float32)).astype(np.float32)


def tiled_predict(predict_fn, big, scale_big, tile=960, overlap=0.2, full=None, scale_full=None,
                  merge="nms", merge_iou=0.6, width=None, height=None):
    """SAHI-подібний інференс: тайли збільшеного кадру + (опційно) цілий кадр, злиття в оригінальних координатах.

    predict_fn(list_of_bgr, imgsz) -> [(xyxy, scores, cls)] у координатах кожного масиву.
    big: кадр, збільшений відносно навчального масштабу (наприклад, довга сторона 1920 замість 960),
    scale_big = (sx, sy) відносно оригіналу; full/scale_full - кадр у навчальному масштабі.
    Дрібні об'єкти на тайлах стають більшими, а цілий кадр зберігає великі об'єкти, які тайли ріжуть.
    """
    bh, bw = big.shape[:2]
    boxes = tile_grid(bw, bh, tile, overlap)
    crops = [np.ascontiguousarray(big[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes]
    parts = []
    for (x0, y0, _, _), (b, s, cl) in zip(boxes, predict_fn(crops, tile)):
        if len(b):
            parts.append((unscale(b + np.array([x0, y0, x0, y0], np.float32), scale_big), s, cl))
    if full is not None:
        b, s, cl = predict_fn([full], max(full.shape[:2]))[0]
        parts.append((unscale(b, scale_full), s, cl))
    if merge == "wbf":
        return wbf(parts, width, height, iou=merge_iou) if parts else EMPTY
    return nms(cat(parts), iou=merge_iou)


class MotionEstimator:
    """Оцінка зсуву камери між сусідніми кадрами: ORB + RANSAC, часткова афінна модель (4 ст. свободи)."""

    def __init__(self, width=640, n_features=1500):
        self.width = width
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev = None

    def __call__(self, img):
        """Повертає 3x3 матрицю, що переводить координати попереднього кадру в поточний."""
        h, w = img.shape[:2]
        s = self.width / w
        gray = cv2.cvtColor(cv2.resize(img, (self.width, round(h * s))), cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        H = np.eye(3)
        if self.prev is not None and des is not None and self.prev[1] is not None:
            matches = self.bf.match(self.prev[1], des)
            if len(matches) >= 12:
                src = np.float32([self.prev[0][m.queryIdx].pt for m in matches])
                dst = np.float32([kp[m.trainIdx].pt for m in matches])
                A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if A is not None and inl is not None and inl.sum() >= 10:
                    S = np.diag([s, s, 1.0])
                    H = np.linalg.inv(S) @ np.vstack([A, [0, 0, 1]]) @ S
        self.prev = (kp, des)
        return H


def warp_boxes(b, H):
    if len(b) == 0 or np.allclose(H, np.eye(3)):
        return b
    pts = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]], axis=1).reshape(-1, 2)
    pts = np.c_[pts, np.ones(len(pts))] @ H.T
    pts = (pts[:, :2] / pts[:, 2:3]).reshape(-1, 4, 2)
    return np.concatenate([pts.min(axis=1), pts.max(axis=1)], axis=1).astype(np.float32)


def box_iou(a, b):
    return torchvision.ops.box_iou(torch.from_numpy(a).float(), torch.from_numpy(b).float()).numpy()


def temporal_rescore(frames, motions=None, iou_thr=0.3, link_min=0.01, max_gap=1, w=0.5,
                     single_penalty=0.7, fill_gap=True, fill_factor=0.5, max_det=c.MAX_DETS):
    """Зв'язує детекції сусідніх кадрів у треки і перераховує впевненість.

    frames: список детекцій послідовних кадрів одного кліпу (у порядку часу).
    motions: список 3x3 матриць "кадр t-1 -> кадр t" (motions[0] ігнорується); None = без компенсації.
    Алгоритм:
      1) на кожному кадрі активні треки зсуваються компенсацією руху камери;
      2) детекції (score >= link_min) зіставляються з треками того самого класу угорським
         алгоритмом за IoU >= iou_thr; трек живе max_gap кадрів без підтвердження;
      3) нова впевненість: s' = (1-w)*s + w*mean(scores треку); для треків з однієї
         детекції s' = s * single_penalty (одиночні спрацювання частіше хибні);
      4) fill_gap: якщо трек пропустив рівно один кадр, туди додається інтерпольований бокс
         з впевненістю fill_factor * min(сусідніх).
    """
    T = len(frames)
    track_of = [np.full(len(f[0]), -1, np.int64) for f in frames]
    tracks = {}
    next_id = 0
    fills = []
    for t in range(T):
        b, s, k = frames[t]
        if motions is not None and t > 0:
            for tr in tracks.values():
                tr["box"] = warp_boxes(tr["box"][None], motions[t])[0]
        alive = list(tracks)
        cand = np.where(s >= link_min)[0]
        if alive and len(cand):
            tb = np.stack([tracks[i]["box"] for i in alive])
            tc = np.array([tracks[i]["cls"] for i in alive])
            iou = box_iou(tb, b[cand])
            iou[tc[:, None] != k[cand][None, :]] = 0
            rows, cols = linear_sum_assignment(-iou)
            for r, col in zip(rows, cols):
                if iou[r, col] >= iou_thr:
                    tid, j = alive[r], cand[col]
                    tr = tracks[tid]
                    if fill_gap and t - tr["last"] == 2:
                        fills.append((t - 1, (tr["box"] + b[j]) / 2, tr["cls"], (t - 2, tr["last_j"]), (t, j)))
                    track_of[t][j] = tid
                    tr.update(box=b[j].copy(), last=t, last_j=j)
        for j in cand:
            if track_of[t][j] < 0:
                tracks[next_id] = {"cls": int(k[j]), "box": b[j].copy(), "last": t, "last_j": j}
                track_of[t][j] = next_id
                next_id += 1
        tracks = {i: tr for i, tr in tracks.items() if t - tr["last"] <= max_gap}

    sums, length = {}, {}
    for t in range(T):
        for j, tid in enumerate(track_of[t]):
            if tid >= 0:
                sums[tid] = sums.get(tid, 0.0) + float(frames[t][1][j])
                length[tid] = length.get(tid, 0) + 1
    new_scores = []
    for t in range(T):
        s, tids = frames[t][1], track_of[t]
        s2 = s * single_penalty
        linked = tids >= 0
        if linked.any():
            ln = np.array([length[i] for i in tids[linked]])
            mean = np.array([sums[i] / length[i] for i in tids[linked]], np.float32)
            s2[linked] = np.where(ln > 1, (1 - w) * s[linked] + w * mean, s[linked] * single_penalty)
        new_scores.append(s2.astype(np.float32))

    extra = [[] for _ in range(T)]
    for t, box, cls, (tp, jp), (tn, jn) in fills:
        extra[t].append((box, fill_factor * min(new_scores[tp][jp], new_scores[tn][jn]), cls))
    out = []
    for t in range(T):
        b, _, k = frames[t]
        s2 = new_scores[t]
        if extra[t]:
            b = np.vstack([b, np.array([e[0] for e in extra[t]], np.float32)])
            s2 = np.concatenate([s2, np.array([e[1] for e in extra[t]], np.float32)])
            k = np.concatenate([k, np.array([e[2] for e in extra[t]], np.int64)])
        order = np.argsort(-s2)[:max_det]
        out.append((b[order], s2[order], k[order]))
    return out


def to_coco(dets, image_id):
    b, s, k = dets
    xywh = np.c_[b[:, :2], b[:, 2:] - b[:, :2]]
    return [{"image_id": image_id, "category_id": int(kk) + 1, "bbox": [round(float(v), 2) for v in bb],
             "score": round(float(ss), 5)} for bb, ss, kk in zip(xywh, s, k)]


def to_submission(dets, width, height):
    """Кортежі (cls, score, x, y, w, h) для common.write_submission, бокси обрізані межами кадру."""
    b, s, k = dets
    b = b.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width - 1e-3)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height - 1e-3)
    wh = b[:, 2:] - b[:, :2]
    ok = (wh[:, 0] > 0.5) & (wh[:, 1] > 0.5)
    return [(int(kk), float(np.clip(ss, 0, 1)), float(x), float(y), float(ww), float(hh))
            for (x, y), (ww, hh), ss, kk in zip(b[ok, :2], wh[ok], s[ok], k[ok])]
