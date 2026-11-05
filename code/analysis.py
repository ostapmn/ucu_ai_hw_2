"""Аналіз помилок, візуалізація збоїв і реєстр результатів для звіту."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as c

ERROR_TYPES = ["tp", "cls", "loc", "dup", "bg", "ignored", "other"]


def _iou(a, b):
    """IoU між наборами xywh (N,4) і (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = (np.minimum(ax1[:, None], bx1) - np.maximum(ax0[:, None], bx0)).clip(0)
    ih = (np.minimum(ay1[:, None], by1) - np.maximum(ay0[:, None], by0)).clip(0)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def _inside(a, b):
    """Частка площі детекції a, що лежить у регіоні b (як iscrowd у pycocotools)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = (np.minimum(a[:, 0:1] + a[:, 2:3], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0:1], b[:, 0])).clip(0)
    ih = (np.minimum(a[:, 1:2] + a[:, 3:4], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1:2], b[:, 1])).clip(0)
    return iw * ih / np.maximum((a[:, 2] * a[:, 3])[:, None], 1e-9)


def size_bucket(area):
    area = np.asarray(area)
    return np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))


def error_breakdown(gt_path, dets, score_thr=0.3, iou_thr=0.5, raw_ann_path=None):
    """Спрощений TIDE на фіксованому порозі впевненості.

    Кожна детекція з score>=score_thr отримує один тип:
      tp - правильна; cls - IoU>=0.5 з об'єктом іншого класу; loc - той самий клас, 0.1<=IoU<0.5;
      dup - дублікат уже знайденого об'єкта; ignored - лежить в ignore-регіоні (не штрафується);
      bg - фон (IoU<0.1 з усіма); other - решта (інший клас і погана локалізація).
    raw_ann_path: сирий annotations_coco.json змагання, звідки береться occlusion (0/1/2) за id анотації.
    Повертає (таблиця детекцій, таблиця GT з ознакою found, матриця плутанини класів).
    """
    gt = c.load_json(gt_path)
    occ = {}
    if raw_ann_path is not None:
        occ = {a["id"]: a.get("occlusion", -1) for a in c.load_json(raw_ann_path)["annotations"]}
    by_img = {}
    for a in gt["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    dets_by_img = {}
    for d in dets:
        if d["score"] >= score_thr:
            dets_by_img.setdefault(d["image_id"], []).append(d)
    det_rows, gt_rows = [], []
    confusion = np.zeros((len(c.CLASSES), len(c.CLASSES)), dtype=int)
    for im in gt["images"]:
        anns = by_img.get(im["id"], [])
        obj = [a for a in anns if not a["iscrowd"]]
        crowd = np.array([a["bbox"] for a in anns if a["iscrowd"]], dtype=float).reshape(-1, 4)
        g = np.array([a["bbox"] for a in obj], dtype=float).reshape(-1, 4)
        gc = np.array([a["category_id"] for a in obj])
        ds = sorted(dets_by_img.get(im["id"], []), key=lambda d: -d["score"])
        b = np.array([d["bbox"] for d in ds], dtype=float).reshape(-1, 4)
        dc = np.array([d["category_id"] for d in ds])
        iou = _iou(b, g)
        ins = _inside(b, crowd)
        matched = np.zeros(len(g), bool)
        for k, d in enumerate(ds):
            same = gc == dc[k]
            iou_same = np.where(same & ~matched, iou[k], 0)
            if len(g) and iou_same.max() >= iou_thr:
                j = int(iou_same.argmax())
                matched[j] = True
                kind = "tp"
            elif len(g) and np.where(same, iou[k], 0).max() >= iou_thr:
                kind = "dup"
            elif len(g) and np.where(~same, iou[k], 0).max() >= iou_thr:
                kind = "cls"
                j = int(np.where(~same, iou[k], 0).argmax())
                confusion[gc[j] - 1, dc[k] - 1] += 1
            elif len(crowd) and ins[k].max() >= 0.5:
                kind = "ignored"
            elif len(g) and np.where(same, iou[k], 0).max() >= 0.1:
                kind = "loc"
            elif len(g) and iou[k].max() >= 0.1:
                kind = "other"
            else:
                kind = "bg"
            det_rows.append((im["id"], c.CLASSES[dc[k] - 1], d["score"], b[k, 2] * b[k, 3], kind))
        for j, a in enumerate(obj):
            gt_rows.append((im["id"], im.get("video"), c.CLASSES[gc[j] - 1], a["bbox"][2] * a["bbox"][3],
                            occ.get(a["id"], a.get("occlusion", -1)), bool(matched[j])))
    det_df = pd.DataFrame(det_rows, columns=["image_id", "cls", "score", "area", "kind"])
    gt_df = pd.DataFrame(gt_rows, columns=["image_id", "video", "cls", "area", "occlusion", "found"])
    det_df["size"], gt_df["size"] = size_bucket(det_df.area), size_bucket(gt_df.area)
    conf = pd.DataFrame(confusion, index=c.CLASSES, columns=c.CLASSES)
    return det_df, gt_df, conf


def summarize_errors(det_df, gt_df):
    """Короткі таблиці: розподіл типів хибних детекцій і recall за розміром/класом."""
    kinds = det_df.kind.value_counts(normalize=True).reindex(ERROR_TYPES).fillna(0).round(3)
    recall_size = gt_df.groupby("size").found.mean().reindex(["small", "medium", "large"]).round(3)
    recall_cls = gt_df.groupby("cls").found.mean().reindex(c.CLASSES).round(3)
    return kinds, recall_size, recall_cls


def worst_frames(gt_df, det_df, k=6):
    """Кадри з найбільшою кількістю пропущених об'єктів і хибних спрацювань."""
    missed = gt_df[~gt_df.found].groupby("image_id").size().rename("missed")
    fp = det_df[det_df.kind.isin(["bg", "cls", "loc", "dup", "other"])].groupby("image_id").size().rename("fp")
    t = pd.concat([missed, fp], axis=1).fillna(0).astype(int)
    t["total"] = t.missed + t.fp
    return t.sort_values("total", ascending=False).head(k)


def draw_errors(ax, image_path, gt_anns, dets, score_thr=0.3, iou_thr=0.5, crop=None):
    """GT зеленим (знайдені) / жовтим (пропущені), FP червоним, ignore сірим."""
    from matplotlib.patches import Rectangle
    from PIL import Image
    img = np.asarray(Image.open(image_path).convert("RGB"))
    ax.imshow(img)
    obj = [a for a in gt_anns if not a["iscrowd"]]
    g = np.array([a["bbox"] for a in obj], dtype=float).reshape(-1, 4)
    gc = np.array([a["category_id"] for a in obj])
    ds = sorted([d for d in dets if d["score"] >= score_thr], key=lambda d: -d["score"])
    b = np.array([d["bbox"] for d in ds], dtype=float).reshape(-1, 4)
    iou = _iou(b, g)
    matched = np.zeros(len(g), bool)
    for k, d in enumerate(ds):
        cand = np.where((gc == d["category_id"]) & ~matched, iou[k], 0) if len(g) else np.zeros(0)
        if len(g) and cand.max() >= iou_thr:
            matched[int(cand.argmax())] = True
        else:
            x, y, w, h = d["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1))
    for a in gt_anns:
        x, y, w, h = a["bbox"]
        if a["iscrowd"]:
            ax.add_patch(Rectangle((x, y), w, h, color="gray", alpha=0.35))
    for j, a in enumerate(obj):
        x, y, w, h = a["bbox"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime" if matched[j] else "yellow", lw=1))
    if crop is not None:
        x0, y0, x1, y1 = crop
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax.axis("off")


def log_result(results_dir, name, family, metrics, **extra):
    """Зберігає один рядок результату (results/<name>.json) для зведеної таблиці у 07_summary."""
    row = {"name": name, "family": family, "time": time.strftime("%Y-%m-%d %H:%M"),
           **{k: v for k, v in metrics.items() if k != "per_class"}, **extra}
    if "per_class" in metrics:
        row.update({f"AP_{k}": v for k, v in metrics["per_class"].items()})
    c.save_json(row, Path(results_dir) / f"{name}.json")
    return row


def load_results(*dirs):
    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            rows.append(c.load_json(p))
    return pd.DataFrame(rows)


def all_results(P):
    """Усі збережені результати: поточний запуск, code/results у репо і виходи інших ноутбуків на Kaggle."""
    dirs = [Path(P.results), c.ROOT / "code" / "results"]
    if P.kaggle:
        dirs += sorted(Path("/kaggle/input").glob("*/runs/results")) + sorted(Path("/kaggle/input").glob("*/*/runs/results"))
    df = load_results(*[d for d in dirs if d.exists()])
    if len(df):
        df = df.sort_values("time").drop_duplicates("name", keep="last").reset_index(drop=True)
        ap = [col for col in df.columns if col.startswith(("AP", "mAP", "AR"))]
        df[ap] = df[ap].where(df[ap] >= 0)
    return df


def map_by_density(gt_path, dets, bins=(0, 20, 40, 80, 1000)):
    """mAP окремо для кадрів з різною кількістю об'єктів (щільність сцени)."""
    gt = c.load_json(gt_path)
    n = pd.Series(0, index=[im["id"] for im in gt["images"]])
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            n[a["image_id"]] += 1
    groups = pd.cut(n, bins)
    rows = {}
    for g, ids in n.groupby(groups, observed=True).groups.items():
        keep = set(ids)
        r = c.evaluate(gt_path, [d for d in dets if d["image_id"] in keep], image_ids=list(ids))
        rows[str(g)] = {"frames": len(ids), "mAP": r["mAP"], "AP50": r["AP50"], "AP_small": r["AP_small"]}
    return pd.DataFrame(rows).T


def md_table(df, floatfmt=3):
    """DataFrame -> Markdown без залежності від tabulate."""
    df = df.copy()
    cols = [str(x) for x in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        vals = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else
                ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def error_report(gt_path, dets, val_root, raw_ann_path, images_dir, prefix, score_thr=0.3, k=4):
    """Повний аналіз помилок однієї моделі: таблиці + матриця плутанини + найгірші кадри.

    Рисунки зберігаються в images_dir як <prefix>_confusion.png і <prefix>_failures.png.
    """
    import matplotlib.pyplot as plt
    from IPython.display import display
    d_df, g_df, conf = error_breakdown(gt_path, dets, score_thr=score_thr, raw_ann_path=raw_ann_path)
    kinds, rec_size, rec_cls = summarize_errors(d_df, g_df)
    display(kinds.to_frame(f"частка детекцій (score ≥ {score_thr})").T)
    display(rec_size.to_frame("recall").T)
    display(rec_cls.to_frame("recall").T)
    occ = g_df.occlusion.map({0: "немає", 1: "часткове", 2: "сильне"}).fillna("невідомо")
    display(g_df.groupby(occ).found.mean().to_frame("recall за перекриттям").T)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    norm = conf.div(conf.sum(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(norm.values, cmap="Blues")
    ax.set_xticks(range(len(c.CLASSES)), c.CLASSES, rotation=45)
    ax.set_yticks(range(len(c.CLASSES)), c.CLASSES)
    ax.set_xlabel("передбачений клас")
    ax.set_ylabel("справжній клас")
    ax.set_title("плутанина класів (частка рядка)")
    fig.colorbar(im)
    fig.savefig(Path(images_dir) / f"{prefix}_confusion.png", bbox_inches="tight", dpi=130)
    plt.show()

    gt = c.load_json(gt_path)
    meta = {im["id"]: im for im in gt["images"]}
    gt_by, dt_by = {}, {}
    for a in gt["annotations"]:
        gt_by.setdefault(a["image_id"], []).append(a)
    for d in dets:
        dt_by.setdefault(d["image_id"], []).append(d)
    worst = worst_frames(g_df, d_df, k=k)
    rows = (len(worst) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (iid, row) in zip(axes.ravel(), worst.iterrows()):
        im = meta[iid]
        draw_errors(ax, Path(val_root) / im["file_name"], gt_by.get(iid, []), dt_by.get(iid, []), score_thr=score_thr)
        ax.set_title(f"{im['video']} кадр {im['frame_id']}: пропущено {row.missed}, хибних {row.fp}")
    fig.suptitle("зелений - знайдено, жовтий - пропущено, червоний - хибна детекція, сірий - ignore")
    fig.savefig(Path(images_dir) / f"{prefix}_failures.png", bbox_inches="tight", dpi=110)
    plt.show()
    return d_df, g_df


def find_run_file(P, rel):
    """Шукає файл runs/<rel> у поточному запуску, а на Kaggle ще й у виходах інших ноутбуків."""
    cands = [Path(P.out) / rel]
    if P.kaggle:
        cands += sorted(Path("/kaggle/input").glob(f"*/runs/{rel}")) + sorted(Path("/kaggle/input").glob(f"*/*/runs/{rel}"))
    return next((p for p in cands if p.exists()), None)
