import contextlib
import csv
import io
import json
import os
import random

import numpy as np
from PIL import Image

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
SEED = 42
LONG_SIDE = 960
TRAIN_STRIDE = 3
VAL_STRIDE = 2
MAX_DETS = 500
GRAY = 114


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_coco(path):
    with open(path) as f:
        coco = json.load(f)
    objects, ignore = {}, {}
    for ann in coco["annotations"]:
        target = ignore if ann["iscrowd"] else objects
        target.setdefault(ann["image_id"], []).append(ann)
    return coco["images"], objects, ignore


def select_frames(images, stride):
    by_video = {}
    for im in sorted(images, key=lambda i: (i["video"], i["frame_id"])):
        by_video.setdefault(im["video"], []).append(im)
    return [im for frames in by_video.values() for im in frames[::stride]]


def load_resized(path, long_side=LONG_SIDE, ignore_boxes=()):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    k = long_side / max(w, h)
    new_w, new_h = round(w * k), round(h * k)
    arr = np.asarray(img.resize((new_w, new_h), Image.BILINEAR)).copy()
    sx, sy = new_w / w, new_h / h
    for x, y, bw, bh in ignore_boxes:
        x0, y0 = max(int(np.floor(x * sx)), 0), max(int(np.floor(y * sy)), 0)
        x1, y1 = int(np.ceil((x + bw) * sx)), int(np.ceil((y + bh) * sy))
        arr[y0:y1, x0:x1] = GRAY
    return arr, sx, sy


def scale_boxes(boxes, sx, sy):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return boxes * np.array([sx, sy, sx, sy])


def to_original(boxes, sx, sy):
    return scale_boxes(boxes, 1 / sx, 1 / sy)


def to_yolo(anns, width, height):
    lines = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        lines.append(f"{ann['category_id'] - 1} {cx:.6f} {cy:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
    return lines


def evaluate(gt_path, detections, image_ids=None):
    try:
        import faster_coco_eval
        faster_coco_eval.init_as_pycocotools()
    except ImportError:
        pass
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_max"]
    if not detections:
        return dict.fromkeys(names, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, MAX_DETS]
        if image_ids is not None:
            ev.params.imgIds = sorted(image_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = ev.stats
    return dict(zip(names, [float(stats[i]) for i in (0, 1, 2, 3, 4, 5, 8)]))


def write_submission(predictions, test_csv, out_path):
    with open(test_csv) as f:
        ids = [row["image_id"] for row in csv.DictReader(f)]
    unknown = set(predictions) - set(ids)
    if unknown:
        raise KeyError(f"{len(unknown)} unknown image_id, e.g. {sorted(unknown)[0]}")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        for image_id in ids:
            dets = sorted(predictions.get(image_id, []), key=lambda d: -d[1])[:MAX_DETS]
            text = " ".join(f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" for c, s, x, y, w, h in dets)
            writer.writerow([image_id, text])


def check_submission(path, test_csv):
    with open(test_csv) as f:
        sizes = {r["image_id"]: (int(r["width"]), int(r["height"])) for r in csv.DictReader(f)}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["image_id", "PredictionString"], "header"
    assert [r["image_id"] for r in rows] == list(sizes), "ids or order differ from test_csv"
    for r in rows:
        values = r["PredictionString"].split()
        assert len(values) % 6 == 0, f"{r['image_id']}: not groups of 6"
        assert len(values) // 6 <= MAX_DETS, f"{r['image_id']}: more than {MAX_DETS} detections"
        width, height = sizes[r["image_id"]]
        for i in range(0, len(values), 6):
            c, s, x, y, w, h = values[i:i + 6]
            assert int(c) in range(len(CLASSES)), f"{r['image_id']}: class {c}"
            assert 0 <= float(s) <= 1, f"{r['image_id']}: score {s}"
            assert float(w) > 0 and float(h) > 0, f"{r['image_id']}: empty box"
            assert float(x) < width and float(y) < height, f"{r['image_id']}: box outside image"
    return len(rows)
