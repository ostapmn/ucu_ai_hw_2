import contextlib
import csv
import io
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
CATEGORIES = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
SEED = 42
LONG_SIDE = 960
TRAIN_STRIDE = 3
VAL_STRIDE = 2
MAX_DETS = 500
GRAY = 114
ROOT = Path(__file__).resolve().parent.parent


def _find_dir(base, marker, name=None, depth=4):
    """Шукає під base теку, у якій лежить marker (і, якщо задано, з іменем name)."""
    base = Path(base)
    if not base.exists():
        return None
    for d in range(depth + 1):
        for hit in sorted(base.glob("/".join(["*"] * d + [marker]))):
            root = hit.parent
            for _ in Path(marker).parts[:-1]:
                root = root.parent
            if name is None or root.name == name:
                return root
    return None


def paths(prep_name="prepared"):
    """Повертає теки проєкту.

    data  - сирі дані змагання (train/, val/, test/, test_images.csv);
    prep  - підготовлений датасет з 01_eda_prepare (prep_name: "prepared" або "prepared_1280");
    out   - куди писати ваги, прогнози й результати.
    Змінні середовища HW2_DATA, HW2_PREPARED, HW2_OUT мають пріоритет.
    """
    kaggle = Path("/kaggle/input").exists()
    data = os.environ.get("HW2_DATA")
    if data is None:
        data = (_find_dir("/kaggle/input", "test_images.csv") if kaggle else None) or ROOT / "data"
    prep = os.environ.get("HW2_PREPARED") if prep_name == "prepared" else None
    if prep is None:
        found = _find_dir("/kaggle/input", "val/annotations_eval.json", name=prep_name) if kaggle else None
        prep = found or (Path("/kaggle/working") / prep_name if kaggle else ROOT / prep_name)
    out = os.environ.get("HW2_OUT") or (Path("/kaggle/working/runs") if kaggle else ROOT / "runs")
    p = SimpleNamespace(root=ROOT, data=Path(data), prep=Path(prep), out=Path(out),
                        images=ROOT / "images", results=Path(out) / "results", kaggle=kaggle)
    p.out.mkdir(parents=True, exist_ok=True)
    p.results.mkdir(parents=True, exist_ok=True)
    return p


def restore_from_inputs(p, sub):
    """Kaggle: копіює runs/<sub> з попередньої версії ноутбука, доданої як Input, у /kaggle/working.

    Так навчання, обірване лімітом сесії, продовжується з last.pt у новій сесії.
    """
    if not p.kaggle:
        return []
    import shutil
    dst = p.out / sub
    copied = []
    for src in sorted(Path("/kaggle/input").glob(f"*/runs/{sub}")) + sorted(Path("/kaggle/input").glob(f"*/*/runs/{sub}")):
        for run in src.iterdir():
            if run.is_dir() and not (dst / run.name).exists():
                shutil.copytree(run, dst / run.name)
                copied.append(run.name)
    return copied


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def load_json(path):
    with open(path) as f:
        return json.load(f)


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


def _box_ann(a, sx, sy):
    x, y, w, h = scale_boxes(a["bbox"], sx, sy)[0]
    return {"id": a["id"], "image_id": a["image_id"], "category_id": a["category_id"],
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2), "iscrowd": a["iscrowd"]}


def build_split(data, out, split, stride, mask, long_side=LONG_SIDE):
    """Кожен stride-й кадр, довга сторона long_side, ignore-зони сірим (якщо mask).

    Пише images/, labels/ (YOLO), annotations_coco.json (координати зменшеного кадру,
    зі scale_x/scale_y) і для val ще annotations_eval.json в оригінальних координатах.
    """
    data, out = Path(data), Path(out)
    images, objects, ignore = load_coco(data / split / "annotations_coco.json")
    frames = select_frames(images, stride)
    root = out / split
    coco_images, coco_anns, eval_images, eval_anns = [], [], [], []
    for im in frames:
        clip, name = im["video"], Path(im["file_name"]).stem
        crowd = ignore.get(im["id"], [])
        boxes = [a["bbox"] for a in crowd] if mask else ()
        arr, sx, sy = load_resized(data / split / im["file_name"], long_side, ignore_boxes=boxes)
        img_path = root / "images" / clip / f"{name}.jpg"
        lbl_path = root / "labels" / clip / f"{name}.txt"
        for p in (img_path, lbl_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(img_path, quality=95)
        lbl_path.write_text("\n".join(to_yolo(objects.get(im["id"], []), im["width"], im["height"])))
        coco_images.append({"id": im["id"], "file_name": f"images/{clip}/{name}.jpg",
                            "width": arr.shape[1], "height": arr.shape[0], "video": clip,
                            "frame_id": im["frame_id"], "scale_x": sx, "scale_y": sy})
        eval_images.append({k: im[k] for k in ("id", "file_name", "width", "height", "video", "frame_id")})
        for a in objects.get(im["id"], []) + crowd:
            coco_anns.append(_box_ann(a, sx, sy))
            eval_anns.append({k: a[k] for k in ("id", "image_id", "category_id", "bbox", "area", "iscrowd")})
    (root / "annotations_coco.json").write_text(
        json.dumps({"images": coco_images, "annotations": coco_anns, "categories": CATEGORIES}))
    if split == "val":
        (root / "annotations_eval.json").write_text(
            json.dumps({"images": eval_images, "annotations": eval_anns, "categories": CATEGORIES}))


def build_test(data, out, long_side=LONG_SIDE):
    data, out = Path(data), Path(out)
    rows = []
    with open(data / "test_images.csv") as f:
        test = list(csv.DictReader(f))
    for r in test:
        arr, sx, sy = load_resized(data / r["file_name"], long_side)
        dst = out / "test" / "images" / r["clip"] / Path(r["file_name"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(dst, quality=95)
        rows.append([r["image_id"], r["clip"], r["frame"], r["width"], r["height"],
                     arr.shape[1], arr.shape[0], sx, sy])
    with open(out / "test" / "scales.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "clip", "frame", "width", "height", "new_width", "new_height", "scale_x", "scale_y"])
        w.writerows(rows)


def build_dataset(data, out, long_side=LONG_SIDE, train_stride=TRAIN_STRIDE, val_stride=VAL_STRIDE):
    """Будує train/val/test один раз; повторний виклик нічого не перезаписує."""
    out = Path(out)
    if not (out / "train" / "annotations_coco.json").exists():
        build_split(data, out, "train", train_stride, mask=True, long_side=long_side)
    if not (out / "val" / "annotations_coco.json").exists():
        build_split(data, out, "val", val_stride, mask=False, long_side=long_side)
    if not (out / "test" / "scales.csv").exists():
        build_test(data, out, long_side)
    (out / "data.yaml").write_text(
        f"train: train/images\nval: val/images\nnc: {len(CLASSES)}\nnames: {json.dumps(CLASSES)}\n")
    save_json({"long_side": long_side, "train_stride": train_stride, "val_stride": val_stride}, out / "info.json")


def evaluate(gt_path, detections, image_ids=None, per_class=False):
    """COCO-метрики на val в оригінальних координатах; ignore-регіони = iscrowd.

    per_class=True додає AP (0.5:0.95) окремо для кожного класу під ключем "per_class".
    """
    try:
        import faster_coco_eval
        faster_coco_eval.init_as_pycocotools()
    except ImportError:
        pass
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_max"]
    if not detections:
        res = dict.fromkeys(names, 0.0)
        if per_class:
            res["per_class"] = dict.fromkeys(CLASSES, 0.0)
        return res
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, MAX_DETS]
        if image_ids is not None:
            ev.params.imgIds = sorted(image_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = ev.stats
    res = dict(zip(names, [float(stats[i]) for i in (0, 1, 2, 3, 4, 5, 8)]))
    if per_class:
        prec = np.asarray(ev.eval["precision"])[:, :, :, 0, -1]
        cat_ids = list(ev.params.catIds)
        res["per_class"] = {}
        for k, cid in enumerate(cat_ids):
            p = prec[:, :, k]
            res["per_class"][CLASSES[cid - 1]] = float(p[p > -1].mean()) if (p > -1).any() else 0.0
    return res


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
