"""Обгортки над Ultralytics (YOLO11, RT-DETR) під спільний протокол і спільну оцінку."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import common as c

os.environ.setdefault("WANDB_MODE", "disabled")


def setup(wandb=False):
    """Вимикає інтеграції, які на Kaggle можуть зависнути на логіні."""
    from ultralytics import settings
    wanted = {"wandb": bool(wandb), "tensorboard": False, "mlflow": False, "clearml": False,
              "comet": False, "dvc": False, "neptune": False}
    settings.update({k: v for k, v in wanted.items() if k in settings})
    if wandb:
        os.environ["WANDB_MODE"] = "online"


def gpu_devices():
    import torch
    n = torch.cuda.device_count()
    return list(range(n)) if n > 1 else (0 if n == 1 else "cpu")


def write_data_yaml(prep, out_yaml, train_every=1):
    """data.yaml з абсолютними шляхами; train_every>1 бере кожен k-й кадр prepared/train (txt-список)."""
    prep, out_yaml = Path(prep), Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    train = str(prep / "train" / "images")
    if train_every > 1:
        meta = c.load_json(prep / "train" / "annotations_coco.json")
        frames = sorted(meta["images"], key=lambda i: (i["video"], i["frame_id"]))
        by_clip = {}
        for im in frames:
            by_clip.setdefault(im["video"], []).append(im)
        files = [str(prep / "train" / im["file_name"]) for v in by_clip.values() for im in v[::train_every]]
        lst = out_yaml.with_suffix(".train.txt")
        lst.write_text("\n".join(files) + "\n")
        train = str(lst)
    cfg = {"path": str(prep), "train": train, "val": str(prep / "val" / "images"),
           "nc": len(c.CLASSES), "names": list(c.CLASSES)}
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_yaml


def _pkg_cfg(rel):
    import ultralytics
    return Path(ultralytics.__file__).parent / "cfg" / "models" / rel


def write_yolo11_p2_yaml(path):
    """YOLO11 з додатковою головою P2/4 (stride 4) для дрібних об'єктів.

    Backbone і перші шари голови збігаються з yolo11.yaml за індексами, тому
    .load("yolo11s.pt") переносить ці ваги; нові шари P2 ініціалізуються випадково.
    """
    base = yaml.safe_load(_pkg_cfg("11/yolo11.yaml").read_text())
    head = base["head"][:6]
    head += [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [128, False]],
        [-1, 1, "Conv", [128, 3, 2]],
        [[-1, 16], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],
        [[19, 22, 25, 28], 1, "Detect", ["nc"]],
    ]
    cfg = {"nc": len(c.CLASSES), "scales": base["scales"], "backbone": base["backbone"], "head": head}
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(path)


def write_rtdetr_yaml(path, nq=300, nd=100, variant="l"):
    """RT-DETR з іншою кількістю запитів nq і denoising-запитів nd (nd=0 вимикає denoising).

    Запити в RT-DETR беруться top-k з енкодера (learnt_init_query=False), тому ваги
    rtdetr-l.pt переносяться без змін при будь-якому nq.
    """
    cfg = yaml.safe_load(_pkg_cfg(f"rt-detr/rtdetr-{variant}.yaml").read_text())
    cfg["nc"] = len(c.CLASSES)
    last = cfg["head"][-1]
    last[3] = ["nc", 256, nq, 4, 8, 6, 1024, 0.0, None, -1, nd]
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(path)


def train(model_cfg, weights, data_yaml, project, name, rtdetr=False, **args):
    """Навчання з автоматичним відновленням після обриву сесії Kaggle.

    model_cfg: yaml архітектури або None (тоді береться weights як є).
    Повертає теку запуску (project/name).
    """
    from ultralytics import RTDETR, YOLO
    cls = RTDETR if rtdetr else YOLO
    run = Path(project) / name
    epochs = args.get("epochs")
    if _unfinished(run, epochs):
        cls(str(run / "weights" / "last.pt")).train(resume=True)
        return run
    if (run / "weights" / "best.pt").exists() and epochs and len(results_csv(run)) >= epochs:
        print(f"{run} уже навчено, пропускаю")
        return run
    model = cls(str(model_cfg)).load(weights) if model_cfg else cls(weights)
    model.train(data=str(data_yaml), project=str(project), name=name, exist_ok=True,
                seed=c.SEED, **args)
    return run


def _unfinished(run, epochs):
    csv = Path(run) / "results.csv"
    last = Path(run) / "weights" / "last.pt"
    if not csv.exists() or not last.exists() or epochs is None:
        return False
    return len(pd.read_csv(csv)) < epochs


def results_csv(run):
    df = pd.read_csv(Path(run) / "results.csv")
    df.columns = [col.strip() for col in df.columns]
    return df


def _to_dets(r, image_id, sx, sy):
    b = r.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy().astype(np.float64)
    xywh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], axis=1)
    xywh = c.to_original(xywh, sx, sy)
    return [{"image_id": image_id, "category_id": int(k) + 1, "bbox": [round(float(v), 2) for v in box],
             "score": round(float(s), 5)}
            for box, s, k in zip(xywh, b.conf.cpu().numpy(), b.cls.cpu().numpy())]


def predict_prepared(model, prep, split="val", imgsz=960, conf=0.001, iou=0.7, max_det=c.MAX_DETS,
                     batch=16, augment=False, every=1):
    """Прогноз на prepared/<split> -> COCO-детекції в оригінальних координатах + середня швидкість."""
    prep = Path(prep)
    meta = c.load_json(prep / split / "annotations_coco.json")["images"][::every]
    dets, speed = [], []
    for i in range(0, len(meta), 256):
        chunk = meta[i:i + 256]
        files = [str(prep / split / im["file_name"]) for im in chunk]
        res = model.predict(files, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, batch=batch,
                            augment=augment, verbose=False, stream=True)
        for im, r in zip(chunk, res):
            dets += _to_dets(r, im["id"], im["scale_x"], im["scale_y"])
            speed.append(sum(r.speed.values()))
    return dets, float(np.mean(speed)) if speed else 0.0


def predict_arrays(model, imgs, imgsz=960, conf=0.001, iou=0.7, max_det=c.MAX_DETS, batch=16, augment=False):
    """Прогноз на списку BGR-масивів. Повертає [(xyxy, scores, cls0)] у координатах кожного масиву."""
    out = []
    for i in range(0, len(imgs), batch):
        for r in model.predict(imgs[i:i + batch], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                               batch=batch, augment=augment, verbose=False):
            b = r.boxes
            out.append((b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                        b.cls.cpu().numpy().astype(np.int64)))
    return out


def measure_speed(model, prep, imgsz=960, n=100, warmup=10):
    """Середній час на кадр (мс): batch=1, препроцесинг + інференс + NMS, як у tv_models.measure_speed."""
    prep = Path(prep)
    files = [str(prep / "val" / im["file_name"]) for im in c.load_json(prep / "val" / "annotations_coco.json")["images"]]
    times = []
    for i in range(n + warmup):
        r = model.predict(files[i % len(files)], imgsz=imgsz, conf=0.001, max_det=c.MAX_DETS, verbose=False)[0]
        if i >= warmup:
            times.append(sum(r.speed.values()))
    return float(np.mean(times))


def patch_tal_topk(topk):
    """Змінює кількість кандидатів TaskAlignedAssigner (за замовчуванням 10) для нових моделей.

    Працює лише в поточному процесі, тому таке навчання запускайте на одному GPU (device=0):
    DDP стартує окремі процеси без цього патчу.
    """
    import ultralytics.nn.tasks as tasks
    from ultralytics.utils.loss import v8DetectionLoss
    tasks.DetectionModel.init_criterion = lambda self: v8DetectionLoss(self, tal_topk=topk)
