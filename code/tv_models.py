"""Torchvision-детектори (Faster R-CNN, RetinaNet) під спільний протокол.

Усі моделі навчаються на prepared/train (960 px, ignore замальовані сірим),
валідуються на prepared/val, а прогнози переводяться в оригінальні координати
і рахуються common.evaluate за тими самими COCO-порогами, що й у змаганні.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import common as c

NUM_CLASSES = len(c.CLASSES) + 1


class PreparedDataset(Dataset):
    """Кадри з prepared/<split>; таргети в координатах зменшеного кадру (xyxy)."""

    def __init__(self, prep, split, train=False, every=1):
        prep = Path(prep)
        meta = c.load_json(prep / split / "annotations_coco.json")
        self.root = prep / split
        self.images = meta["images"][::every]
        self.train = train
        self.anns = {}
        for a in meta["annotations"]:
            if not a["iscrowd"]:
                self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = decode_jpeg(read_file(str(self.root / im["file_name"])), mode=ImageReadMode.RGB)
        img = img.float().div_(255)
        h, w = img.shape[1:]
        boxes = np.array([a["bbox"] for a in self.anns.get(im["id"], [])], dtype=np.float32).reshape(-1, 4)
        labels = np.array([a["category_id"] for a in self.anns.get(im["id"], [])], dtype=np.int64)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        boxes, labels = torch.from_numpy(boxes[keep]), torch.from_numpy(labels[keep])
        if self.train and torch.rand(1).item() < 0.5:
            img = img.flip(-1)
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
        return img, {"boxes": boxes, "labels": labels, "image_id": im["id"]}


def collate(batch):
    return tuple(zip(*batch))


def anchors_from_kmeans(path, levels=5):
    """Базовий розмір anchor = sqrt(w*h) найменшого центру k-means, далі x2 на кожен рівень FPN."""
    centers = np.array(c.load_json(path))
    base = float(np.sqrt(centers.prod(axis=1)).min())
    base = max(8, int(round(base / 2) * 2))
    return [base * 2 ** i for i in range(levels)]


def build_model(arch, anchor_base=None, octave_scales=False, aspect_ratios=(0.5, 1.0, 2.0),
                min_size=540, max_size=960, train_min_sizes=None, det_per_img=c.MAX_DETS, pretrained=True, **kw):
    """arch: "faster_rcnn" або "retinanet" (обидві ResNet50-FPN, ваги COCO).

    anchor_base: список базових розмірів по рівнях FPN (None = стандартні torchvision).
    min_size/max_size = 540/960: кадри 960x540 подаються без внутрішнього ресайзу.
    train_min_sizes: кортеж для multi-scale навчання (GeneralizedRCNNTransform бере випадковий).
    kw: інші параметри конструктора torchvision (rpn_*, box_*, fg_iou_thresh, ...).
    pretrained=False лише для швидких тестів без інтернету.
    """
    if not pretrained:
        kw["weights_backbone"] = None
    if train_min_sizes:
        common_kw = dict(min_size=tuple(train_min_sizes) + (min_size,),
                         max_size=int(max(max(train_min_sizes), min_size) * max_size / min_size) + 1)
    else:
        common_kw = dict(min_size=min_size, max_size=max_size)
    if arch == "faster_rcnn":
        if anchor_base is not None:
            kw["rpn_anchor_generator"] = AnchorGenerator(tuple((s,) for s in anchor_base),
                                                         (tuple(aspect_ratios),) * len(anchor_base))
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
                                        box_detections_per_img=det_per_img, box_score_thresh=0.001,
                                        **common_kw, **kw)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    elif arch == "retinanet":
        if anchor_base is not None:
            sizes = tuple((x, int(x * 2 ** (1 / 3)), int(x * 2 ** (2 / 3))) for x in anchor_base)
            kw["anchor_generator"] = AnchorGenerator(sizes, (tuple(aspect_ratios),) * len(sizes))
        box_loss = kw.pop("box_loss", "l1")
        model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
                                       detections_per_img=det_per_img, score_thresh=0.001, **common_kw, **kw)
        head = model.head.classification_head
        head.num_classes = NUM_CLASSES
        head.cls_logits = nn.Conv2d(256, head.num_anchors * NUM_CLASSES, 3, padding=1)
        nn.init.normal_(head.cls_logits.weight, std=0.01)
        nn.init.constant_(head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
        model.head.regression_head._loss_type = box_loss
    else:
        raise ValueError(arch)
    return model


def set_proposals(model, pre_nms, post_nms):
    """Кількість RPN-пропозицій на інференсі (Faster R-CNN); перенавчання не потрібне."""
    model.rpn._pre_nms_top_n["testing"] = pre_nms
    model.rpn._post_nms_top_n["testing"] = post_nms


def _lr_lambda(warmup, total, milestones, iters_per_epoch, schedule):
    def f(it):
        warm = min(1.0, (it + 1) / warmup) if warmup else 1.0
        if schedule == "cosine":
            return warm * 0.5 * (1 + math.cos(math.pi * it / total))
        epoch = it // iters_per_epoch
        return warm * 0.1 ** sum(epoch >= m for m in milestones)
    return f


def train(model, prep, out_dir, epochs=6, batch=4, lr=0.01, weight_decay=1e-4, milestones=(4, 5),
          schedule="step", warmup=500, workers=4, amp=True, val_every=1, train_every=1, val_frames_every=1,
          log=print):
    """SGD + warmup + step/cosine, AMP, чекпоінт після кожної епохи (відновлюється з last.pt).

    Повертає історію [{epoch, loss, mAP, ...}] і шлях до best.pt (за val mAP).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = PreparedDataset(prep, "train", train=True, every=train_every)
    g = torch.Generator().manual_seed(c.SEED)
    dl = DataLoader(ds, batch, shuffle=True, num_workers=workers, collate_fn=collate,
                    generator=g, drop_last=len(ds) > batch, persistent_workers=workers > 0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(warmup, total, milestones, max(1, len(dl)), schedule))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    history, start, best = [], 0, -1.0
    last = out_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        history, start, best = ck["history"], ck["epoch"], ck["best"]
        log(f"відновлено з епохи {start}")
    for epoch in range(start, epochs):
        model.train()
        t0, losses = time.time(), []
        for it, (imgs, tgts) in enumerate(dl):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts]
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                log(f"нескінченний loss на ітерації {it}, пропускаю батч")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append({k: float(v.detach()) for k, v in loss_dict.items()})
            if it % 200 == 0:
                log(f"ep {epoch + 1} it {it}/{len(dl)} loss {float(loss.detach()):.3f} lr {opt.param_groups[0]['lr']:.5f}")
        row = {"epoch": epoch + 1, "train_min": (time.time() - t0) / 60,
               **{k: float(np.mean([l[k] for l in losses])) for k in losses[0]}}
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            dets = predict(model, prep, "val", every=val_frames_every, workers=workers)
            ids = {d["id"] for d in PreparedDataset(prep, "val", every=val_frames_every).images}
            row.update(c.evaluate(prep / "val" / "annotations_eval.json", dets, image_ids=ids))
            if row["mAP"] > best:
                best = row["mAP"]
                torch.save(model.state_dict(), out_dir / "best.pt")
        history.append(row)
        log({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "epoch": epoch + 1, "best": best}, last)
        c.save_json(history, out_dir / "history.json")
    return history, out_dir / "best.pt"


@torch.no_grad()
def predict(model, prep, split="val", batch=8, workers=4, every=1):
    """COCO-детекції в ОРИГІНАЛЬНИХ координатах (category_id 1..8)."""
    prep = Path(prep)
    device = next(model.parameters()).device
    model.eval()
    ds = PreparedDataset(prep, split, every=every)
    meta = {im["id"]: im for im in ds.images}
    dl = DataLoader(ds, batch, num_workers=workers, collate_fn=collate)
    dets = []
    for imgs, tgts in dl:
        outs = model([i.to(device) for i in imgs])
        for o, t in zip(outs, tgts):
            im = meta[t["image_id"]]
            b = o["boxes"].float().cpu().numpy()
            b[:, 2:] -= b[:, :2]
            b = c.to_original(b, im["scale_x"], im["scale_y"])
            for box, s, l in zip(b, o["scores"].cpu().numpy(), o["labels"].cpu().numpy()):
                dets.append({"image_id": t["image_id"], "category_id": int(l),
                             "bbox": [round(float(v), 2) for v in box], "score": round(float(s), 5)})
    return dets


@torch.no_grad()
def measure_speed(model, prep, n=100, warmup=10):
    """Середній час інференсу на кадр 960x540, batch=1, FP32, з постобробкою (мс)."""
    device = next(model.parameters()).device
    model.eval()
    ds = PreparedDataset(prep, "val")
    times = []
    for i in range(n + warmup):
        img = ds[i % len(ds)][0].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model([img])
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.mean(times))


@torch.no_grad()
def rpn_recall(model, prep, top_ns=(100, 300, 1000, 2000), every=10, iou=0.5):
    """Recall RPN-пропозицій (до другої стадії) за розміром об'єкта в оригінальних пікселях.

    Порівняння з recall фінальних детекцій показує внесок кожної стадії.
    """
    device = next(model.parameters()).device
    model.eval()
    ds = PreparedDataset(prep, "val", every=every)
    old_pre, old_post = dict(model.rpn._pre_nms_top_n), dict(model.rpn._post_nms_top_n)
    set_proposals(model, max(top_ns), max(top_ns))
    hits = {n: {"small": [0, 0], "medium": [0, 0], "large": [0, 0]} for n in top_ns}
    meta = {x["id"]: x for x in ds.images}
    for img, tgt in ds:
        im = meta[tgt["image_id"]]
        images, _ = model.transform([img.to(device)])
        feats = model.backbone(images.tensors)
        props, _ = model.rpn(images, feats)
        p = props[0]
        ry = img.shape[1] / images.image_sizes[0][0]
        rx = img.shape[2] / images.image_sizes[0][1]
        p = p * torch.tensor([rx, ry, rx, ry], device=device)
        gt = tgt["boxes"].to(device)
        if len(gt) == 0:
            continue
        area = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).cpu().numpy() / (im["scale_x"] * im["scale_y"])
        size = np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))
        for n in top_ns:
            ious = torchvision.ops.box_iou(gt, p[:n]) if len(p) else torch.zeros(len(gt), 1, device=device)
            ok = (ious.max(dim=1).values >= iou).cpu().numpy()
            for s in ("small", "medium", "large"):
                hits[n][s][0] += int(ok[size == s].sum())
                hits[n][s][1] += int((size == s).sum())
    model.rpn._pre_nms_top_n, model.rpn._post_nms_top_n = old_pre, old_post
    return {n: {s: h / max(t, 1) for s, (h, t) in v.items()} for n, v in hits.items()}


@torch.no_grad()
def anchor_match_stats(model, prep, split="train", every=50, fg_iou=0.5):
    """Яка частка GT-об'єктів отримує хоча б один anchor з IoU >= fg_iou (тобто стає позитивом).

    Рахується на реальній сітці anchors моделі (центри + розміри), за розміром об'єкта
    в оригінальних пікселях. Не потребує навчання - лише конфігурації anchors.
    """
    ds = PreparedDataset(prep, split, every=every)
    gen = model.anchor_generator if hasattr(model, "anchor_generator") else model.rpn.anchor_generator
    model.eval()
    device = next(model.parameters()).device
    meta = {x["id"]: x for x in ds.images}
    stats = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    best_iou = []
    cache = {}
    for img, tgt in ds:
        gt = tgt["boxes"]
        if len(gt) == 0:
            continue
        key = tuple(img.shape)
        if key not in cache:
            images, _ = model.transform([img.to(device)])
            feats = model.backbone(images.tensors)
            feats = list(feats.values()) if isinstance(feats, dict) else feats
            cache[key] = gen(images, feats)[0].cpu()
        iou = torchvision.ops.box_iou(gt, cache[key]).max(dim=1).values.numpy()
        im = meta[tgt["image_id"]]
        area = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).numpy() / (im["scale_x"] * im["scale_y"])
        size = np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))
        for s in stats:
            stats[s][0] += int((iou[size == s] >= fg_iou).sum())
            stats[s][1] += int((size == s).sum())
        best_iou.append(iou)
    res = {s: h / max(t, 1) for s, (h, t) in stats.items()}
    res["median_best_iou"] = float(np.median(np.concatenate(best_iou))) if best_iou else 0.0
    return res
