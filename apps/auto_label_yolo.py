"""YOLO 自动标注脚本：把原始图片批量标成 YOLO 数据集标签。

与 apps/auto_label.py（Florence-2）同构，区别在检测器：
  - Florence-2 用开放词汇 + classes.yaml 定义类别；
  - YOLO 直接用模型自带的类别（model.names），可选 --classes yaml 过滤/重排 class_id。

流程：读取原始图片目录 → YOLO 目标检测（置信度过滤）→ 像素框按图像宽高归一化
转 YOLO (cx,cy,w,h) → 写 label.txt + 带框预览图 + summary.json + data.yaml。
复用 utils 现有工具（load_yolo_model / collect_source_items / get_next_exp_dir /
configure_logging / imread_unicode / resize_max_edge / boxes_to_yolo /
write_yolo_label / draw_detections），不外造轮子。

用法示例:
    python auto_label_yolo.py --model weights/cbook-hand.pt --source imgs
    python auto_label_yolo.py --source ./raw --classes apps/classes.yaml --conf 0.5
"""

import argparse
import json
import logging
import os

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch
import cv2
from tqdm import tqdm

from utils.config import (
    DEFAULT_YOLO_INPUT_DIR, DEFAULT_YOLO_LABEL_MODEL, DEFAULT_YOLO_CONF,
    DEFAULT_YOLO_LABEL_CLASSES, DEFAULT_YOLO_SAVE_DIR,
)
from utils.core import collect_source_items, configure_logging, get_next_exp_dir, imread_unicode, load_yolo_model
from utils.cv import resize_max_edge
from utils.florence2 import boxes_to_yolo, draw_detections, parse_classes_yaml, write_yolo_label

logger = logging.getLogger(__name__)


class _ClassFilter:
    """YOLO 类别过滤/重排：默认用模型自带类别；提供 yaml 时仅保留 names 中的类。

    提供 yaml 时按 yaml 顺序重排 class_id（新 id = 在 names 中的下标）：
    适合模型类别多、只想标注其中少数类，或要把模型类别映射到目标数据集 id 的场景。
    """

    def __init__(self, model_names, classes_path=None):
        # model_names: dict {原始class_id: 名称}，来自 YOLO 的 model.names
        if classes_path:
            names = parse_classes_yaml(classes_path)["names"]
            self._new_id_by_name = {n: i for i, n in enumerate(names)}
            self.names = names
        else:
            # 模型类别可能不连续（如 0,1,5），统一按索引取一个连续列表
            self.names = [model_names[i] for i in range(len(model_names))]
            self._new_id_by_name = {n: i for i, n in enumerate(self.names)}

    def remap(self, raw_class_id, raw_name):
        """返回 (new_class_id, 显示名)；类别不在过滤集合内返回 (None, None)。"""
        name = str(raw_name)
        new_id = self._new_id_by_name.get(name)
        if new_id is None:
            return None, None
        return new_id, name


def run_auto_label_yolo(model_path, source, classes_path, conf, save_dir,
                        fp16=True, copy_undetected=False, export_max_edge=None,
                        list_file=None, imgsz=None):
    model, device_info = load_yolo_model(model_path)
    if model is None:
        logger.error(f"模型加载失败: {model_path}")
        return None, None
    filter_ = _ClassFilter(model.names, classes_path)

    items, error = collect_source_items(source, list_file=list_file, image_only=True, recursive=True)
    if error:
        logger.error(f"错误：{error}")
        return None, None

    save_dir = get_next_exp_dir(save_dir)
    labels_dir = os.path.join(save_dir, "labels")
    images_dir = os.path.join(save_dir, "images")
    preview_dir = os.path.join(save_dir, "previews")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    use_half = fp16 and torch.cuda.is_available()
    kwargs = {'conf': conf, 'verbose': False}
    if use_half:
        kwargs['quantize'] = 16  # 新版 ultralytics 以 quantize 取代 half；16=FP16
    if imgsz:
        kwargs['imgsz'] = imgsz

    all_results = []
    total_dets = 0

    for path, _ in tqdm(items, desc="自动标注(YOLO)", unit="张"):
        try:
            bgr = imread_unicode(path)
            if bgr is None:
                logger.warning(f"读图失败，跳过: {path}")
                continue
            ih, iw = bgr.shape[:2]
            with torch.no_grad():
                result = model(bgr, **kwargs)[0]

            # 导出图可选降分辨率（标签为归一化坐标，缩放后仍有效）；None 时保持原图
            export_bgr = resize_max_edge(bgr, export_max_edge) if export_max_edge else bgr

            base = os.path.splitext(os.path.basename(path))[0]
            records = []
            if result.boxes is not None:
                for box in result.boxes:
                    new_id, name = filter_.remap(int(box.cls), result.names[int(box.cls)])
                    if new_id is None:
                        continue
                    score = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    norm_xyxy = (x1 / iw, y1 / ih, x2 / iw, y2 / ih)
                    cx, cy, bw, bh = boxes_to_yolo([norm_xyxy])[0]
                    records.append({
                        "class_id": new_id, "label": name, "score": score,
                        "xyxy": norm_xyxy, "yolo": (cx, cy, bw, bh),
                    })

            if not records and not copy_undetected:
                logger.info(f"未检出目标，跳过（不拷贝原图）: {base}")
                continue

            # 拷贝原图到 images（与 cbook 数据集结构对齐，供 train.py 引用）
            cv2.imwrite(os.path.join(images_dir, os.path.basename(path)), export_bgr)
            # YOLO 标签（归一化 cx,cy,w,h）
            write_yolo_label(
                os.path.join(labels_dir, base + ".txt"),
                [(r["class_id"], *r["yolo"]) for r in records],
            )
            # 带框预览图：draw_detections 期待归一化 xyxy 框
            preview = draw_detections(
                export_bgr, [(r["class_id"], r["xyxy"], r["score"]) for r in records],
                filter_.names,
            )
            cv2.imwrite(os.path.join(preview_dir, base + ".png"), preview)

            img_result = {
                "image": os.path.basename(path),
                "label_file": os.path.join(labels_dir, base + ".txt"),
                "detection_count": len(records),
                "detections": [
                    {"class_id": r["class_id"], "label": r["label"],
                     "score": r["score"], "yolo_xywh": [round(v, 6) for v in r["yolo"]]}
                    for r in records
                ],
            }
            all_results.append(img_result)
            total_dets += len(records)
        except Exception as e:
            logger.error(f"处理 {path} 失败: {e}")

    # 写出 data.yaml（train 指向 images，YOLO 训练直接引用）
    with open(os.path.join(save_dir, "data.yaml"), 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(save_dir)}\n")
        f.write("train: images\nval: images\ntest: images\n")
        f.write(f"nc: {len(filter_.names)}\n")
        f.write(f"names: {json.dumps(filter_.names, ensure_ascii=False)}\n")

    # 汇总
    summary = {
        "model_path": model_path,
        "device": device_info,
        "source": source,
        "classes": {"names": filter_.names},
        "confidence_threshold": conf,
        "total_images": len(all_results),
        "total_detections": total_dets,
        "results": all_results,
    }
    with open(os.path.join(save_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n标注完成：{len(all_results)} 张图，共 {total_dets} 个目标。结果目录: {save_dir}")
    return save_dir, all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO 自动标注 → YOLO 数据集')
    parser.add_argument('--model', type=str, default=DEFAULT_YOLO_LABEL_MODEL,
                        help='YOLO 模型权重路径（设备由 load_yolo_model 自动选择）')
    parser.add_argument('--source', type=str, default=DEFAULT_YOLO_INPUT_DIR,
                        help='待标注图片目录/单图/txt 列表')
    parser.add_argument('--list-file', type=str, default=None,
                        help='待标注图片路径txt列表（每行一个，支持#注释），优先于--source')
    parser.add_argument('--classes', type=str, default=DEFAULT_YOLO_LABEL_CLASSES,
                        help='类别过滤 yaml（可选，仅标注 names 中的类并按此顺序重排 class_id）；默认用模型自带类别')
    parser.add_argument('--conf', type=float, default=DEFAULT_YOLO_CONF,
                        help='置信度阈值，低于此值的结果丢弃')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_YOLO_SAVE_DIR,
                        help='结果父目录，每个输入生成独立 expN 子目录')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='禁用 FP16 推理（仅 GPU 生效）')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='模型前向尺寸（如 480），None 用模型默认 imgsz')
    parser.add_argument('--copy-undetected', dest='copy_undetected', action='store_true',
                        help='未检出目标的图片也复制原图到输出 images 目录（默认跳过不拷贝）')
    parser.add_argument('--export-max-edge', type=int, default=None,
                        help='导出图（images+previews）的最长边像素，按比例等比缩小；默认不缩放')

    args = parser.parse_args()
    configure_logging()
    run_auto_label_yolo(args.model, args.source, args.classes, args.conf,
                        args.save_dir, args.fp16, args.copy_undetected,
                        args.export_max_edge, list_file=args.list_file, imgsz=args.imgsz)