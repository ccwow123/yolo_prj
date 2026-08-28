"""Florence-2 自动标注脚本：把原始图片批量标成 YOLO 数据集标签。

流程：读取原始图片目录 → Florence-2 开放词汇检测（classes.yaml 定义类别）
→ 置信度过滤 → 坐标转 YOLO 归一化 (cx,cy,w,h) → 写 label.txt + 带框预览图
+ summary.json。复用 utils/core.py 的 collect_source_items / get_next_exp_dir /
configure_logging / imread_unicode，不外造轮子。

用法示例:
    python auto_label.py --source imgs --classes classes.yaml
    python auto_label.py --source ./raw --model weights/florence2/Florence-2-base
"""

import argparse
import json
import logging
import os

import cv2

from utils.config import (
    DEFAULT_FLORENCE2_CLASSES, DEFAULT_FLORENCE2_CONF, DEFAULT_FLORENCE2_INPUT_DIR, DEFAULT_FLORENCE2_MODEL,
    DEFAULT_FLORENCE2_SAVE_DIR,
)
from utils.core import collect_source_items, configure_logging, get_next_exp_dir, imread_unicode
from utils.florence2 import (
    Florence2Annotator, boxes_to_yolo, draw_detections, parse_classes_yaml,
    write_yolo_label,
)

logger = logging.getLogger(__name__)


def run_auto_label(model_path, source, classes_path, conf, save_dir,
                   device='cuda', fp16=True, copy_undetected=False):
    classes = parse_classes_yaml(classes_path)
    names = classes["names"]
    prompts = classes["prompts"]

    annotator = Florence2Annotator(model_path, device=device, fp16=fp16)

    items, error = collect_source_items(source, image_only=True)
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

    all_results = []
    total_dets = 0

    for path, _ in items:
        try:
            bgr = imread_unicode(path)
            if bgr is None:
                logger.warning(f"读图失败，跳过: {path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = annotator.detect(rgb, prompts)

            fil = [d for d in dets if d["score"] >= conf]
            base = os.path.splitext(os.path.basename(path))[0]
            if not fil and not copy_undetected:
                logger.info(f"未检出目标，跳过（不拷贝原图）: {base}")
                continue
            records = [
                (d["class_id"], boxes_to_yolo([d["box_xyxy"]])[0], d["score"])
                for d in fil
            ]

            # 拷贝原图到 images（与 cbook 数据集结构对齐，供 train.py 引用）
            cv2.imwrite(os.path.join(images_dir, os.path.basename(path)), bgr)
            # YOLO 标签（归一化 cx,cy,w,h）
            write_yolo_label(
                os.path.join(labels_dir, base + ".txt"),
                [(c, cx, cy, w, h) for c, (cx, cy, w, h), _ in records],
            )
            # 带框预览图：draw_detections 期待归一化 xyxy 框；(x0,y0,x1,y1) 来自 detect
            preview = draw_detections(
                bgr, [(d["class_id"], d["box_xyxy"], d["score"]) for d in fil], names
            )
            cv2.imwrite(os.path.join(preview_dir, base + ".png"), preview)

            img_result = {
                "image": os.path.basename(path),
                "label_file": os.path.join(labels_dir, base + ".txt"),
                "detection_count": len(records),
                "detections": [
                    {"class_id": c, "label": names[c] if c < len(names) else str(c),
                     "score": sc, "yolo_xywh": [round(v, 6) for v in box]}
                    for c, box, sc in records
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
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {json.dumps(names, ensure_ascii=False)}\n")

    # 汇总
    summary = {
        "model_path": model_path,
        "source": source,
        "classes": classes,
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
    parser = argparse.ArgumentParser(description='Florence-2 自动标注 → YOLO 数据集')
    parser.add_argument('--model', type=str, default=DEFAULT_FLORENCE2_MODEL,
                        help='Florence-2 模型本地目录（含 config.json + *.safetensors）')
    parser.add_argument('--source', type=str, default=DEFAULT_FLORENCE2_INPUT_DIR,
                        help='待标注图片目录/单图/txt 列表')
    parser.add_argument('--classes', type=str, default=DEFAULT_FLORENCE2_CLASSES,
                        help='类别定义 yaml（顺序决定 class_id）')
    parser.add_argument('--conf', type=float, default=DEFAULT_FLORENCE2_CONF,
                        help='置信度阈值，低于此值的结果丢弃')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_FLORENCE2_SAVE_DIR,
                        help='结果父目录，每个输入生成独立 expN 子目录')
    parser.add_argument('--device', type=str, default='cuda', help='推理设备（cuda/cpu）')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='禁用 FP16 推理（仅 GPU 生效）')
    parser.add_argument('--copy-undetected', dest='copy_undetected', action='store_true',
                        help='未检出目标的图片也复制原图到输出 images 目录（默认跳过不拷贝）')

    args = parser.parse_args()
    configure_logging()
    run_auto_label(args.model, args.source, args.classes, args.conf,
                   args.save_dir, args.device, args.fp16, args.copy_undetected)