"""工作流脚本：先跑 hand_distance.py 去重截图，再把去重图片喂给 auto_label.py 标注。

串联两个既有 CLI 的 run_* 函数：
  1. run_hand_distance → 输出每段视频的去重图目录（filtered_dir）
  2. 汇总所有去重 jpg 到临时 list.txt（复用 collect_source_items / load_source_list 约定）
  3. run_auto_label(list_file=...) → 生成 YOLO 标签 + 预览图 + summary.json

用法示例:
    python hand2label.py --source ./videos
    python hand2label.py --source ./videos --distance-threshold 1200 --stable-duration 2
"""

import argparse
import logging
import os
import tempfile

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from auto_label import run_auto_label
from hand_distance import run_hand_distance
from utils.config import (
    DEFAULT_FLORENCE2_CLASSES, DEFAULT_FLORENCE2_CONF, DEFAULT_FLORENCE2_MODEL,
    DEFAULT_FLORENCE2_SAVE_DIR, DEFAULT_HAND_MODEL, DEFAULT_HAND_SAVE_DIR, DEFAULT_INFER_IMGSZ, DEFAULT_INFER_MAX_EDGE,
    DEFAULT_MOTION_THRESHOLD, DEFAULT_VIDEO_SOURCE,
)
from utils.core import collect_source_items, configure_logging

logger = logging.getLogger(__name__)


def collect_dedup_images(hand_stats):
    """从 hand_distance 结果中收集所有去重截图目录下的图片路径列表。"""
    images = []
    for res in hand_stats.get('video_results', []):
        filtered_dir = res.get('filtered_dir')
        if not filtered_dir or not os.path.isdir(filtered_dir):
            continue
        items, error = collect_source_items(filtered_dir, image_only=True, recursive=True)
        if error:
            logger.warning(f"收集去重图失败（{filtered_dir}）: {error}")
            continue
        for path, _ in items:
            images.append(path)
        logger.info(f"  - {os.path.basename(filtered_dir)}: {len(items)} 张去重图")
    return images


def main():
    parser = argparse.ArgumentParser(description='工作流：hand_distance 去重截图 → auto_label 自动标注')
    # hand_distance 阶段
    parser.add_argument('--source', type=str, default=DEFAULT_VIDEO_SOURCE,
                        help='视频/图片源路径或目录')
    parser.add_argument('--hand-model', type=str, default= DEFAULT_HAND_MODEL,
                        help='手部检测模型权重路径')
    parser.add_argument('--hand-save-dir', type=str, default=DEFAULT_HAND_SAVE_DIR,
                        help='hand_distance 输出父目录（每段视频一个 expN）')
    parser.add_argument('--conf', type=float, default=0.6, help='手部检测置信度阈值')
    parser.add_argument('--distance-threshold', type=int, default=1200,
                        help='触发截图的距离阈值（像素）')
    parser.add_argument('--stable-duration', type=float, default=1,
                        help='触发截图所需的稳定时长（秒）')
    parser.add_argument('--crop-ratio', type=float, default=0,
                        help='截图两边向中央裁剪的总比例（默认0，即不裁剪）')
    parser.add_argument('--quality', type=int, default=100, help='截图压缩质量（1-100）')
    parser.add_argument('--max-edge', type=int, default=DEFAULT_INFER_MAX_EDGE,
                        help='视频输入帧最长边像素（0 表示不预缩放）')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_INFER_IMGSZ,
                        help='模型前向尺寸（0 用模型默认）')
    parser.add_argument('--motion-threshold', type=float, default=DEFAULT_MOTION_THRESHOLD,
                        help='静止判定阈值（0-255 帧间MAD，0 关闭）')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false', default=True,
                        help='禁用手部检测 FP16 混合精度推理')
    parser.add_argument('--no-video', dest='write_video', action='store_false', default=True,
                        help='hand_distance 阶段不生成输出视频（纯截图，最快）')
    parser.add_argument('--no-annotate-video', dest='annotate_video', action='store_false', default=True,
                        help='hand_distance 输出视频不叠加检测标注')
    # auto_label 阶段
    parser.add_argument('--florence-model', type=str, default=DEFAULT_FLORENCE2_MODEL,
                        help='Florence-2 模型本地目录')
    parser.add_argument('--classes', type=str, default=DEFAULT_FLORENCE2_CLASSES,
                        help='类别定义 yaml（顺序决定 class_id）')
    parser.add_argument('--label-conf', type=float, default=DEFAULT_FLORENCE2_CONF,
                        help='Florence-2 检测置信度阈值')
    parser.add_argument('--label-save-dir', type=str, default=DEFAULT_FLORENCE2_SAVE_DIR,
                        help='标注结果父目录（生成独立 expN 子目录）')
    parser.add_argument('--device', type=str, default='cuda', help='Florence-2 推理设备（cuda/cpu）')
    parser.add_argument('--no-label-fp16', dest='label_fp16', action='store_false', default=True,
                        help='禁用 Florence-2 FP16 推理')
    parser.add_argument('--copy-undetected', action='store_true',
                        help='未检出目标的图片也复制到标注输出（默认跳过）')
    parser.add_argument('--export-max-edge', type=int, default=None,
                        help='标注导出图最长边像素，等比缩放；默认不缩放')

    args = parser.parse_args()
    configure_logging()

    logger.info("=" * 60)
    logger.info("工作流：hand_distance 去重 + auto_label 标注")
    logger.info("=" * 60)
    logger.info(f"源: {args.source}")
    logger.info(f"手部模型: {args.hand_model}")
    logger.info(f"距离阈值: {args.distance_threshold} px, 稳定时长: {args.stable_duration}s")
    logger.info(f"静止判定阈值: {args.motion_threshold} MAD")
    logger.info(f"Florence-2 模型: {args.florence_model}")
    logger.info(f"标注输出目录: {args.label_save_dir}")
    logger.info("=" * 60)

    # 阶段一：hand_distance 去重截图
    logger.info("\n[阶段 1/2] 运行 hand_distance 去重截图...")
    hand_stats = run_hand_distance(
        args.hand_model, args.source, args.conf, args.hand_save_dir,
        save_txt=True, distance_threshold=args.distance_threshold,
        stable_duration=args.stable_duration, crop_ratio=args.crop_ratio,
        quality=args.quality, max_edge=(args.max_edge or None),
        imgsz=(args.imgsz or None), fp16=args.fp16,
        motion_threshold=args.motion_threshold,
        write_video=args.write_video, annotate_video=args.annotate_video,
    )
    if not hand_stats or hand_stats.get('success_count', 0) == 0:
        logger.error("hand_distance 阶段没有成功处理任何输入，工作流中止。")
        return 1

    # 汇总所有去重图
    labels = collect_dedup_images(hand_stats)
    if not labels:
        logger.error("未收集到任何去重截图（可能输入不是视频，或视频未触发截图）。工作流中止。")
        return 1
    logger.info(f"共收集 {len(labels)} 张去重截图，进入标注阶段。")

    tmp_list = None
    try:
        # 写临时 list.txt，复用 load_source_list 读取约定
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                         encoding='utf-8') as f:
            f.write('\n'.join(labels))
            tmp_list = f.name

        # 阶段二：auto_label 标注
        logger.info("\n[阶段 2/2] 运行 auto_label 自动标注...")
        save_dir, results = run_auto_label(
            args.florence_model, None, args.classes, args.label_conf,
            args.label_save_dir, device=args.device, fp16=args.label_fp16,
            copy_undetected=args.copy_undetected,
            export_max_edge=args.export_max_edge, list_file=tmp_list,
        )
        if not save_dir:
            logger.error("auto_label 阶段失败。")
            return 1

        logger.info("\n" + "=" * 60)
        logger.info("工作流完成")
        logger.info(f"  去重截图: {len(labels)} 张")
        logger.info(f"  标注图片: {len(results) if results else 0} 张")
        logger.info(f"  标注结果: {save_dir}")
        logger.info("=" * 60)
        return 0
    finally:
        if tmp_list and os.path.exists(tmp_list):
            os.remove(tmp_list)


if __name__ == '__main__':
    raise SystemExit(main())