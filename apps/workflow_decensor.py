import argparse
import logging
import os
import shutil
import tempfile

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from detect import run_detection
from utils import ComfyUIClient, load_yolo_model, configure_logging
from utils import unzip_to_temp, zip_directory, collect_zip_sources
from utils.config import (DEFAULT_CENSOR_SOURCE, DEFAULT_CENSOR_MODEL,
                          DEFAULT_DECENSOR_OUT_DIR)

'''
这个脚本用于检测漫画输入（文件夹1）中的图片，并使用ComfyUI去码处理。

输入自动判断：
  - .zip 文件            → 单 zip 模式
  - 目录内含 .zip        → 批量 zip 模式（每个 zip 产出一个输出 zip）
  - 图片目录 / 图片文件  → 直接处理图片（结果落在 comfyui-save-dir）

zip 模式下，每个 zip 处理完成后：
  打包为同目录下的「源文件名[去码].zip」，并自动删除临时输出目录。
检测结果保存在 detect-save-dir（expN），ComfyUI 处理保存在输出目录。
流程仅处理图片文件，跳过视频。
'''


# logger
logger = logging.getLogger(__name__)


def _run_pipeline(model, model_path, source, args, comfyui_save_dir):
    """对单个输入源执行 检测→分类→复制/ComfyUI，结果写入 comfyui_save_dir。

    检测输出写到 runs\\detections\\expN（全局共享）。
    返回结果 dict；检测失败返回 None。
    """
    detect_output_dir, results, annotated_dir = run_detection(
        model, model_path, source, args.conf,
        args.detect_save_dir, args.save_json,
        save_annotated=False, image_only=True,
        annotated_dir=args.annotated_dir,
        annotate_classes=args.annotate_classes
    )
    if detect_output_dir is None:
        return None

    det, nodet = [], []
    for r in results:
        (det if r["detection_count"] > 0 else nodet).append(
            os.path.join(detect_output_dir, r["image_filename"])
        )

    logger.info(f"  检测: 共 {len(results)} 张, 有目标 {len(det)} / 无目标 {len(nodet)}")
    os.makedirs(comfyui_save_dir, exist_ok=True)

    for p in nodet:
        shutil.copy2(p, os.path.join(comfyui_save_dir, os.path.basename(p)))

    if det:
        if not args.run_comfyui:
            logger.info(f"[步骤3] 未启用 ComfyUI，直接复制 {len(det)} 张原图...")
            for p in det:
                shutil.copy2(p, os.path.join(comfyui_save_dir, os.path.basename(p)))
        else:
            if not os.path.exists(args.workflow):
                logger.error(f"工作流文件不存在: {args.workflow}")
                raise SystemExit(1)
            logger.info(f"[步骤3] 运行 ComfyUI（{len(det)} 张）...")
            client = ComfyUIClient(args.comfyui_server, poll_timeout=args.poll_timeout)
            saved = client.process_batch_images(det, args.workflow, comfyui_save_dir)
            logger.info(f"  ComfyUI 处理完成，成功 {len(saved)} / {len(det)} 张")
            done = set(os.path.basename(p) for p in saved)
            fallback = 0
            for p in det:
                fname = os.path.basename(p)
                if fname not in done:
                    shutil.copy2(p, os.path.join(comfyui_save_dir, fname))
                    fallback += 1
            if fallback:
                logger.warning(f"  ComfyUI 失败回退 {fallback} 张（保留原图保证完整性）")

    return {
        'detect_output_dir': detect_output_dir,
        'annotated_dir': annotated_dir,
        'images_with_detection': det,
        'images_without_detection': nodet,
    }


def _run_zip_batch(model, args, zip_paths, out_dir):
    """批量 zip 模式：逐个解压→处理→打包「源文件名[去码].zip」到 out_dir 并清理临时目录。"""
    os.makedirs(out_dir, exist_ok=True)
    total = len(zip_paths)
    for i, zp in enumerate(zip_paths, 1):
        logger.info(f"\n[{i}/{total}] 处理压缩包: {zp}")
        tmp_in = unzip_to_temp(zp, suffix='_input')
        tmp_out = tempfile.mkdtemp(prefix='decensor_out_')
        try:
            info = _run_pipeline(model, args.model, tmp_in, args, tmp_out)
            if info is None:
                logger.error(f"  检测失败，跳过: {zp}")
                continue
            stem = os.path.splitext(os.path.basename(zp))[0]
            output_zip = os.path.join(out_dir, stem + '[去码]' + '.zip')
            zip_directory(tmp_out, output_zip)
            logger.info(f"  已打包并删除输出目录: {output_zip}")
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
            shutil.rmtree(tmp_in, ignore_errors=True)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description='检测 + ComfyUI 去码处理流程（仅图片）')

    # Detect 参数
    parser.add_argument('--model', type=str, default=DEFAULT_CENSOR_MODEL,
                        help='模型权重文件路径（默认从 config 读取）')
    parser.add_argument('--source', type=str, default=DEFAULT_CENSOR_SOURCE,
                        help='输入：图片目录 / 图片文件 / zip / 含 zip 的目录')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='检测置信度阈值')
    parser.add_argument('--detect-save-dir', type=str, default=r'runs\detections',
                        help='检测结果保存目录（自动生成 expN）')
    parser.add_argument('--save-json', action='store_true', default=False,
                        help='保存单个检测json文件（汇总json始终保存）')
    parser.add_argument('--annotated-dir', action='store_true', default=True,
                        help='启用后把带检测框预览保存到 expN\\annotated 目录（不影响源图）')
    parser.add_argument('--annotate-classes', type=str, default=['penis','pussy'],
                        help='按类别画框：list 格式的类别名，如 ["对话气泡","文字"]；仅对匹配类别的检测框画框；不传则画所有框')

    # ComfyUI 参数
    parser.add_argument('--run-comfyui', action='store_true', default=False,
                        help='运行 ComfyUI 去码（默认不运行，有检测目标的图片仅复制原图）')
    parser.add_argument('--workflow', type=str, default=r'workflows\f2k-漫画去码-py.json',
                        help='ComfyUI工作流JSON路径')
    parser.add_argument('--comfyui-save-dir', type=str,
                        default=DEFAULT_CENSOR_SOURCE + '[去码]',
                        help='图片模式下结果保存目录（zip 模式下每包用独立临时目录）')
    parser.add_argument('--comfyui-server', type=str, default='http://127.0.0.1:8188',
                        help='ComfyUI服务器地址')
    parser.add_argument('--poll-timeout', type=int, default=300,
                        help='ComfyUI 轮询等待超时（秒），0 不限制')
    parser.add_argument('--out-zip-dir', type=str, default=DEFAULT_DECENSOR_OUT_DIR,
                        help='zip 模式下输出 zip 的目标目录（文件名保持「源文件名[去码].zip」）')

    args = parser.parse_args()

    # list 格式类别名（如 ["对话气泡","文字"]）→ 集合；不传/空则画所有框
    if args.annotate_classes:
        if isinstance(args.annotate_classes, str):
            try:
                raw = ast.literal_eval(args.annotate_classes)
            except (ValueError, SyntaxError):
                logger.error("--annotate-classes 需为 list 格式，如 [\"对话气泡\",\"文字\"]")
                exit(1)
        else:
            raw = args.annotate_classes
        args.annotate_classes = {str(c).strip() for c in raw if str(c).strip()}
    else:
        args.annotate_classes = None

    is_zip_mode, zip_paths = collect_zip_sources(args.source)

    print("=" * 60)
    if is_zip_mode:
        print(f"检测 + ComfyUI 去码（zip 模式，共 {len(zip_paths)} 个压缩包）")
    else:
        print("检测 + ComfyUI 去码（图片目录/图片文件模式）")
    if args.annotate_classes:
        print(f"按类别画框: {', '.join(sorted(args.annotate_classes))}")
    print("=" * 60)

    model, device_info = load_yolo_model(args.model)
    if model is None:
        logger.error("模型加载失败")
        exit(1)

    if is_zip_mode:
        _run_zip_batch(model, args, zip_paths, args.out_zip_dir)
        print("\n" + "=" * 60)
        print("全部 zip 处理完成！")
        print("=" * 60)
        return

    # 图片目录 / 图片文件 模式：结果落在 comfyui-save-dir，不自动打包
    logger.info(f"  输入: {args.source}")

    info = _run_pipeline(model, args.model, args.source, args, args.comfyui_save_dir)
    if info is None:
        logger.error("检测未完成，未生成输出目录")
        exit(1)

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    logger.info(f"  检测结果目录: {info['detect_output_dir']}")
    if info['annotated_dir']:
        logger.info(f"  带框预览目录: {info['annotated_dir']}")
    logger.info(f"  ComfyUI 输出目录: {args.comfyui_save_dir}")
    logger.info(f"  有检测目标: {len(info['images_with_detection'])} 张")
    logger.info(f"  无检测目标: {len(info['images_without_detection'])} 张")


if __name__ == '__main__':
    main()