import argparse
import os
import shutil
import logging

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from detect import run_detection
from utils import ComfyUIClient, load_yolo_model, configure_logging
from utils import is_zip_file, unzip_to_temp, zip_directory
from utils.config import DEFAULT_ALBUM_SOURCE, DEFAULT_CENSOR_MODEL

'''
这个脚本用于检测漫画输入目录（文件夹1）中的图片，
并使用ComfyUI去码处理。
检测结果保存在detect-save-dir目录（文件夹2）中，
ComfyUI处理保存在comfyui-save-dir目录（文件夹3）中。
json文件保存在save-json目录（文件夹4）中。
流程仅处理图片文件，跳过视频。
'''


# logger
logger = logging.getLogger(__name__)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description='检测 + ComfyUI 去码处理流程（仅图片）')

    # Detect 参数
    parser.add_argument('--model', type=str, default=DEFAULT_CENSOR_MODEL,
                        help='模型权重文件路径（默认从 config 读取）')
    parser.add_argument('--source', type=str, default=DEFAULT_ALBUM_SOURCE,
                        help='检测输入目录（仅图片）')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='检测置信度阈值')
    parser.add_argument('--detect-save-dir', type=str, default=r'runs\detections',
                        help='检测结果保存目录（自动生成 expN）')
    parser.add_argument('--save-json', action='store_true', default=False,
                        help='保存单个检测json文件（汇总json始终保存）')
    parser.add_argument('--annotated-dir', default=True,
                        help='启用后把带检测框预览保存到 expN\\annotated 目录（不影响源图）')

    # ComfyUI 参数
    parser.add_argument('--run-comfyui',  default=False,
                        help='运行 ComfyUI 去码（默认不运行，有检测目标的图片仅复制原图）')
    parser.add_argument('--workflow', type=str, default=r'workflows\f2k-漫画去码-py.json',
                        help='ComfyUI工作流JSON路径')
    parser.add_argument('--comfyui-save-dir', type=str, default=DEFAULT_ALBUM_SOURCE + '[去码]',
                        help='ComfyUI结果保存目录（文件夹3）')
    parser.add_argument('--comfyui-server', type=str, default='http://127.0.0.1:8188',
                        help='ComfyUI服务器地址')
    parser.add_argument('--poll-timeout', type=int, default=300,
                        help='ComfyUI 轮询等待超时（秒），0 不限制')


    args = parser.parse_args()

    # 支持 zip 输入：source 为 .zip 时解压到临时目录作为输入源
    original_source = args.source
    tmp_input_dir = None
    if is_zip_file(args.source):
        logger.info(f"检测到 zip 输入: {args.source}")
        tmp_input_dir = unzip_to_temp(args.source, suffix='_input')
        args.source = tmp_input_dir

    print("=" * 60)
    print("检测 + ComfyUI 去码处理流程（仅图片）")
    print("=" * 60)

    # 步骤1: 调用 detect.py 进行推理（保存原图，不带检测框）
    logger.info("\n[步骤1] 运行 YOLO 检测...")
    logger.info(f"  输入目录: {args.source}")
    if original_source != args.source:
        logger.info(f"  原始输入: {original_source}")
    logger.info(f"  模型路径: {args.model}")
    logger.info(f"  置信阈值: {args.conf}")
    logger.info(f"  输出原图（不带检测框）")

    model, device_info = load_yolo_model(args.model)
    if model is None:
        logger.error("模型加载失败")
        exit(1)

    # image_only=True 只收集图片，跳过视频，避免 None 崩溃
    detect_output_dir, results, annotated_dir = run_detection(
        model, args.model, args.source, args.conf,
        args.detect_save_dir, args.save_json,
        save_annotated=False, image_only=True,
        annotated_dir=args.annotated_dir
    )
    if detect_output_dir is None:
        logger.error("检测未完成，未生成输出目录")
        exit(1)
    logger.info(f"  输出目录: {detect_output_dir}")
    if annotated_dir:
        logger.info(f"  带框预览目录: {annotated_dir}")

    # 步骤2: 直接用 detect.py 返回值分类有/无检测目标
    images_with_detection = []
    images_without_detection = []

    for result in results:
        # image_only 模式下 collect_source_items 只返回图片，result["image_filename"] 必存在
        img_path = os.path.join(detect_output_dir, result["image_filename"])
        if result["detection_count"] > 0:
            images_with_detection.append(img_path)
        else:
            images_without_detection.append(img_path)

    logger.info(f"\n检测完成")
    logger.info(f"  共处理图片: {len(results)} 张")
    logger.info(f"  有检测目标: {len(images_with_detection)} 张")
    logger.info(f"  无检测目标: {len(images_without_detection)} 张")

    # 步骤3: 创建输出文件夹
    os.makedirs(args.comfyui_save_dir, exist_ok=True)

    # 步骤4: 无检测目标直接复制
    if images_without_detection:
        logger.info(f"\n[步骤2] 复制无检测目标图片到输出目录...")
        for img_path in images_without_detection:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(args.comfyui_save_dir, filename)
            shutil.copy2(img_path, dest_path)
            logger.debug(f"  复制: {filename}")

    # 步骤5: 有检测目标处理（ComfyUI 或直接复制）
    if images_with_detection:
        if not args.run_comfyui:
            # 未加 --run-comfyui：跳过 ComfyUI，直接复制原图
            logger.info(f"\n[步骤3] 未启用 ComfyUI（未加 --run-comfyui），直接复制 {len(images_with_detection)} 张原图...")
            for img_path in images_with_detection:
                dest_path = os.path.join(args.comfyui_save_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dest_path)
                logger.debug(f"  复制: {os.path.basename(img_path)}")
        else:
            logger.info(f"\n[步骤3] 运行 ComfyUI 处理...")
            logger.info(f"  工作流: {args.workflow}")
            logger.info(f"  服务器: {args.comfyui_server}")
            logger.info(f"  输出目录: {args.comfyui_save_dir}")

            if not os.path.exists(args.workflow):
                logger.error(f"工作流文件不存在: {args.workflow}")
                exit(1)

            comfy_client = ComfyUIClient(args.comfyui_server, poll_timeout=args.poll_timeout)
            saved_paths = comfy_client.process_batch_images(
                images_with_detection, args.workflow, args.comfyui_save_dir
            )
            logger.info(f"  ComfyUI 处理完成，成功保存 {len(saved_paths)} / {len(images_with_detection)} 张")

            # 回退：ComfyUI 失败的图片复制原检测图
            processed_filenames = set(os.path.basename(p) for p in saved_paths)
            fallback_count = 0
            for img_path in images_with_detection:
                filename = os.path.basename(img_path)
                if filename not in processed_filenames:
                    dest_path = os.path.join(args.comfyui_save_dir, filename)
                    shutil.copy2(img_path, dest_path)
                    logger.warning(f"  ComfyUI 处理失败，回退复制原图: {filename}")
                    fallback_count += 1
            if fallback_count > 0:
                logger.warning(f"  共回退 {fallback_count} 张图片（保留原图保证输出完整性）")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    logger.info(f"  检测结果目录: {detect_output_dir}")
    logger.info(f"  ComfyUI 输出目录: {args.comfyui_save_dir}")
    logger.info(f"  有检测目标: {len(images_with_detection)} 张")
    logger.info(f"  无检测目标: {len(images_without_detection)} 张")

    # zip 输入时：把 comfyui-save-dir 打包成同名 zip，并清理临时输入目录
    if original_source != args.source and tmp_input_dir:
        output_zip = zip_directory(args.comfyui_save_dir, args.comfyui_save_dir + '.zip')
        logger.info(f"  输出已打包: {output_zip}")
        shutil.rmtree(tmp_input_dir, ignore_errors=True)
        logger.info(f"  已清理临时输入目录: {tmp_input_dir}")

if __name__ == '__main__':
    main()