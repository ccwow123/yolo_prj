import argparse
import os
import shutil
import logging
from detect import run_detection
from utils import ComfyUIClient, load_yolo_model, configure_logging
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
    parser.add_argument('--conf', type=float, default=0.6,
                        help='检测置信度阈值')
    parser.add_argument('--detect-save-dir', type=str, default=r'runs\detections',
                        help='检测结果保存目录（自动生成 expN）')
    parser.add_argument('--save-json', action='store_true', default=False,
                        help='保存单个检测json文件（汇总json始终保存）')

    # ComfyUI 参数
    parser.add_argument('--workflow', type=str, default=r'workflows\f2k-漫画去码-py.json',
                        help='ComfyUI工作流JSON路径')
    parser.add_argument('--comfyui-save-dir', type=str, default=DEFAULT_ALBUM_SOURCE + '[去码]',
                        help='ComfyUI结果保存目录（文件夹3）')
    parser.add_argument('--comfyui-server', type=str, default='http://127.0.0.1:8188',
                        help='ComfyUI服务器地址')
    parser.add_argument('--poll-timeout', type=int, default=300,
                        help='ComfyUI 轮询等待超时（秒），0 不限制')

    args = parser.parse_args()

    print("=" * 60)
    print("检测 + ComfyUI 去码处理流程（仅图片）")
    print("=" * 60)

    # 步骤1: 调用 detect.py 进行推理（保存原图，不带检测框）
    logger.info("\n[步骤1] 运行 YOLO 检测...")
    logger.info(f"  输入目录: {args.source}")
    logger.info(f"  模型路径: {args.model}")
    logger.info(f"  置信阈值: {args.conf}")
    logger.info(f"  输出原图（不带检测框）")

    model, device_info = load_yolo_model(args.model)
    if model is None:
        logger.error("模型加载失败")
        exit(1)

    # image_only=True 只收集图片，跳过视频，避免 None 崩溃
    detect_output_dir, results = run_detection(
        model, args.model, args.source, args.conf,
        args.detect_save_dir, args.save_json,
        save_annotated=False, image_only=True
    )
    if detect_output_dir is None:
        logger.error("检测未完成，未生成输出目录")
        exit(1)
    logger.info(f"  输出目录: {detect_output_dir}")

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

    # 步骤5: 有检测目标走 ComfyUI 处理
    if images_with_detection:
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

if __name__ == '__main__':
    main()