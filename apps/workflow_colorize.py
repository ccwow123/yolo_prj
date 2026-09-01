import argparse
import logging
import os
import shutil
import tempfile

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from tqdm import tqdm
from utils import ComfyUIClient, is_grayscale, collect_source_items, configure_logging
from utils import unzip_to_temp, zip_directory, collect_zip_sources
from utils.config import DEFAULT_COLORIZE_SOURCE, COLORIZE_OUT_SUFFIX

'''
这个脚本用于将漫画灰度图（文件夹1）上色，并通过ComfyUI输出彩色图。

输入自动判断：
  - .zip 文件            → 单 zip 模式
  - 目录内含 .zip        → 批量 zip 模式（每个 zip 产出一个输出 zip）
  - 图片目录 / 图片文件  → 直接处理图片（结果落在 --output 目录）

zip 模式下，每个 zip 处理完成后：
  打包为同目录下的「源文件名[上色].zip」，并自动删除临时输出目录。
流程仅处理图片文件，跳过视频。
'''


# logger
logger = logging.getLogger(__name__)


def _process_dir(source_dir, output_dir, args):
    """对单个输入源执行 分析→彩色复制/灰度上色，结果写入 output_dir。

    返回 bool 表示是否成功。
    """
    items, error = collect_source_items(source_dir, image_only=True)
    if error:
        logger.error(str(error))
        return False
    image_paths = [p for p, _ in items]
    if not image_paths:
        logger.error(f"没有找到有效的图片文件: {source_dir}")
        return False

    logger.info(f"\n共发现 {len(image_paths)} 张图片")

    # 分类图像
    grayscale_images = []
    color_images = []

    logger.info("\n[步骤1] 分析图像类型...")
    for img_path in tqdm(image_paths, desc="分析进度", unit="张"):
        try:
            if is_grayscale(img_path):
                grayscale_images.append(img_path)
            else:
                color_images.append(img_path)
        except Exception as e:
            logger.warning(f"无法分析图像 {os.path.basename(img_path)}: {e}")

    logger.info(f"\n分析完成")
    logger.info(f"  灰度图（需要上色）: {len(grayscale_images)} 张")
    logger.info(f"  彩色图（直接复制）: {len(color_images)} 张")

    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 彩色图直接复制到输出目录
    if color_images:
        logger.info(f"\n[步骤2] 复制彩色图像到输出目录...")
        for img_path in tqdm(color_images, desc="复制进度", unit="张"):
            try:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(output_dir, filename))
                logger.debug(f"  复制: {filename}")
            except Exception as e:
                logger.error(f"复制失败 {os.path.basename(img_path)}: {e}")

    # 步骤3: 灰度图通过ComfyUI上色
    if grayscale_images:
        logger.info(f"\n[步骤3] 运行 ComfyUI 上色处理...")
        logger.info(f"  工作流: {args.workflow}")
        logger.info(f"  输出目录: {output_dir}")

        if not os.path.exists(args.workflow):
            logger.error(f"工作流文件不存在: {args.workflow}")
            return False

        try:
            comfy_client = ComfyUIClient(args.comfyui_server)
            saved_paths = comfy_client.process_batch_images(grayscale_images, args.workflow, output_dir)
            logger.info(f"  ComfyUI上色完成，保存了 {len(saved_paths)} 张图像")
        except Exception as e:
            logger.error(f"ComfyUI处理失败: {e}")
            return False

    return True


def _run_zip_batch(args, zip_paths):
    """批量 zip 模式：逐个解压→处理→打包「源文件名[上色].zip」并清理临时目录。"""
    total = len(zip_paths)
    for i, zp in enumerate(zip_paths, 1):
        logger.info(f"\n[{i}/{total}] 处理压缩包: {zp}")
        tmp_in = unzip_to_temp(zp, suffix='_input')
        tmp_out = tempfile.mkdtemp(prefix='colorize_out_')
        try:
            if not _process_dir(tmp_in, tmp_out, args):
                logger.error(f"  处理失败，跳过: {zp}")
                continue
            stem = os.path.splitext(os.path.basename(zp))[0]
            output_zip = os.path.join(os.path.dirname(zp), stem + COLORIZE_OUT_SUFFIX + '.zip')
            zip_directory(tmp_out, output_zip)
            logger.info(f"  已打包并删除输出目录: {output_zip}")
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
            shutil.rmtree(tmp_in, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='图像上色处理流程（仅图片，支持 zip）')

    # 输入输出参数
    parser.add_argument('--source', type=str, default=DEFAULT_COLORIZE_SOURCE,
                        help='输入：图片目录 / 图片文件 / zip / 含 zip 的目录')
    parser.add_argument('--output', type=str, default='runs',
                        help='图片模式下输出目录（zip 模式下每包用独立临时目录）')

    # ComfyUI 参数
    parser.add_argument('--workflow', type=str, default=r'workflows\anima漫画上色-py.json',
                        help='ComfyUI上色工作流JSON路径')
    parser.add_argument('--comfyui-server', type=str, default='http://127.0.0.1:8188',
                        help='ComfyUI服务器地址')

    args = parser.parse_args()

    # 初始化日志
    configure_logging()

    is_zip_mode, zip_paths = collect_zip_sources(args.source)

    print("=" * 60)
    if is_zip_mode:
        print(f"图像上色（zip 模式，共 {len(zip_paths)} 个压缩包）")
    else:
        print("图像上色（图片目录/图片文件模式）")
    print("=" * 60)

    if is_zip_mode:
        _run_zip_batch(args, zip_paths)
        print("\n" + "=" * 60)
        print("全部 zip 处理完成！")
        print("=" * 60)
        return

    # 图片目录 / 图片文件 模式：结果落在 --output，不自动打包
    os.makedirs(args.output, exist_ok=True)
    if not _process_dir(args.source, args.output, args):
        logger.error("处理未完成")
        exit(1)

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    logger.info(f"  输出目录: {args.output}")


if __name__ == '__main__':
    main()