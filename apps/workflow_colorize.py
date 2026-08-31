import argparse
import os
import shutil
import logging

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from tqdm import tqdm
from utils import ComfyUIClient, is_grayscale, collect_source_items
from utils.config import DEFAULT_ALBUM_SOURCE

# logger
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='图像上色处理流程')
    
    # 输入输出参数
    parser.add_argument('--source', type=str, default=DEFAULT_ALBUM_SOURCE, 
                        help='输入目录或图像文件路径（文件夹1）')
    parser.add_argument('--output', type=str, default='runs', 
                        help='输出目录（文件夹2）')
    
    # ComfyUI 参数
    parser.add_argument('--workflow', type=str, default=r'workflows\anima漫画上色-py.json', 
                        help='ComfyUI上色工作流JSON路径')
    parser.add_argument('--comfyui-server', type=str, default='http://127.0.0.1:8188', 
                        help='ComfyUI服务器地址')
    
    args = parser.parse_args()
    
    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    
    logger.info("=" * 60)
    logger.info("图像上色处理流程")
    logger.info("=" * 60)
    
    # 验证输入
    if not os.path.exists(args.source):
        logger.error(f"输入不存在: {args.source}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 获取所有图像文件（支持单个文件或目录）
    items, error = collect_source_items(args.source, image_only=True)
    if error:
        logger.error(str(error))
        return
    image_paths = [p for p, _ in items]
    if not image_paths:
        logger.error("没有找到有效的图片文件")
        return
    
    logger.info(f"\n共发现 {len(image_paths)} 张图片")
    
    # 分类图像
    grayscale_images = []
    color_images = []
    
    logger.info("\n[步骤1] 分析图像类型...")
    for img_path in tqdm(image_paths, desc="分析进度", unit="张"):
        try:
            if is_grayscale(img_path):
                grayscale_images.append(img_path)
                logger.debug(f"  灰度图: {os.path.basename(img_path)}")
            else:
                color_images.append(img_path)
                logger.debug(f"  彩色图: {os.path.basename(img_path)}")
        except Exception as e:
            logger.warning(f"无法分析图像 {os.path.basename(img_path)}: {e}")
    
    logger.info(f"\n分析完成")
    logger.info(f"  灰度图（需要上色）: {len(grayscale_images)} 张")
    logger.info(f"  彩色图（直接复制）: {len(color_images)} 张")
    
    # 步骤2: 彩色图直接复制到输出目录
    if color_images:
        logger.info(f"\n[步骤2] 复制彩色图像到输出目录...")
        for img_path in tqdm(color_images, desc="复制进度", unit="张"):
            try:
                filename = os.path.basename(img_path)
                dest_path = os.path.join(args.output, filename)
                shutil.copy2(img_path, dest_path)
                logger.debug(f"  复制: {filename}")
            except Exception as e:
                logger.error(f"复制失败 {os.path.basename(img_path)}: {e}")
    
    # 步骤3: 灰度图通过ComfyUI上色
    if grayscale_images:
        logger.info(f"\n[步骤3] 运行 ComfyUI 上色处理...")
        logger.info(f"  工作流: {args.workflow}")
        logger.info(f"  输出目录: {args.output}")
        
        # 验证工作流文件
        if not os.path.exists(args.workflow):
            logger.error(f"工作流文件不存在: {args.workflow}")
            return
        
        try:
            comfy_client = ComfyUIClient(args.comfyui_server)
            logger.info(f"批量处理 {len(grayscale_images)} 张灰度图...")
            saved_paths = comfy_client.process_batch_images(grayscale_images, args.workflow, args.output)
            logger.info(f"  ComfyUI上色完成，保存了 {len(saved_paths)} 张图像")
        except Exception as e:
            logger.error(f"ComfyUI处理失败: {e}")
            return
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info(f"  输出目录: {args.output}")
    logger.info(f"  灰度图（已上色）: {len(grayscale_images)} 张")
    logger.info(f"  彩色图（直接复制）: {len(color_images)} 张")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()