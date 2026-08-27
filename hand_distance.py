import argparse
import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

from utils import (
    get_next_exp_dir, validate_parameters, collect_source_items, imread_unicode,
    detect_hands, calculate_hand_distance, crop_image, save_screenshot,
    deduplicate_screenshots, save_distance_summary, save_frame_distance_log,
    load_yolo_model, convert_screenshots_to_jpg
)

# logger
logger = logging.getLogger(__name__)


def process_single_image(model, image_path, save_dir, save_txt, crop_ratio=0.3, quality=100, conf=0.6):
    """
    处理单张图像
    
    Args:
        model: YOLO模型对象
        image_path: 图像文件路径
        save_dir: 结果保存目录
        save_txt: 是否保存距离结果为txt文件
        crop_ratio: 裁剪比例
        quality: 图像压缩质量
        conf: 置信度阈值
    
    Returns:
        dict: 处理结果信息（包含距离、是否成功等）
    """
    image = imread_unicode(image_path)
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return {'success': False, 'error': '无法读取图片', 'filename': os.path.basename(image_path)}
    
    result, has_hands = detect_hands(model, image, conf)
    
    if not has_hands:
        logger.warning(f"未检测到足够的手: {os.path.basename(image_path)}")
        return {'success': False, 'error': '未检测到足够的手', 'filename': os.path.basename(image_path)}
    
    distance, annotated, hands = calculate_hand_distance(image, result.boxes)
    
    annotated = crop_image(annotated, crop_ratio)
    
    img_name = os.path.basename(image_path)
    name = os.path.splitext(img_name)[0]
    
    if quality < 100:
        output_path = os.path.join(save_dir, f'{name}.jpg')
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(output_path, annotated, jpeg_params)
    else:
        output_path = os.path.join(save_dir, f'{name}.png')
        cv2.imwrite(output_path, annotated)
    
    if save_txt and distance is not None:
        with open(os.path.join(save_dir, f'{name}_distance.txt'), 'w', encoding='utf-8') as f:
            f.write(f"图片: {img_name}\n")
            f.write(f"左手: ({hands[0]['x']:.1f}, {hands[0]['y']:.1f}) 置信度={hands[0]['conf']:.2f}\n")
            f.write(f"右手: ({hands[1]['x']:.1f}, {hands[1]['y']:.1f}) 置信度={hands[1]['conf']:.2f}\n")
            f.write(f"距离: {distance:.1f} 像素\n")
    
    logger.debug(f"已处理 {img_name}: 距离 = {distance:.1f} px")
    
    return {
        'success': True,
        'filename': img_name,
        'distance': distance,
        'hands': hands
    }


def process_video(model, video_path, save_dir, save_txt, distance_threshold=1400, 
                  stable_duration=1.0, crop_ratio=0.3, quality=80, conf=0.6):
    """
    处理视频文件，检测手部距离并自动截图
    
    Args:
        model: YOLO模型对象
        video_path: 视频文件路径
        save_dir: 结果保存目录
        save_txt: 是否保存距离结果为txt文件
        distance_threshold: 触发截图的距离阈值（像素）
        stable_duration: 触发截图所需的稳定时长（秒）
        crop_ratio: 图像裁剪比例
        quality: 图像压缩质量
        conf: 置信度阈值
    
    Returns:
        dict: 处理结果统计信息
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return {'success': False, 'error': '无法打开视频文件', 'filename': os.path.basename(video_path)}
    
    fps, width, height, total_frames = (
        int(cap.get(cv2.CAP_PROP_FPS)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    need_frames = int(fps * stable_duration)
    
    video_name = os.path.basename(video_path)
    output_video_path = os.path.join(save_dir, f'hand_distance_{video_name}')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
    
    screenshots_dir = os.path.join(save_dir, 'screenshots')
    os.makedirs(screenshots_dir, exist_ok=True)
    
    distances, frame_distance_log, screenshot_distances = [], [], []
    continue_long_dist_frame, screenshot_count, last_screenshot_frame = 0, 0, -1000
    triggered_frames = []
    
    logger.info(f"正在处理视频: {video_name} (FPS={fps}, 距离阈值={distance_threshold}px, 稳定时长={stable_duration}s)")
    
    min_screenshot_interval = int(fps * 0.5)
    
    with tqdm(total=total_frames, desc="处理帧", unit="帧") as pbar:
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count == 0:
                save_screenshot(frame, frame_count, screenshots_dir, crop_ratio)
                screenshot_count += 1
                triggered_frames.append(frame_count)
                last_screenshot_frame = frame_count
            
            current_distance = None
            annotated = frame.copy()
            
            result, has_hands = detect_hands(model, frame, conf)
            
            if has_hands:
                distance, annotated, _ = calculate_hand_distance(frame, result.boxes)
                current_distance = distance
                distances.append(distance)
                
                if distance > distance_threshold:
                    continue_long_dist_frame += 1
                    cv2.putText(annotated, f"Count: {continue_long_dist_frame}/{need_frames}", 
                                (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    if continue_long_dist_frame >= need_frames:
                        if (frame_count - last_screenshot_frame) >= min_screenshot_interval:
                            save_screenshot(frame, frame_count, screenshots_dir, crop_ratio)
                            screenshot_count += 1
                            triggered_frames.append(frame_count)
                            screenshot_distances.append(distance)
                            last_screenshot_frame = frame_count
                        continue_long_dist_frame = 0
                else:
                    continue_long_dist_frame = 0
                    cv2.putText(annotated, f"Reset - Distance: {distance:.1f}px", 
                                (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(annotated, "Insufficient hands detected", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue_long_dist_frame = 0
            
            frame_distance_log.append((frame_count, current_distance))
            out.write(annotated)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    logger.info(f"正在去重截图...")
    filtered_frames, removed_frames = deduplicate_screenshots(screenshots_dir, triggered_frames, similarity_threshold=0.8)
    
    filtered_dir = os.path.join(save_dir, os.path.splitext(video_name)[0])
    convert_screenshots_to_jpg(screenshots_dir, filtered_frames, filtered_dir, quality)
    
    avg_distance = np.mean(distances) if distances else None
    avg_screenshot_distance = np.mean(screenshot_distances) if screenshot_distances else None
    album_ratio = avg_screenshot_distance / width if avg_screenshot_distance else None
    
    if save_txt:
        save_distance_summary(save_dir, video_name, fps, distance_threshold, stable_duration, need_frames,
                             frame_count, distances, screenshot_count, filtered_frames, removed_frames,
                             screenshot_distances, avg_screenshot_distance, width)
        save_frame_distance_log(save_dir, frame_distance_log, fps)
    
    logger.info(f"视频处理完成: {video_name}")
    logger.info(f"  截图统计: 捕获={screenshot_count}, 去重后={len(filtered_frames)}, 移除={len(removed_frames)}")
    
    return {
        'success': True,
        'filename': video_name,
        'output_video': output_video_path,
        'screenshots_dir': screenshots_dir,
        'total_frames': frame_count,
        'valid_distance_frames': len(distances),
        'avg_distance': avg_distance,
        'screenshot_count': screenshot_count,
        'filtered_count': len(filtered_frames),
        'removed_count': len(removed_frames),
        'avg_screenshot_distance': avg_screenshot_distance,
        'album_ratio': album_ratio,
        'width': width
    }


def run_hand_distance(model_path, source, conf, save_dir, save_txt, 
                      distance_threshold=1400, stable_duration=1.0, 
                      crop_ratio=0.3, quality=80, list_file=None):
    """
    运行手部距离计算主函数
    
    Args:
        model_path: 模型权重路径
        source: 源文件/目录路径
        conf: 置信度阈值
        save_dir: 结果保存目录
        save_txt: 是否保存txt日志
        distance_threshold: 距离阈值
        stable_duration: 稳定时长
        crop_ratio: 裁剪比例
        quality: 图像质量
        list_file: 可选，从txt文件批量导入源路径（每行一个），优先于source
    
    Returns:
        dict: 汇总统计信息
    """
    params = {
        'crop_ratio': crop_ratio,
        'quality': quality,
        'distance_threshold': distance_threshold,
        'stable_duration': stable_duration,
        'conf': conf
    }
    errors = validate_parameters(params)
    if errors:
        logger.error("参数校验失败:")
        for error in errors:
            logger.error(f"  - {error}")
        return {'success': False, 'errors': errors}
    
    model, device_info = load_yolo_model(model_path)
    if model is None:
        return {'success': False, 'error': '模型加载失败'}
    
    save_dir = get_next_exp_dir(save_dir) if save_dir else get_next_exp_dir('runs/hand_distance')
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"结果将保存到: {save_dir}")
    
    stats = {
        'total_files': 0,
        'success_count': 0,
        'failed_count': 0,
        'image_count': 0,
        'video_count': 0,
        'video_results': []
    }
    
    # 收集待处理的 (路径, 是否视频) 列表
    items, error = collect_source_items(source, list_file=list_file)
    if error:
        logger.error(str(error))
        return {'success': False, 'error': error}
    if list_file:
        logger.info(f"从txt导入 {len(items)} 个文件: {list_file}")

    stats['total_files'] = len(items)
    stats['video_count'] = sum(1 for _, is_video in items if is_video)
    stats['image_count'] = stats['total_files'] - stats['video_count']

    logger.info(f"待处理: 视频 {stats['video_count']} 个，图片 {stats['image_count']} 个")

    for path, is_video in items:
        if is_video:
            result = process_video(model, path, save_dir, save_txt, distance_threshold,
                                   stable_duration, crop_ratio, quality, conf)
            if result['success']:
                stats['success_count'] += 1
                stats['video_results'].append(result)
            else:
                stats['failed_count'] += 1
        else:
            logger.info(f"正在处理图片: {os.path.basename(path)}")
            result = process_single_image(model, path, save_dir, save_txt, crop_ratio, quality, conf)
            if result['success']:
                stats['success_count'] += 1
            else:
                stats['failed_count'] += 1

    logger.info("所有文件处理完成")
    
    print_summary(stats, save_dir)
    
    return stats


def print_summary(stats, save_dir):
    """
    打印汇总统计信息
    
    Args:
        stats: 统计信息字典
        save_dir: 保存目录
    """
    logger.info("="*50)
    logger.info("汇总统计")
    logger.info("="*50)
    logger.info(f"处理文件总数: {stats['total_files']}")
    logger.info(f"  - 视频文件: {stats['video_count']}")
    logger.info(f"  - 图片文件: {stats['image_count']}")
    logger.info(f"成功: {stats['success_count']}")
    logger.info(f"失败: {stats['failed_count']}")
    logger.info(f"结果保存到: {save_dir}")
    logger.info("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO手部距离计算器（自动截图）')
    parser.add_argument('--model', type=str, default=r'./weights/ultralytics/hand_yolov8n.pt', 
                        help='手部检测模型权重路径')
    parser.add_argument('--source', type=str, default=r"D:\cute aggression ういり画集 日版.mp4", 
                        help='源目录、图片路径或视频文件路径')
    parser.add_argument('--list-file', type=str, default=None,
                        help='从txt文件批量导入源路径（每行一个，支持#注释和空行），优先于--source')
    parser.add_argument('--conf', type=float, default=0.6, help='置信度阈值')
    parser.add_argument('--save-dir', type=str, default='runs\\hand_distance', 
                        help='输出目录（未指定时自动递增）')
    parser.add_argument('--no-save-txt', action='store_true', help='不保存距离结果为txt文件')
    parser.add_argument('--distance-threshold', type=int, default=1400, 
                        help='触发截图的距离阈值（像素）')
    parser.add_argument('--stable-duration', type=float, default=2, 
                        help='触发截图所需的稳定时长（秒）')
    parser.add_argument('--crop-ratio', type=float, default=0.2, 
                        help='图像两边向中央裁剪的总比例（默认0，即不裁剪）')
    parser.add_argument('--quality', type=int, default=100, 
                        help='图像压缩质量（1-100，值越高质量越好）')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    
    logger.info("="*50)
    logger.info("YOLO手部距离计算器")
    logger.info("="*50)
    logger.info(f"模型: {args.model}")
    if args.list_file:
        logger.info(f"源列表txt: {args.list_file}")
    else:
        logger.info(f"源: {args.source}")
    logger.info(f"置信度阈值: {args.conf}")
    logger.info(f"距离阈值: {args.distance_threshold} px")
    logger.info(f"稳定时长: {args.stable_duration} 秒")
    logger.info(f"裁剪比例: {args.crop_ratio}")
    logger.info(f"压缩质量: {args.quality}")
    logger.info("="*50)
    
    run_hand_distance(args.model, args.source, args.conf, args.save_dir, 
                      not args.no_save_txt, args.distance_threshold, 
                      args.stable_duration, args.crop_ratio, args.quality,
                      list_file=args.list_file)