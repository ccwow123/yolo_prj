import os
import cv2
import numpy as np
import logging

# logger
logger = logging.getLogger(__name__)


def detect_hands(model, frame, conf):
    """
    检测手部（公共函数）
    
    Args:
        model: YOLO模型对象
        frame: 图像帧（numpy数组）
        conf: 置信度阈值
    
    Returns:
        result: YOLO检测结果
        has_hands: 是否检测到至少2只手
    """
    results = model(frame, verbose=False, conf=conf)
    result = results[0]
    has_hands = result.boxes is not None and len(result.boxes) >= 2
    return result, has_hands


def calculate_hand_distance(image, boxes):
    """
    计算双手之间的距离并标注图像
    
    Args:
        image: 原始图像（numpy数组）
        boxes: YOLO检测到的边界框列表
    
    Returns:
        distance: 双手之间的像素距离（None如果不足2只手）
        annotated: 带有标注的图像
        hands: 左右手信息列表
    """
    if len(boxes) < 2:
        return None, image, []
    
    hands = [{'x': float(box.xywh[0][0]), 'y': float(box.xywh[0][1]), 'conf': float(box.conf)} for box in boxes]
    hands.sort(key=lambda h: h['x'])
    
    left_hand, right_hand = hands[0], hands[-1]
    distance = np.sqrt((right_hand['x'] - left_hand['x'])**2 + (right_hand['y'] - left_hand['y'])**2)
    
    annotated = image.copy()
    cv2.circle(annotated, (int(left_hand['x']), int(left_hand['y'])), 8, (0, 0, 255), -1)
    cv2.putText(annotated, f'Left ({left_hand["conf"]:.2f})', (int(left_hand['x']) - 50, int(left_hand['y']) - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.circle(annotated, (int(right_hand['x']), int(right_hand['y'])), 8, (0, 255, 0), -1)
    cv2.putText(annotated, f'Right ({right_hand["conf"]:.2f})', (int(right_hand['x']) - 50, int(right_hand['y']) - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.line(annotated, (int(left_hand['x']), int(left_hand['y'])), (int(right_hand['x']), int(right_hand['y'])), (255, 0, 0), 2)
    
    mid_x, mid_y = (left_hand['x'] + right_hand['x']) // 2, (left_hand['y'] + right_hand['y']) // 2
    cv2.putText(annotated, f'Distance: {distance:.1f}px', (int(mid_x) - 50, int(mid_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return distance, annotated, [left_hand, right_hand]


def crop_image(image, crop_ratio=0.3):
    """
    按比例裁剪图像两边
    
    Args:
        image: 输入图像（numpy数组）
        crop_ratio: 图像两边向中央裁剪的总比例
    
    Returns:
        裁剪后的图像
    """
    if crop_ratio <= 0:
        return image
    height, width = image.shape[:2]
    half_crop = crop_ratio / 2
    return image[:, int(width * half_crop):int(width * (1 - half_crop))]


def save_screenshot(frame, frame_count, screenshots_dir, crop_ratio):
    """
    保存截图到指定目录
    
    Args:
        frame: 图像帧（numpy数组）
        frame_count: 帧序号
        screenshots_dir: 截图保存目录
        crop_ratio: 裁剪比例
    """
    cropped_frame = crop_image(frame, crop_ratio)
    screenshot_path = os.path.join(screenshots_dir, f'screenshot_{frame_count:06d}.png')
    cv2.imwrite(screenshot_path, cropped_frame)


def deduplicate_screenshots(screenshots_dir, triggered_frames, similarity_threshold=0.8):
    """
    使用ORB特征匹配去除相似截图（委托给 image_dedup.deduplicate_screenshotsV2）

    Args:
        screenshots_dir: 截图目录
        triggered_frames: 触发截图的帧号列表
        similarity_threshold: 相似度阈值（越低越严格）

    Returns:
        filtered_frames: 去重后的帧号列表
        removed_frames: 被移除的帧号列表
    """
    from .image_dedup import deduplicate_screenshotsV2

    return deduplicate_screenshotsV2(
        screenshots_dir,
        triggered_frames=triggered_frames,
        similarity_threshold=similarity_threshold,
        method="orb",
        ext=".png"
    )


def save_distance_summary(save_dir, video_name, fps, distance_threshold, stable_duration, need_frames,
                          frame_count, distances, screenshot_count, filtered_frames, removed_frames,
                          screenshot_distances, avg_screenshot_distance, width):
    """
    保存距离统计摘要文件
    
    Args:
        save_dir: 保存目录
        video_name: 视频文件名
        fps: 视频帧率
        distance_threshold: 距离阈值
        stable_duration: 稳定时长
        need_frames: 需要的连续帧数
        frame_count: 处理的总帧数
        distances: 所有距离列表
        screenshot_count: 捕获截图总数
        filtered_frames: 去重后的帧号列表
        removed_frames: 被移除的帧号列表
        screenshot_distances: 截图时的距离列表
        avg_screenshot_distance: 截图时平均距离
        width: 图像宽度
    """
    album_ratio = avg_screenshot_distance / width if avg_screenshot_distance else None
    
    with open(os.path.join(save_dir, 'distance_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"视频: {video_name}\nFPS: {fps}\n距离阈值: {distance_threshold} px\n稳定时长: {stable_duration} 秒 ({need_frames} 帧)\n")
        f.write(f"处理总帧数: {frame_count}\n有效距离帧数: {len(distances)}\n")
        if distances:
            f.write(f"平均距离: {np.mean(distances):.1f} px\n最小距离: {np.min(distances):.1f} px\n最大距离: {np.max(distances):.1f} px\n")
        f.write(f"\n=== 截图统计 ===\n捕获截图总数: {screenshot_count}\n去重后截图数: {len(filtered_frames)}\n已移除重复截图: {len(removed_frames)}\n")
        if avg_screenshot_distance:
            f.write(f"\n=== 截图时双手距离统计 ===\n截图时距离列表: {', '.join([f'{d:.1f}' for d in screenshot_distances])} px\n")
            f.write(f"截图时平均距离: {avg_screenshot_distance:.1f} px\n图像宽度: {width} px\n")
            f.write(f"画册所占比例:{album_ratio * 100:.2f}%,建议剪裁比例为({1-album_ratio:.2f})\n")
        if filtered_frames:
            f.write(f"\n最终触发帧: {', '.join(map(str, filtered_frames))}\n最终触发时间戳(秒):\n")
            for frame in filtered_frames:
                f.write(f"  帧 {frame}: {frame/fps:.2f}秒\n")
        if removed_frames:
            f.write(f"\n已移除重复帧: {', '.join(map(str, removed_frames))}\n已移除时间戳(秒):\n")
            for frame in removed_frames:
                f.write(f"  帧 {frame}: {frame/fps:.2f}秒\n")


def save_frame_distance_log(save_dir, frame_distance_log, fps):
    """
    保存每帧距离日志文件
    
    Args:
        save_dir: 保存目录
        frame_distance_log: 帧距离日志列表
        fps: 视频帧率
    """
    with open(os.path.join(save_dir, 'frame_distance_log.txt'), 'w', encoding='utf-8') as f:
        f.write("# 帧,时间(秒),距离(px)\n")
        for frame_num, distance in frame_distance_log:
            f.write(f"{frame_num},{frame_num/fps:.3f},{distance:.1f}\n" if distance else f"{frame_num},{frame_num/fps:.3f},-1\n")


def convert_screenshots_to_jpg(screenshots_dir, filtered_frames, output_dir, quality=80):
    """
    将PNG截图转换为JPG格式
    
    Args:
        screenshots_dir: PNG截图目录
        filtered_frames: 去重后的帧号列表
        output_dir: JPG输出目录
        quality: JPEG压缩质量（1-100）
    
    Returns:
        转换成功的文件数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    count = 0
    for frame_num in filtered_frames:
        src_path = os.path.join(screenshots_dir, f'screenshot_{frame_num:06d}.png')
        if os.path.exists(src_path):
            image = cv2.imread(src_path)
            dst_path = os.path.join(output_dir, f'screenshot_{frame_num:06d}.jpg')
            cv2.imwrite(dst_path, image, jpeg_params)
            count += 1
    
    logger.debug(f"已转换 {count} 张截图到 {output_dir}")
    return count