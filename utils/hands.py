import argparse
import logging
import os

import cv2
import numpy as np
from tqdm import tqdm

from .core import get_next_exp_dir, load_yolo_model, imread_unicode
from .similarity import deduplicate_screenshotsV2

logger = logging.getLogger(__name__)


# ------------------------- 手部检测 / 距离 -------------------------
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


class _ScaledBox:
    """降采样推理后映射回原分辨率的轻量检测框，兼容 calculate_hand_distance。"""
    __slots__ = ('_xywh', '_conf')

    def __init__(self, cx, cy, w, h, conf):
        self._xywh = np.array([[cx, cy, w, h]], dtype=np.float32)
        self._conf = conf

    @property
    def xywh(self):
        return self._xywh

    @property
    def conf(self):
        return self._conf


def detect_hands_scaled(model, frame, conf, max_edge=None, imgsz=None, half=False):
    """
    在可选降采样的帧上检测手部，并把检测框坐标映射回原始分辨率。

    推理 FLOPs 近似随模型前向尺寸 imgsz 的平方下降；对 1080p 及以上输入，
    设置 imgsz（如 480/320）能直接减少网络计算量，比单纯预缩放帧更有效。
    坐标按缩放比例放大回原图，距离/截图等下游计算的像素语义保持不变。

    Args:
        model: YOLO模型对象
        frame: 原始分辨率 BGR 帧
        conf: 置信度阈值
        max_edge: 输入帧最长边像素；None 或大于原图最长边时不预缩放
        imgsz: 模型前向尺寸（如 480、320）；None 用模型默认 imgsz
        half: 是否启用 FP16 推理（仅 CUDA 生效，新版以 quantize=16 传入）

    Returns:
        (boxes_list, has_hands):
            boxes_list: 与原分辨率坐标对齐的轻量框列表，可直接传给 calculate_hand_distance；
                        未检测到足够的手时为 None
            has_hands: 是否检测到至少2只手
    """
    h_img, w_img = frame.shape[:2]
    infer_frame = frame
    if max_edge and max(h_img, w_img) > max_edge:
        scale = max_edge / max(h_img, w_img)
        infer_frame = cv2.resize(
            frame,
            (max(1, int(w_img * scale)), max(1, int(h_img * scale))),
            interpolation=cv2.INTER_AREA,
        )

    import torch
    use_half = half and torch.cuda.is_available()
    # 新版 ultralytics 以 quantize 取代已废弃的 half；16=FP16，32/省略=FP32
    kwargs = {'conf': conf}
    if use_half:
        kwargs['quantize'] = 16
    if imgsz:
        kwargs['imgsz'] = imgsz
    with torch.no_grad():
        results = model(infer_frame, verbose=False, **kwargs)

    result = results[0]
    if result.boxes is None or len(result.boxes) < 2:
        return None, False

    sx = w_img / infer_frame.shape[1]
    sy = h_img / infer_frame.shape[0]
    boxes = []
    for box in result.boxes:
        cx, cy, bw, bh = box.xywh[0].tolist()
        boxes.append(_ScaledBox(cx * sx, cy * sy, bw * sx, bh * sy, float(box.conf)))
    return boxes, True


def calculate_hand_distance(image, boxes, annotate=True):
    """
    计算双手之间的距离并标注图像

    Args:
        image: 原始图像（numpy数组）
        boxes: YOLO检测到的边界框列表
        annotate: 是否在图像上绘制手点/距离标注；False 时只算距离，annotated 为 None，
                  省去整帧拷贝与绘制开销

    Returns:
        distance: 双手之间的像素距离（None如果不足2只手）
        annotated: 带有标注的图像（annotate=False 时为 None）；不足2只手时返回 image
        hands: 左右手信息列表
    """
    if len(boxes) < 2:
        return None, image, []

    hands = [{'x': float(box.xywh[0][0]), 'y': float(box.xywh[0][1]), 'conf': float(box.conf)} for box in boxes]
    hands.sort(key=lambda h: h['x'])

    left_hand, right_hand = hands[0], hands[-1]
    distance = np.sqrt((right_hand['x'] - left_hand['x'])**2 + (right_hand['y'] - left_hand['y'])**2)

    if not annotate:
        return distance, None, [left_hand, right_hand]

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
    使用ORB特征匹配去除相似截图（委托给 deduplicate_screenshotsV2）

    Args:
        screenshots_dir: 截图目录
        triggered_frames: 触发截图的帧号列表
        similarity_threshold: 相似度阈值（越低越严格）

    Returns:
        filtered_frames: 去重后的帧号列表
        removed_frames: 被移除的帧号列表
    """
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
            image = imread_unicode(src_path)
            dst_path = os.path.join(output_dir, f'screenshot_{frame_num:06d}.jpg')
            cv2.imwrite(dst_path, image, jpeg_params)
            count += 1

    logger.debug(f"已转换 {count} 张截图到 {output_dir}")
    return count


# ------------------------- 手部移除（分割 + 白化） -------------------------
def remove_hands(image_path, model, conf=0.4, book_width_ratio=0.0, min_area=5000,
                 max_area=100000, aspect_ratio_range=(0.3, 3.0), visualize_mask=False):
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h_img, w_img = img.shape[:2]

    book_area = None
    if book_width_ratio > 0:
        book_width = int(w_img * book_width_ratio)
        book_x1 = (w_img - book_width) // 2
        book_x2 = book_x1 + book_width
        book_area = (book_x1, 0, book_x2, h_img)
        print(f"  自动计算书页区域: ({book_x1}, 0) - ({book_x2}, {h_img})")

    results = model.predict(image_path, conf=conf, iou=0.45, verbose=False)

    hand_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    for res in results:
        if res.masks is None:
            continue

        for mask, box in zip(res.masks.data, res.boxes):
            cls_id = int(box.cls)
            cls_name = res.names[cls_id]

            seg_mask = cv2.resize(mask.cpu().numpy(), (w_img, h_img))

            area = np.sum(seg_mask > 0.5)

            if area < min_area or area > max_area:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            aspect_ratio = bbox_w / bbox_h

            if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                continue

            moments = cv2.moments((seg_mask > 0.5).astype(np.uint8))
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            if book_area is not None:
                book_x1, book_y1, book_x2, book_y2 = book_area
                if not (book_x1 < cx < book_x2 and book_y1 < cy < book_y2):
                    hand_mask[seg_mask > 0.5] = 255
            else:
                hand_mask[seg_mask > 0.5] = 255

    img_no_hand = img.copy()

    if np.any(hand_mask == 255):
        img_no_hand[hand_mask == 255] = [255, 255, 255]
        print("  手部区域已填充白色")
    else:
        print("  未检测到手部")

    plot_img = None
    if len(results) > 0:
        plot_img = res.plot()

    return img_no_hand, hand_mask, plot_img


def create_comparison(img_original, img_result, mask):
    h, w = img_original.shape[:2]

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_rgb[mask == 255] = [0, 0, 255]

    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)

    combined[:, :w] = img_original
    combined[:, w:w*2] = mask_rgb
    combined[:, w*2:] = img_result

    cv2.putText(combined, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Hand Mask", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Result", (w * 2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return combined


def process_folder(input_dir, output_dir, model_path, conf=0.4, book_width_ratio=0.0,
                   min_area=5000, max_area=100000, aspect_ratio_range=(0.3, 3.0), visualize=False):
    model, device_info = load_yolo_model(model_path)
    if model is None:
        raise RuntimeError(f"无法加载模型: {model_path}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(image_extensions)])

    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    if visualize:
        compare_dir = os.path.join(output_dir, 'comparison')
        os.makedirs(compare_dir, exist_ok=True)
        seg_dir = os.path.join(output_dir, 'segmentation')
        os.makedirs(seg_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    print(f"发现 {len(files)} 张图片")
    print(f"使用模型: {model_path}")
    print(f"置信度阈值: {conf}")
    print(f"面积范围: {min_area} - {max_area}")
    print(f"宽高比范围: {aspect_ratio_range[0]} - {aspect_ratio_range[1]}")
    if book_width_ratio > 0:
        print(f"书页宽度比例: {book_width_ratio * 100:.1f}% (自动居中)")
        print("仅处理书页区域外的手部")

    with tqdm(total=len(files), desc="处理图片", unit="张") as pbar:
        for filename in files:
            try:
                input_path = os.path.join(input_dir, filename)
                name, ext = os.path.splitext(filename)

                img_no_hand, mask, plot_img = remove_hands(
                    input_path, model, conf, book_width_ratio, min_area, max_area, aspect_ratio_range
                )

                output_path = os.path.join(output_dir, f"{name}_nohand{ext}")
                if ext.lower() in ['.jpg', '.jpeg']:
                    cv2.imwrite(output_path, img_no_hand, [cv2.IMWRITE_JPEG_QUALITY, 85])
                else:
                    cv2.imwrite(output_path, img_no_hand)

                mask_path = os.path.join(mask_dir, f"{name}_mask.png")
                cv2.imwrite(mask_path, mask)

                if visualize:
                    original = imread_unicode(input_path)
                    comparison = create_comparison(original, img_no_hand, mask)
                    comparison_path = os.path.join(compare_dir, f"{name}_compare.png")
                    cv2.imwrite(comparison_path, comparison)

                    if plot_img is not None:
                        seg_path = os.path.join(seg_dir, f"{name}_seg.png")
                        cv2.imwrite(seg_path, plot_img)

                status = "含手" if np.any(mask > 0) else "无手"
                success_count += 1
                print(f"  ✓ {filename} -> {name}_nohand{ext} [{status}]")

            except Exception as e:
                fail_count += 1
                print(f"  ✗ {filename} -> 失败: {str(e)}")

            pbar.update(1)

    report_path = os.path.join(output_dir, 'processing_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 手部移除处理报告 ===\n\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"使用模型: {model_path}\n")
        f.write(f"置信度阈值: {conf}\n")
        f.write(f"面积范围: {min_area} - {max_area}\n")
        f.write(f"宽高比范围: {aspect_ratio_range[0]} - {aspect_ratio_range[1]}\n")
        if book_width_ratio > 0:
            f.write(f"书页宽度比例: {book_width_ratio * 100:.1f}%\n")
        f.write(f"处理总数: {len(files)}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {fail_count}\n")
        f.write(f"成功率: {success_count / len(files) * 100:.1f}%\n")

    print(f"\n=== 处理完成 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理总数: {len(files)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='使用YOLO手部分割模型移除图片中的手部')
    parser.add_argument('--input', type=str, default=r'./runs/hand_distance/exp9/screenshots', help='输入图片目录')
    parser.add_argument('--output', type=str, default=r'./runs/remove_hands/', help='输出目录')
    parser.add_argument('--model', type=str, default=r'./weights/ultralytics/yolov8s-seg.pt', help='手部分割模型路径')
    parser.add_argument('--conf', type=float, default=0.4, help='置信度阈值，越高误检越少')
    parser.add_argument('--min-area', type=int, default=5000, help='最小面积阈值，过滤小面积误检')
    parser.add_argument('--max-area', type=int, default=100000, help='最大面积阈值，过滤大面积误检')
    parser.add_argument('--aspect-ratio', type=float, nargs=2, default=[0.3, 3.0],
                        help='宽高比范围，过滤畸形检测框')
    parser.add_argument('--book-width-ratio', type=float, default=0.7,
                        help='书页宽度占原图宽度的比例，默认0.7，自动居中')
    parser.add_argument('--visualize', action='store_true', help='生成对比图和分割可视化')

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return

    output_dir = get_next_exp_dir(args.output)
    print(f"输出目录: {output_dir}")

    process_folder(
        args.input,
        output_dir,
        args.model,
        conf=args.conf,
        book_width_ratio=args.book_width_ratio,
        min_area=args.min_area,
        max_area=args.max_area,
        aspect_ratio_range=tuple(args.aspect_ratio),
        visualize=args.visualize
    )


if __name__ == '__main__':
    main()