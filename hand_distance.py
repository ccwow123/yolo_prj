import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

def get_next_exp_dir(base_dir='runs/hand_distance'):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith('exp')]
    
    if not existing_dirs:
        return os.path.join(base_dir, 'exp')
    
    max_num = 0
    for d in existing_dirs:
        if d == 'exp':
            num = 1
        elif d.startswith('exp') and d[3:].isdigit():
            num = int(d[3:])
        else:
            continue
        if num > max_num:
            max_num = num
    
    if max_num == 0:
        return os.path.join(base_dir, 'exp')
    else:
        return os.path.join(base_dir, f'exp{max_num + 1}')

def calculate_hand_distance(image, boxes):
    if len(boxes) < 2:
        return None, image, []
    
    hands = []
    for box in boxes:
        x_center = float(box.xywh[0][0])
        y_center = float(box.xywh[0][1])
        conf = float(box.conf)
        hands.append({'x': x_center, 'y': y_center, 'conf': conf})
    
    hands.sort(key=lambda h: h['x'])
    
    left_hand = hands[0]
    right_hand = hands[-1]
    
    distance = np.sqrt((right_hand['x'] - left_hand['x'])**2 + (right_hand['y'] - left_hand['y'])**2)
    
    annotated = image.copy()
    
    cv2.circle(annotated, (int(left_hand['x']), int(left_hand['y'])), 8, (0, 0, 255), -1)
    cv2.putText(annotated, f'Left ({left_hand["conf"]:.2f})', 
                (int(left_hand['x']) - 50, int(left_hand['y']) - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.circle(annotated, (int(right_hand['x']), int(right_hand['y'])), 8, (0, 255, 0), -1)
    cv2.putText(annotated, f'Right ({right_hand["conf"]:.2f})', 
                (int(right_hand['x']) - 50, int(right_hand['y']) - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.line(annotated, 
             (int(left_hand['x']), int(left_hand['y'])), 
             (int(right_hand['x']), int(right_hand['y'])), 
             (255, 0, 0), 2)
    
    mid_x = (left_hand['x'] + right_hand['x']) // 2
    mid_y = (left_hand['y'] + right_hand['y']) // 2
    cv2.putText(annotated, f'Distance: {distance:.1f}px', 
                (int(mid_x) - 50, int(mid_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return distance, annotated, [left_hand, right_hand]

def crop_image(image, crop_ratio=0.3):
    if crop_ratio <= 0:
        return image
    height, width = image.shape[:2]
    half_crop = crop_ratio / 2
    start_x = int(width * half_crop)
    end_x = int(width * (1 - half_crop))
    return image[:, start_x:end_x]

def compress_image(image, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def process_single_image(model, image_path, save_dir, save_txt, crop_ratio=0.3, quality=80, conf=0.6):
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
    results = model(image, verbose=False, conf=conf)
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        print(f"在 {os.path.basename(image_path)} 中未检测到手部")
        return
    
    distance, annotated, hands = calculate_hand_distance(image, result.boxes)
    
    annotated = crop_image(annotated, crop_ratio)
    annotated = compress_image(annotated, quality)
    
    img_name = os.path.basename(image_path)
    name, ext = os.path.splitext(img_name)
    output_path = os.path.join(save_dir, f'{name}.jpg')
    cv2.imwrite(output_path, annotated)
    
    if save_txt and distance is not None:
        txt_path = os.path.join(save_dir, os.path.splitext(img_name)[0] + '_distance.txt')
        with open(txt_path, 'w') as f:
            f.write(f"图片: {img_name}\n")
            f.write(f"左手: ({hands[0]['x']:.1f}, {hands[0]['y']:.1f}) 置信度={hands[0]['conf']:.2f}\n")
            f.write(f"右手: ({hands[1]['x']:.1f}, {hands[1]['y']:.1f}) 置信度={hands[1]['conf']:.2f}\n")
            f.write(f"距离: {distance:.1f} 像素\n")
    
    if distance is not None:
        print(f"已处理 {img_name}: 距离 = {distance:.1f} px")
    else:
        print(f"已处理 {img_name}: 检测到少于2只手")

def orb_distance(image1, image2):
    orb = cv2.ORB_create(nfeatures=500)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 1.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) == 0:
        return 1.0
    
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]
    
    if max(len(kp1), len(kp2)) == 0:
        return 1.0
    
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    return 1.0 - similarity

def deduplicate_screenshots(screenshots_dir, triggered_frames, similarity_threshold=0.7):
    if len(triggered_frames) <= 1:
        return triggered_frames, []
    
    print(f"去重设置: 相似度阈值={similarity_threshold}")
    print(f"原始触发帧: {triggered_frames}")
    
    filtered_frames = []
    removed_frames = []
    
    if triggered_frames:
        last_kept_frame = triggered_frames[0]
        last_kept_path = os.path.join(screenshots_dir, f'screenshot_{last_kept_frame:06d}.jpg')
        last_kept_image = cv2.imread(last_kept_path)
        
        for frame in triggered_frames[1:]:
            current_path = os.path.join(screenshots_dir, f'screenshot_{frame:06d}.jpg')
            current_image = cv2.imread(current_path)
            
            if current_image is None or last_kept_image is None:
                removed_frames.append(last_kept_frame)
                old_path = os.path.join(screenshots_dir, f'screenshot_{last_kept_frame:06d}.jpg')
                if os.path.exists(old_path):
                    os.remove(old_path)
                last_kept_frame = frame
                last_kept_image = current_image
                print(f"  替换帧 {last_kept_frame} (图片读取错误)")
                continue
            
            distance = orb_distance(last_kept_image, current_image)
            
            if distance <= similarity_threshold:
                removed_frames.append(last_kept_frame)
                old_path = os.path.join(screenshots_dir, f'screenshot_{last_kept_frame:06d}.jpg')
                if os.path.exists(old_path):
                    os.remove(old_path)
                last_kept_frame = frame
                last_kept_image = current_image
                # print(f"  替换帧 {frame} (与帧 {last_kept_frame} 相似, 保留最新)")
            else:
                filtered_frames.append(last_kept_frame)
                last_kept_frame = frame
                last_kept_image = current_image
                # print(f"  保留帧 {last_kept_frame} (与前一帧不同)")
        
        filtered_frames.append(last_kept_frame)
    
    print(f"过滤后帧: {filtered_frames}")
    print(f"已删除帧: {removed_frames}")
    
    return filtered_frames, removed_frames

def process_video(model, video_path, save_dir, save_txt, distance_threshold=1400, stable_duration=1.0, crop_ratio=0.3, quality=80, conf=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    need_frames = int(fps * stable_duration)
    print(f"视频FPS: {fps}, 需要 {need_frames} 连续帧以满足 {stable_duration}秒触发条件")
    
    video_name = os.path.basename(video_path)
    output_video_path = os.path.join(save_dir, f'hand_distance_{video_name}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out_width = int(width * (1 - crop_ratio)) if crop_ratio > 0 else width
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, height), True)
    
    screenshots_dir = os.path.join(save_dir, 'screenshots')
    os.makedirs(screenshots_dir, exist_ok=True)
    
    distances = []
    frame_distance_log = []
    screenshot_distances = []
    continue_long_dist_frame = 0
    screenshot_count = 0
    triggered_frames = []
    
    print(f"正在处理视频: {video_name}")
    print(f"距离阈值: {distance_threshold} px")
    print(f"稳定时长: {stable_duration} 秒")
    
    with tqdm(total=total_frames, desc="处理帧", unit="frame") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count == 0:
                cropped_frame = crop_image(frame, crop_ratio)
                cropped_frame = compress_image(cropped_frame, quality)
                screenshot_path = os.path.join(screenshots_dir, f'screenshot_{frame_count:06d}.jpg')
                cv2.imwrite(screenshot_path, cropped_frame)
                screenshot_count += 1
                triggered_frames.append(frame_count)
                print(f"\n首帧截图已保存到 {screenshot_path}")
            
            results = model(frame, verbose=False, conf=conf)
            result = results[0]
            
            current_distance = None
            if result.boxes is not None and len(result.boxes) >= 2:
                distance, annotated, hands = calculate_hand_distance(frame, result.boxes)
                current_distance = distance
                if distance is not None:
                    distances.append(distance)
                    
                    if distance > distance_threshold:
                        continue_long_dist_frame += 1
                        
                        cv2.putText(annotated, f"Count: {continue_long_dist_frame}/{need_frames}", 
                                    (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        if continue_long_dist_frame >= need_frames:
                            cropped_frame = crop_image(frame, crop_ratio)
                            cropped_frame = compress_image(cropped_frame, quality)
                            screenshot_path = os.path.join(screenshots_dir, f'screenshot_{frame_count:06d}.jpg')
                            cv2.imwrite(screenshot_path, cropped_frame)
                            screenshot_count += 1
                            triggered_frames.append(frame_count)
                            screenshot_distances.append(distance)
                            print(f"\n在帧 {frame_count} 触发截图！已保存到 {screenshot_path} (距离: {distance:.1f}px)")
                            continue_long_dist_frame = 0
                    else:
                        continue_long_dist_frame = 0
                        cv2.putText(annotated, f"Reset - Distance: {distance:.1f}px", 
                                    (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                annotated = frame.copy()
                cv2.putText(annotated, "Insufficient hands detected", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue_long_dist_frame = 0
            
            frame_distance_log.append((frame_count, current_distance))
            
            annotated = crop_image(annotated, crop_ratio)
            out.write(annotated)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print("\n正在使用ORB相似度进行去重...")
    filtered_frames, removed_frames = deduplicate_screenshots(screenshots_dir, triggered_frames, similarity_threshold=0.7)
    
    if save_txt:
        summary_path = os.path.join(save_dir, 'distance_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"视频: {video_name}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"距离阈值: {distance_threshold} px\n")
            f.write(f"稳定时长: {stable_duration} 秒 ({need_frames} 帧)\n")
            f.write(f"处理总帧数: {frame_count}\n")
            f.write(f"有效距离帧数: {len(distances)}\n")
            if distances:
                f.write(f"平均距离: {np.mean(distances):.1f} px\n")
                f.write(f"最小距离: {np.min(distances):.1f} px\n")
                f.write(f"最大距离: {np.max(distances):.1f} px\n")
            f.write(f"\n=== 截图统计 ===\n")
            f.write(f"捕获截图总数: {screenshot_count}\n")
            f.write(f"去重后截图数: {len(filtered_frames)}\n")
            f.write(f"已移除重复截图: {len(removed_frames)}\n")
            if screenshot_distances:
                avg_screenshot_distance = np.mean(screenshot_distances)
                f.write(f"\n=== 截图时双手距离统计 ===\n")
                f.write(f"截图时距离列表: {', '.join([f'{d:.1f}' for d in screenshot_distances])} px\n")
                f.write(f"截图时平均距离: {avg_screenshot_distance:.1f} px\n")
                f.write(f"图像宽度: {width} px\n")
                album_ratio = avg_screenshot_distance / width
                f.write(f"画册所占比例:{album_ratio * 100:.2f}%,建议剪裁比例为({1-album_ratio:.2f})\n")
            if filtered_frames:
                f.write(f"\n最终触发帧: {', '.join(map(str, filtered_frames))}\n")
                f.write("最终触发时间戳(秒):\n")
                for frame in filtered_frames:
                    timestamp = frame / fps
                    f.write(f"  帧 {frame}: {timestamp:.2f}秒\n")
            if removed_frames:
                f.write(f"\n已移除重复帧: {', '.join(map(str, removed_frames))}\n")
                f.write("已移除时间戳(秒):\n")
                for frame in removed_frames:
                    timestamp = frame / fps
                    f.write(f"  帧 {frame}: {timestamp:.2f}秒\n")
        
        frame_log_path = os.path.join(save_dir, 'frame_distance_log.txt')
        with open(frame_log_path, 'w') as f:
            f.write("# 帧,时间(秒),距离(px)\n")
            for frame_num, distance in frame_distance_log:
                time_sec = frame_num / fps
                if distance is not None:
                    f.write(f"{frame_num},{time_sec:.3f},{distance:.1f}\n")
                else:
                    f.write(f"{frame_num},{time_sec:.3f},-1\n")
    
    print(f"\n=== 视频处理完成 ===")
    print(f"输出视频: {output_video_path}")
    print(f"截图保存到: {screenshots_dir}")
    print(f"\n=== 截图统计 ===")
    print(f"捕获截图总数: {screenshot_count}")
    print(f"去重后截图数: {len(filtered_frames)}")
    print(f"已移除重复截图: {len(removed_frames)}")
    # if filtered_frames:
    #     print(f"\n最终触发帧: {filtered_frames}")
    #     print("最终触发时间戳:")
    #     for frame in filtered_frames:
    #         print(f"  帧 {frame}: {frame / fps:.2f}秒")
    if distances:
        print(f"\n平均距离: {np.mean(distances):.1f} px")
    
    if screenshot_distances:
        avg_screenshot_distance = np.mean(screenshot_distances)
        album_ratio = avg_screenshot_distance / width
        print(f"\n=== 截图时双手距离统计 ===")
        # print(f"截图时距离列表: {', '.join([f'{d:.1f}' for d in screenshot_distances])} px")
        print(f"截图时平均距离: {avg_screenshot_distance:.1f} px")
        print(f"图像宽度: {width} px")
        print(f"画册所占比例:{album_ratio * 100:.2f}%,建议剪裁比例为({1-album_ratio:.2f})\n")

def validate_parameters(crop_ratio, quality, distance_threshold, stable_duration, conf):
    errors = []
    if crop_ratio < 0 or crop_ratio >= 1:
        errors.append(f"裁剪比例 crop_ratio 必须在 [0, 1) 范围内，当前值: {crop_ratio}")
    if quality < 1 or quality > 100:
        errors.append(f"压缩质量 quality 必须在 [1, 100] 范围内，当前值: {quality}")
    if distance_threshold < 0:
        errors.append(f"距离阈值 distance_threshold 必须大于等于0，当前值: {distance_threshold}")
    if stable_duration <= 0:
        errors.append(f"稳定时长 stable_duration 必须大于0，当前值: {stable_duration}")
    if conf < 0 or conf > 1:
        errors.append(f"置信度 conf 必须在 [0, 1] 范围内，当前值: {conf}")
    return errors

def is_video_file(filepath):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')
    return filepath.lower().endswith(video_extensions)

def run_hand_distance(model_path, source, conf, save_dir, save_txt, distance_threshold=1400, stable_duration=1.0, crop_ratio=0.3, quality=80):
    errors = validate_parameters(crop_ratio, quality, distance_threshold, stable_duration, conf)
    if errors:
        print("参数校验失败:")
        for error in errors:
            print(f"  - {error}")
        return
    
    if not os.path.exists(source):
        print(f"错误：源文件/目录 {source} 不存在")
        return
    
    print(f"\n=== 加载模型 ===")
    try:
        model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
        
        device = model.device
        if device.type == 'cuda':
            print(f"使用设备: GPU (CUDA)")
            print(f"GPU名称: {torch.cuda.get_device_name(device.index)}")
        else:
            print(f"使用设备: CPU")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    if save_dir is None:
        save_dir = get_next_exp_dir()
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    if os.path.isfile(source):
        if is_video_file(source):
            process_video(model, source, save_dir, save_txt, distance_threshold, stable_duration, crop_ratio, quality, conf)
        else:
            process_single_image(model, source, save_dir, save_txt, crop_ratio, quality, conf)
    
    elif os.path.isdir(source):
        files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
        
        if not files:
            print(f"警告：目录 {source} 为空")
            return
        
        video_files = [f for f in files if is_video_file(os.path.join(source, f))]
        image_files = [f for f in files if not is_video_file(os.path.join(source, f))]
        
        print(f"\n发现 {len(video_files)} 个视频文件，{len(image_files)} 个图片文件")
        
        with tqdm(total=len(files), desc="处理文件", unit="file") as pbar:
            for filename in files:
                filepath = os.path.join(source, filename)
                if is_video_file(filepath):
                    process_video(model, filepath, save_dir, save_txt, distance_threshold, stable_duration, crop_ratio, quality, conf)
                else:
                    process_single_image(model, filepath, save_dir, save_txt, crop_ratio, quality, conf)
                pbar.update(1)
        print(f"\n所有文件处理完成。结果保存到 {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO手部距离计算器（自动截图）')
    parser.add_argument('--model', type=str, default=r'./weights/ultralytics/hand_yolov8n.pt', 
                        help='手部检测模型权重路径')
    parser.add_argument('--source', type=str, default=r"E:\Download\travidebla.mp4", 
                        help='源目录、图片路径或视频文件路径')
    parser.add_argument('--conf', type=float, default=0.6, help='置信度阈值')
    parser.add_argument('--save-dir', type=str, default=None, 
                        help='输出目录（未指定时自动递增）')
    parser.add_argument('--no-save-txt', action='store_true', 
                        help='不保存距离结果为txt文件')
    parser.add_argument('--distance-threshold', type=int, default=1500, 
                        help='触发截图的距离阈值（像素）')
    parser.add_argument('--stable-duration', type=float, default=1, 
                        help='触发截图所需的稳定时长（秒）')
    parser.add_argument('--crop-ratio', type=float, default=0.2, 
                        help='图像两边向中央裁剪的总比例（默认0，即不裁剪）。例如0.3表示左右各裁剪15%，总共裁剪30%')
    parser.add_argument('--quality', type=int, default=80, 
                        help='图像/视频压缩质量（1-100，默认80，值越高质量越好文件越大）')
    
    args = parser.parse_args()
    save_txt = not args.no_save_txt
    
    print(f"使用模型: {args.model}")
    print(f"置信度阈值: {args.conf}")
    print(f"距离阈值: {args.distance_threshold} px")
    print(f"稳定时长: {args.stable_duration} 秒")
    print(f"裁剪比例: {args.crop_ratio}")
    print(f"压缩质量: {args.quality}")
    
    run_hand_distance(args.model, args.source, args.conf, args.save_dir, save_txt,
                      args.distance_threshold, args.stable_duration, args.crop_ratio, args.quality)