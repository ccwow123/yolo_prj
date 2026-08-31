import os
import json
import subprocess
import tempfile
import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# logger
logger = logging.getLogger(__name__)


class _TqdmLogHandler(logging.StreamHandler):
    """把日志经 tqdm.write 输出，避免刷新进度条时被日志行"顶开"。

    tqdm 进度条与控制台日志都占用 stderr，直接用 StreamHandler 打印会在
    每次回合把进度条挤换行。tqdm.write 打印前会临时收起进度条、写完再重绘，
    因此日志与进度条能和谐共存；无进度条时退化为普通单行打印，行为不变。
    """

    def emit(self, record):
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def configure_logging(level=logging.INFO, fmt='[%(asctime)s] %(levelname)s: %(message)s'):
    """统一日志初始化，各 CLI 入口调用以复用相同格式/级别。

    日志经 _TqdmLogHandler 输出：与 tqdm 进度条共存，不被顶开。
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = _TqdmLogHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt))
    root.addHandler(handler)


def get_next_exp_dir(base_dir='runs/exp'):
    """
    获取下一个实验目录路径（自动递增）

    Args:
        base_dir: 基础目录路径

    Returns:
        下一个实验目录的完整路径，格式为 base_dir/exp1, exp2...（从 exp1 开始）
    """
    os.makedirs(base_dir, exist_ok=True)
    exp_num = 1
    while True:
        exp_dir = os.path.join(base_dir, f"exp{exp_num}")
        if not os.path.exists(exp_dir):
            return exp_dir
        exp_num += 1


def imread_unicode(path):
    """兼容含非 ASCII 字符路径的读图 (cv2.imread 在 Windows 上对这类路径会静默返回 None)。
    文件不存在/无法解码时返回 None。"""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def is_video_file(filepath):
    """
    判断文件是否为视频文件
    
    Args:
        filepath: 文件路径
    
    Returns:
        True 如果是视频文件，否则 False
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm',
                        '.wmv', '.m4v', '.ts', '.mpeg', '.mpg', '.3gp', '.rmvb')
    return filepath.lower().endswith(video_extensions)


def is_image_file(filepath):
    """
    判断文件是否为图片文件
    
    Args:
        filepath: 文件路径
    
    Returns:
        True 如果是图片文件，否则 False
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                        '.webp', '.gif', '.heic', '.heif', '.ico', '.svg')
    return filepath.lower().endswith(image_extensions)


def is_grayscale_v2(image_path, saturation_threshold=10, variance_threshold=10, sample_size=64):
    """
    快速判断图像是否为灰度图（V2版，先缩放采样再判断，速度更快）

    Args:
        image_path: 图像文件路径
        saturation_threshold: HSV饱和度阈值（0-255），低于此值认为是灰度图
        variance_threshold: 通道差异阈值，低于此值认为三个通道差异很小
        sample_size: 采样尺寸，越小越快，默认64

    Returns:
        bool: True表示灰度图（需要上色），False表示彩色图
    """
    import numpy as np

    try:
        img = imread_unicode(image_path)
        if img is None:
            return False

        if len(img.shape) == 2:
            return True

        if img.shape[2] != 3:
            return False

        height, width = img.shape[:2]
        if min(height, width) <= 1:
            return True

        # 只对缩小后的采样图做判断，显著提升速度
        scale = min(1.0, sample_size / max(height, width))
        sample_width = max(1, int(width * scale))
        sample_height = max(1, int(height * scale))
        sample = cv2.resize(img, (sample_width, sample_height), interpolation=cv2.INTER_AREA)

        # 基于通道间差异判断：灰度图各通道几乎一致
        channel_diff = np.max(sample, axis=2) - np.min(sample, axis=2)
        mean_diff = float(np.mean(channel_diff))
        if mean_diff < variance_threshold:
            return True

        # 仅在小样本上做一次HSV饱和度判断，仍然保持较高准确率
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        if mean_saturation < saturation_threshold:
            return True

        return False

    except Exception as e:
        print(f"判断图像类型时出错: {e}")
        return False


def is_grayscale(image_path, saturation_threshold=10, variance_threshold=10):
    """
    兼容旧接口的灰度图检测入口，默认使用 V2 版本。
    """
    return is_grayscale_v2(
        image_path,
        saturation_threshold=saturation_threshold,
        variance_threshold=variance_threshold,
    )


def get_files_by_extension(directory, extensions):
    """
    获取目录中指定扩展名的所有文件
    
    Args:
        directory: 目录路径
        extensions: 扩展名元组，如 ('.jpg', '.png')
    
    Returns:
        符合条件的文件路径列表（排序后）
    """
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and f.lower().endswith(extensions)
    ]
    return sorted(files)


def _build_detections(result):
    """从 YOLO 结果提取检测列表，统一构建逻辑，避免在多处重复。"""
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xywh = box.xywh[0].tolist()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": cls,
            "class_name": result.names[cls] if hasattr(result, 'names') else "unknown",
            "confidence": conf,
            "bbox_xywh": xywh,
            "bbox_xyxy": [x1, y1, x2, y2]
        })
    return detections


def save_detection_results(result, save_dir, filename, save_json=False, save_annotated=True):
    """
    保存检测结果到文件
    
    Args:
        result: YOLO检测结果对象
        save_dir: 保存目录
        filename: 文件名
        save_json: 是否保存单个json标注文件（汇总json始终保存）
        save_annotated: 是否保存带检测框的图片，False则保存原图
    
    Returns:
        dict: 检测结果数据（用于汇总），如果没有检测结果返回None
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图片结果
    if save_annotated:
        # YOLO自带的保存方法，内部按 BGR 语义处理，颜色正确
        result.save(os.path.join(save_dir, filename))
    else:
        # 保存原始图像：result.orig_img 实际为 BGR（由 cv2 读入），
        # 需先转 RGB 再交给 PIL，否则会 B/R 通道交换导致变色
        from PIL import Image
        img = Image.fromarray(result.orig_img[:, :, ::-1])
        img.save(os.path.join(save_dir, filename))
    
    # 准备检测结果数据
    data = {
        "image_filename": filename,
        "detection_count": 0,
        "detections": []
    }
    
    detections = _build_detections(result)
    data["detection_count"] = len(detections)
    data["detections"] = detections
    
    # 保存单个json标注（仅当save_json=True时）
    if save_json and detections:
        json_path = os.path.join(save_dir, os.path.splitext(filename)[0] + '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data


def validate_parameters(params):
    """
    校验参数是否在有效范围内
    
    Args:
        params: 参数字典，包含需要校验的参数
    
    Returns:
        错误信息列表（如果没有错误则为空列表）
    """
    errors = []
    
    if 'crop_ratio' in params and not (0 <= params['crop_ratio'] < 1):
        errors.append(f"裁剪比例 crop_ratio 必须在 [0, 1) 范围内，当前值: {params['crop_ratio']}")
    
    if 'quality' in params and not (1 <= params['quality'] <= 100):
        errors.append(f"压缩质量 quality 必须在 [1, 100] 范围内，当前值: {params['quality']}")
    
    if 'distance_threshold' in params and params['distance_threshold'] < 0:
        errors.append(f"距离阈值 distance_threshold 必须大于等于0，当前值: {params['distance_threshold']}")
    
    if 'stable_duration' in params and params['stable_duration'] <= 0:
        errors.append(f"稳定时长 stable_duration 必须大于0，当前值: {params['stable_duration']}")
    
    if 'conf' in params and not (0 <= params['conf'] <= 1):
        errors.append(f"置信度 conf 必须在 [0, 1] 范围内，当前值: {params['conf']}")
    
    return errors


def split_video(video_path, output_dir, segment_duration=30, codec='mp4v'):
    """
    将视频按指定时长切分为多个片段

    Args:
        video_path: 输入视频文件路径
        output_dir: 输出目录路径
        segment_duration: 每段时长（秒），默认30秒
        codec: 视频编码格式，默认 mp4v（兼容MP4）

    Returns:
        list: 生成的片段文件路径列表；若读取失败返回空列表
    """
    import math

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        print(f"[ERROR] 视频参数异常: fps={fps}, frames={total_frames}")
        cap.release()
        return []

    total_duration = total_frames / fps
    segment_frames = int(segment_duration * fps)
    num_segments = math.ceil(total_frames / segment_frames)

    base_name = os.path.splitext(os.path.basename(video_path))[0].strip().replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 视频信息: {os.path.basename(video_path)}")
    print(f"  分辨率: {width}x{height}, 帧率: {fps:.2f}fps")
    print(f"  总时长: {total_duration:.2f}s, 总帧数: {total_frames}")
    print(f"  切分段数: {num_segments}, 每段: {segment_duration}s ({segment_frames}帧)")
    print(f"  输出目录: {output_dir}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    generated_paths = []
    segment_idx = 0
    frame_count = 0
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % segment_frames == 0:
            if writer is not None:
                writer.release()
            segment_idx += 1
            seg_path = os.path.join(output_dir, f"{base_name}_part{segment_idx:03d}.mp4")
            writer = cv2.VideoWriter(seg_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"[ERROR] 无法创建输出文件: {seg_path}")
                break
            generated_paths.append(seg_path)
            print(f"[WRITE] 正在写入片段 {segment_idx}/{num_segments}: {os.path.basename(seg_path)}")

        if writer is not None:
            writer.write(frame)

        frame_count += 1

    if writer is not None:
        writer.release()
    cap.release()

    print(f"[DONE] 切分完成，共生成 {len(generated_paths)} 个片段")
    return generated_paths


def merge_videos(video_paths, output_path, direction='horizontal', codec='mp4v', target_fps=None, audio_source=None):
    """
    将多个视频拼接为一个视频（横向或纵向拼接）

    Args:
        video_paths: 视频文件路径列表，至少2个视频
        output_path: 输出视频文件路径
        direction: 拼接方向，'horizontal' 横向拼接，'vertical' 纵向拼接
        codec: 视频编码格式，默认 mp4v
        target_fps: 目标帧率，None 则使用第一个视频的帧率
        audio_source: 音频来源，支持以下格式：
            - None: 无音频（默认）
            - int: 使用第 N 个输入视频的音频（0-indexed）
            - str: 指定音频文件路径

    Returns:
        bool: 拼接成功返回 True，否则 False
    """
    import numpy as np

    if len(video_paths) < 2:
        print("[ERROR] 至少需要2个视频进行拼接")
        return False

    if direction not in ('horizontal', 'vertical'):
        print(f"[ERROR] 拼接方向无效: {direction}，可选 horizontal / vertical")
        return False

    # 打开所有视频
    caps = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {vp}")
            for c in caps:
                c.release()
            return False
        caps.append(cap)

    # 获取第一个视频的基础参数
    base_fps = caps[0].get(cv2.CAP_PROP_FPS)
    if base_fps <= 0:
        base_fps = 30.0

    fps = target_fps if target_fps else base_fps
    fps_ratio = fps / base_fps if base_fps > 0 else 1.0

    base_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    base_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算每个视频的目标尺寸（保持宽高比，对齐到目标分辨率）
    target_size_list = []
    for i, cap in enumerate(caps):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if direction == 'horizontal':
            # 横向拼接：高度统一，宽度按比例缩放
            target_h = base_height
            target_w = int(w * (target_h / h))
            target_size_list.append((target_w, target_h))
        else:
            # 纵向拼接：宽度统一，高度按比例缩放
            target_w = base_width
            target_h = int(h * (target_w / w))
            target_size_list.append((target_w, target_h))

    # 计算输出视频的总尺寸
    if direction == 'horizontal':
        total_width = sum(s[0] for s in target_size_list)
        total_height = max(s[1] for s in target_size_list)
    else:
        total_width = max(s[0] for s in target_size_list)
        total_height = sum(s[1] for s in target_size_list)

    # 获取所有视频的总帧数
    total_frames_list = []
    for i, cap in enumerate(caps):
        v_fps = cap.get(cv2.CAP_PROP_FPS)
        v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if v_fps > 0 and v_frames > 0:
            # 换算到目标帧率下的帧数
            adjusted_frames = int(v_frames * (fps / v_fps))
        else:
            adjusted_frames = 0
        total_frames_list.append(adjusted_frames)

    max_frames = max(total_frames_list)
    video_count = len(caps)

    print(f"[INFO] 视频拼接配置:")
    print(f"  拼接方向: {'横向' if direction == 'horizontal' else '纵向'}")
    print(f"  视频数量: {video_count}")
    print(f"  输出分辨率: {total_width}x{total_height}")
    print(f"  目标帧率: {fps:.2f}fps")
    print(f"  最长视频帧数: {max_frames}")
    for i, (vp, sz, frames) in enumerate(zip(video_paths, target_size_list, total_frames_list)):
        print(f"  视频{i+1}: {os.path.basename(vp)} -> {sz[0]}x{sz[1]}, {frames}帧")

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    if not out_writer.isOpened():
        print(f"[ERROR] 无法创建输出文件: {output_path}")
        for c in caps:
            c.release()
        return False

    # 逐帧读取并拼接
    frame_idx = 0
    progress_step = max(1, max_frames // 20)  # 每5%输出一次进度

    while frame_idx < max_frames:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()

            if not ret:
                # 视频已结束，用黑底填充
                tw, th = target_size_list[i]
                frame = np.zeros((th, tw, 3), dtype=np.uint8)
            else:
                # 缩放到目标尺寸
                tw, th = target_size_list[i]
                h, w = frame.shape[:2]
                if (w, h) != (tw, th):
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            frames.append(frame)

        # 拼接帧
        if direction == 'horizontal':
            merged = np.hstack(frames)
        else:
            merged = np.vstack(frames)

        out_writer.write(merged)
        frame_idx += 1

        if frame_idx % progress_step == 0 or frame_idx == max_frames:
            progress = frame_idx / max_frames * 100
            print(f"\r[PROGRESS] {frame_idx}/{max_frames} ({progress:.1f}%)", end='', flush=True)

    print(f"\n[DONE] 拼接完成，共写入 {frame_idx} 帧")
    print(f"[OUTPUT] 输出文件: {output_path}")

    # 释放资源
    out_writer.release()
    for cap in caps:
        cap.release()

    # 合成音频
    if audio_source is not None:
        success = merge_audio_with_video(output_path, audio_source, video_paths)
        if not success:
            print("[WARN] 音频合成失败，已输出无声视频")

    return True


def check_ffmpeg():
    """
    检查系统是否安装了 ffmpeg

    Returns:
        bool: 可用返回 True，否则 False
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def merge_audio_with_video(video_path, audio_source, video_paths):
    """
    使用 ffmpeg 将音频合成到视频中

    Args:
        video_path: 视频文件路径（无声视频）
        audio_source: 音频来源，支持以下格式：
            - None: 无音频（直接返回 True）
            - int: 使用第 N 个输入视频的音频（0-indexed）
            - str: 指定视频文件的音频路径
        video_paths: 原始输入视频路径列表（用于索引查找）

    Returns:
        bool: 合成成功返回 True
    """
    if audio_source is None:
        return True

    if not check_ffmpeg():
        print("[WARN] 未检测到 ffmpeg，无法合成音频。请安装 ffmpeg: https://ffmpeg.org/download.html")
        print("[WARN] 已输出无声视频")
        return False

    # 确定音频来源文件
    if isinstance(audio_source, int):
        if audio_source < 0 or audio_source >= len(video_paths):
            print(f"[ERROR] 音频索引超出范围: {audio_source}，共 {len(video_paths)} 个视频")
            return False
        source_file = video_paths[audio_source]
        source_label = f"视频{audio_source + 1}"
    elif isinstance(audio_source, str):
        source_file = os.path.abspath(audio_source)
        if not os.path.exists(source_file):
            print(f"[ERROR] 音频来源文件不存在: {source_file}")
            return False
        source_label = os.path.basename(source_file)
    else:
        print(f"[ERROR] 无效的音频来源类型: {type(audio_source)}")
        return False

    print(f"[INFO] 音频来源: {source_label}")

    # 创建临时文件存储最终输出
    temp_output = video_path + '.temp.mp4'

    # ffmpeg 命令：视频流直接复制，音频从指定文件提取
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', source_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-movflags', '+faststart',
        temp_output
    ]

    try:
        print("[INFO] 正在合成音频...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            # 用合成后的文件替换原文件
            os.replace(temp_output, video_path)
            print("[INFO] 音频合成完成")
            return True
        else:
            # 尝试没有音频流的情况
            print(f"[WARN] 音频合成失败: {result.stderr[:200]}")
            # 检查源视频是否有音频流
            check_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type',
                        '-of', 'csv=p=0', source_file]
            try:
                probe_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
                if 'audio' not in probe_result.stdout:
                    print("[WARN] 音频来源视频无音频轨道，输出无声视频")
                    return True
            except Exception:
                pass
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] 音频合成超时")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False
    except Exception as e:
        print(f"[ERROR] 音频合成异常: {e}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False


def extract_video_frames(video_path, output_dir, frame_types=None, image_format='jpg'):
    """
    提取单个视频的首帧和/或尾帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_types: 要提取的帧类型列表，可选 'first'、'last'，默认 ['first', 'last']
        image_format: 图片保存格式，默认 jpg

    Returns:
        list: 提取并保存的图片文件路径列表；失败返回空列表
    """
    if frame_types is None:
        frame_types = ['first', 'last']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[ERROR] 视频帧数异常: {video_path}")
        cap.release()
        return []

    video_name = os.path.splitext(os.path.basename(video_path))[0].strip().replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    # 提取首帧
    if 'first' in frame_types:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"{video_name}_first.{image_format}")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            print(f"  [首帧] {os.path.basename(out_path)}")

    # 提取尾帧（往回退2帧避免读取失败）
    if 'last' in frame_types:
        last_pos = max(0, total_frames - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_pos)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"{video_name}_last.{image_format}")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            print(f"  [尾帧] {os.path.basename(out_path)}")

    cap.release()
    return saved_paths


def load_source_list(list_file):
    """
    从txt文件读取源路径列表，每行一个路径。

    Args:
        list_file: txt文件路径

    Returns:
        list[str]: 路径列表（已跳过空行和#注释行，并去除两端引号）

    Raises:
        FileNotFoundError: 当txt文件不存在时
    """
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"txt文件不存在: {list_file}")

    paths = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().strip('"').strip("'")
            if not p or p.startswith('#'):
                continue
            paths.append(p)
    return paths


def collect_source_items(source, list_file=None, image_only=False, recursive=False):
    """
    收集待处理的媒体文件列表，统一处理单文件/目录/txt列表三种来源。

    Args:
        source: 目录路径或媒体文件路径（list_file 为空时必填）
        list_file: 可选，txt文件路径（每行一个路径，优先生效）
        image_only: True 时仅收集图片文件
        recursive: True 时递归遍历 source 目录下所有子文件夹

    Returns:
        items: [(path, is_video), ...] 列表
        error: 错误信息字符串，无错误时为 None
    """
    if not image_only:
        def match(p):
            return is_image_file(p) or is_video_file(p)
    else:
        def match(p):
            return is_image_file(p)

    def to_item(path):
        return (path, (not image_only) and is_video_file(path))

    if list_file:
        try:
            sources = load_source_list(list_file)
        except FileNotFoundError as e:
            return [], str(e)
        items = []
        for p in sources:
            if os.path.isfile(p):
                items.append(to_item(p))
            else:
                logger.warning(f"跳过（路径不存在或不是文件）: {p}")
        if not items:
            return [], f"txt文件中没有有效的媒体文件路径: {list_file}"
        return items, None

    if os.path.isfile(source):
        return [to_item(source)], None
    if os.path.isdir(source):
        if recursive:
            files = sorted(
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(source)
                for f in fn
                if match(f)
            )
        else:
            files = sorted(
                os.path.join(source, f) for f in os.listdir(source)
                if os.path.isfile(os.path.join(source, f)) and match(f)
            )
        if not files:
            return [], f"目录中没有找到媒体文件: {source}"
        return [to_item(fp) for fp in files], None
    return [], f"源文件/目录不存在: {source}"


def load_yolo_model(model_path):
    """
    加载YOLO模型并选择最佳设备

    Args:
        model_path: 模型权重文件路径

    Returns:
        model: 加载好的YOLO模型对象（已移动到合适设备）
        device_info: 设备信息字符串
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(model_path).to(device)
        model.eval()
        logger.info(f"模型加载成功: {model_path}")
        
        actual_device = model.device
        if actual_device.type == 'cuda':
            device_info = f"GPU (CUDA) - {torch.cuda.get_device_name(actual_device.index)}"
            logger.info(f"使用设备: {device_info}")
        else:
            device_info = "CPU"
            logger.info(f"使用设备: {device_info}")
            if torch.cuda.is_available():
                logger.warning("CUDA可用但模型仍在CPU上运行，尝试强制移动到GPU")
                model.cuda()
                if model.device.type == 'cuda':
                    device_info = f"GPU (CUDA) - {torch.cuda.get_device_name(model.device.index)}"
                    logger.info(f"成功移动到GPU: {device_info}")
            else:
                logger.info("CUDA不可用")
        
        return model, device_info
    
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return None, None