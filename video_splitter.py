import os
import sys
from pathlib import Path

from utils import split_video, is_video_file, get_files_by_extension


def process_single_video(video_path, output_dir, segment_duration, codec):
    """
    处理单个视频的切分

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        segment_duration: 每段时长（秒）
        codec: 视频编码格式

    Returns:
        int: 生成的片段数量
    """
    video_name = Path(video_path).stem.strip().replace(" ", "_")
    seg_dir = os.path.join(output_dir, video_name)
    segments = split_video(video_path, seg_dir, segment_duration, codec)
    return len(segments)


def process_directory(source_dir, output_dir, segment_duration, codec):
    """
    批量处理目录下所有视频文件

    Args:
        source_dir: 源目录路径
        output_dir: 输出根目录
        segment_duration: 每段时长（秒）
        codec: 视频编码格式
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')
    files = get_files_by_extension(source_dir, video_extensions)

    if not files:
        print(f"[WARN] 目录下未找到视频文件: {source_dir}")
        return

    total_videos = len(files)
    total_segments = 0

    print(f"[INFO] 发现 {total_videos} 个视频文件，开始批量切分...\n")

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(source_dir, filename)
        print(f"\n{'='*50}")
        print(f"[{idx}/{total_videos}] 处理: {filename}")
        print(f"{'='*50}")

        seg_count = process_single_video(filepath, output_dir, segment_duration, codec)
        total_segments += seg_count

    print(f"\n{'='*50}")
    print(f"[ALL DONE] 共处理 {total_videos} 个视频，生成 {total_segments} 个片段")
    print(f"[OUTPUT] 输出目录: {output_dir}")


def main():
    # ======================== 配置区域 ========================
    INPUT_PATH = r"D:\ComfyUI-aki-v3\ComfyUI\input\Rapi .webm"          # 输入视频文件路径或目录路径
    OUTPUT_DIR = r"E:\output_video"   # 输出目录路径
    SEGMENT_DURATION = 5           # 每段时长（秒）
    CODEC = "mp4v"                    # 视频编码格式（mp4v / avc1 / XVID）
    # ============================================================

    input_path = os.path.abspath(INPUT_PATH)
    output_dir = os.path.abspath(OUTPUT_DIR)
    segment_duration = SEGMENT_DURATION
    codec = CODEC

    if not os.path.exists(input_path):
        print(f"[ERROR] 输入路径不存在: {input_path}")
        sys.exit(1)

    if segment_duration <= 0:
        print("[ERROR] 切分时长必须大于 0")
        sys.exit(1)

    print(f"[CONFIG] 输入: {input_path}")
    print(f"[CONFIG] 输出: {output_dir}")
    print(f"[CONFIG] 每段时长: {segment_duration}s")
    print(f"[CONFIG] 编码: {codec}")

    if os.path.isfile(input_path):
        if is_video_file(input_path):
            seg_count = process_single_video(input_path, output_dir, segment_duration, codec)
            print(f"\n[DONE] 切分完成，共生成 {seg_count} 个片段")
        else:
            print(f"[ERROR] 非视频文件: {input_path}")
            sys.exit(1)

    elif os.path.isdir(input_path):
        process_directory(input_path, output_dir, segment_duration, codec)

    else:
        print(f"[ERROR] 无效的输入路径类型: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()