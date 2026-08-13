import os
import sys
from pathlib import Path

from utils import merge_videos, is_video_file, check_ffmpeg


def validate_video_paths(video_paths):
    """
    校验视频路径列表是否有效

    Args:
        video_paths: 视频文件路径列表

    Returns:
        list: 有效的视频文件绝对路径列表
    """
    if len(video_paths) < 2:
        print("[ERROR] 至少需要2个视频文件进行拼接")
        sys.exit(1)

    valid_paths = []
    for vp in video_paths:
        abs_path = os.path.abspath(vp)
        if not os.path.exists(abs_path):
            print(f"[ERROR] 视频文件不存在: {abs_path}")
            sys.exit(1)
        if not is_video_file(abs_path):
            print(f"[ERROR] 非视频文件: {abs_path}")
            sys.exit(1)
        valid_paths.append(abs_path)

    return valid_paths


def main():
    # ======================== 配置区域 ========================
    VIDEO_PATHS = [
        r"D:\ComfyUI-aki-v3\ComfyUI\output\Video\0813\cut_00001_.mp4",
        r"D:\ComfyUI-aki-v3\ComfyUI\output\Video\MiniMax_H3_00002_.mp4",
        r"D:\ComfyUI-aki-v3\ComfyUI\output\Video\MiniMax_H3_00001_.mp4"
    ]
    OUTPUT_PATH = r"D:\ComfyUI-aki-v3\ComfyUI\output\Video\merged_output.mp4"
    DIRECTION = "horizontal"  # 横向拼接或纵向拼接 (horizontal/vertical)
    CODEC = "mp4v"
    TARGET_FPS = None  # None=使用视频1的帧率；指定数值则强制统一帧率，如 30
    AUDIO_SOURCE = 0   # None=无音频；数字=指定视频的音频(0=视频1,1=视频2)；也可填音频文件路径

    # ============================================================

    # 参数校验
    if DIRECTION not in ('horizontal', 'vertical'):
        print("[ERROR] DIRECTION 必须为 'horizontal' 或 'vertical'")
        sys.exit(1)

    # 校验视频路径
    video_paths = validate_video_paths(VIDEO_PATHS)
    output_path = os.path.abspath(OUTPUT_PATH)

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 打印配置信息
    print(f"[CONFIG] 拼接方向: {'横向' if DIRECTION == 'horizontal' else '纵向'}")
    print(f"[CONFIG] 视频数量: {len(video_paths)}")
    print(f"[CONFIG] 输出路径: {output_path}")
    print(f"[CONFIG] 编码格式: {CODEC}")
    print(f"[CONFIG] 目标帧率: {TARGET_FPS or '自动（使用第一个视频帧率）'}")

    # 音频配置信息
    if AUDIO_SOURCE is None:
        print(f"[CONFIG] 音频来源: 无音频")
    elif isinstance(AUDIO_SOURCE, int):
        if 0 <= AUDIO_SOURCE < len(video_paths):
            print(f"[CONFIG] 音频来源: 视频{AUDIO_SOURCE + 1} ({os.path.basename(video_paths[AUDIO_SOURCE])})")
            if not check_ffmpeg():
                print("[WARN] 未检测到 ffmpeg，音频合成需要 ffmpeg")
                print("[WARN] 下载地址: https://ffmpeg.org/download.html")
                print("[WARN] 或将 ffmpeg 添加到系统 PATH 环境变量")
        else:
            print(f"[ERROR] AUDIO_SOURCE 索引超出范围: {AUDIO_SOURCE}，共 {len(video_paths)} 个视频")
            sys.exit(1)
    elif isinstance(AUDIO_SOURCE, str):
        audio_abs = os.path.abspath(AUDIO_SOURCE)
        if os.path.exists(audio_abs):
            print(f"[CONFIG] 音频来源: {os.path.basename(audio_abs)}")
            if not check_ffmpeg():
                print("[WARN] 未检测到 ffmpeg，音频合成需要 ffmpeg")
                print("[WARN] 下载地址: https://ffmpeg.org/download.html")
        else:
            print(f"[ERROR] 音频来源文件不存在: {audio_abs}")
            sys.exit(1)
    else:
        print(f"[ERROR] AUDIO_SOURCE 类型无效: {type(AUDIO_SOURCE)}")
        print(f"  支持类型: None (无音频), int (视频索引), str (文件路径)")
        sys.exit(1)

    # 执行拼接
    success = merge_videos(
        video_paths=video_paths,
        output_path=output_path,
        direction=DIRECTION,
        codec=CODEC,
        target_fps=TARGET_FPS,
        audio_source=AUDIO_SOURCE,
    )

    if success:
        print(f"\n[ALL DONE] 视频拼接成功！")
    else:
        print(f"\n[FAILED] 视频拼接失败，请检查输入视频是否有效")
        sys.exit(1)


if __name__ == '__main__':
    main()