"""集中管理各 CLI 脚本的本机路径默认值。

这些值原本散落在各 argparse 的 default 里，导致本机路径泄露且难以统一调整。
现统一收口到本文件：需改动路径时只改这里，脚本通过 utils 导入使用。
"""

# detect_comfyui / colorize 共用：画册输入目录
DEFAULT_ALBUM_SOURCE = r'E:\储藏室\画册\ss - 副本\[Cuvie] Bitter Addiction [DL版][机翻]'

# hand_distance：示例视频
DEFAULT_VIDEO_SOURCE = r'E:\Download'

# canny_cut：单图输入
DEFAULT_CANNY_INPUT = r'C:\Users\Administrator\Desktop\1.png'

# copy_detected_images：来源与目标目录
DEFAULT_DETECT_SOURCE_DIR = r'E:\Share\剩下'
DEFAULT_DETECT_TARGET_DIR = r'E:\Share\剩下\detected'

# crop_book 系列：测试画册目录
DEFAULT_BOOK_INPUT = r'E:\储藏室\画册\扫描\testbook'

# replace_font_paths：待处理 json 目录
DEFAULT_FONT_JSON_DIR = r"E:\Share\1\original_images\manga_translator_work\json"

# hand_distance：视频推理输入的最长边像素（预缩放，0/None 表示不缩放）
DEFAULT_INFER_MAX_EDGE = None

# hand_distance：模型前向尺寸（imgsz，直接决定网络 FLOPs；None 用模型默认 640）
DEFAULT_INFER_IMGSZ = None

# hand_distance：静止判定阈值（0-255 帧间平均绝对差；帧间运动低于此值视为静止）。
# >0 时：静止帧复用上次推理并才累计"稳定时长"，运动帧重置计数，保证截图帧清晰静止；0 关闭。
# 目标机实测：m=20 时截图仍清晰、速度最快(约47帧/s，实时富余)。注意该值已高到几乎把所有帧
# 判为静止（该片逐帧MAD p99≈24.6），静止门槛近乎失效、近似“只按距离累计+最大复用”，
# 若今后遇到严格需“静止才截”的片子应下调(如4~6)。
DEFAULT_MOTION_THRESHOLD = 4