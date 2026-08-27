"""集中管理各 CLI 脚本的本机路径默认值。

这些值原本散落在各 argparse 的 default 里，导致本机路径泄露且难以统一调整。
现统一收口到本文件：需改动路径时只改这里，脚本通过 utils 导入使用。
"""

# detect_comfyui / colorize 共用：画册输入目录
DEFAULT_ALBUM_SOURCE = r'E:\储藏室\画册\ss - 副本\[Cuvie] Bitter Addiction [DL版][机翻]'

# hand_distance：示例视频
DEFAULT_VIDEO_SOURCE = r"D:\cute aggression ういり画集 日版.mp4"

# canny_cut：单图输入
DEFAULT_CANNY_INPUT = r'C:\Users\Administrator\Desktop\1.png'

# copy_detected_images：来源与目标目录
DEFAULT_DETECT_SOURCE_DIR = r'E:\Share\剩下'
DEFAULT_DETECT_TARGET_DIR = r'E:\Share\剩下\detected'

# crop_book 系列：测试画册目录
DEFAULT_BOOK_INPUT = r'E:\储藏室\画册\扫描\testbook'

# replace_font_paths：待处理 json 目录
DEFAULT_FONT_JSON_DIR = r"E:\Share\1\original_images\manga_translator_work\json"