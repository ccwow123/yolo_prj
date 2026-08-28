"""集中管理各 CLI 脚本的本机路径默认值。

这些值原本散落在各 argparse 的 default 里，导致本机路径泄露且难以统一调整。
现统一收口到本文件：需改动路径时只改这里，脚本通过 utils 导入使用。
"""

# detect_comfyui / colorize 共用：画册输入目录
DEFAULT_ALBUM_SOURCE = r'E:\储藏室\画册\ss - 副本\[Cuvie] Bitter Addiction [DL版][机翻]'

# hand_distance：手部检测模型权重（默认）
DEFAULT_HAND_MODEL = r'weights/cbook-hand.pt'

# hand_distance：示例视频
DEFAULT_VIDEO_SOURCE = r'E:\Download\新建文件夹 (4)'

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
# 统一取 4：在"画面静止才截"与推理复用速度之间平衡；素材对静止门槛要求更高可下调(如2~3)。
DEFAULT_MOTION_THRESHOLD = 4

# auto_label：Florence-2 模型本地仓库目录（含 config.json + model.safetensors，
# 从 HuggingFace 下载 microsoft/Florence-2-base 或 Florence-2-large 后放到这里）
DEFAULT_FLORENCE2_MODEL = r'weights\Florence-2-base'
# auto_label：Florence-2 OVD 输出中过滤"退化点框"的最小归一化边长。
# 模型对无目标背景偶尔吐出接近图像原点(0,0)、边长<0.1% 的无效小框，score 恒为 1
# 无法用置信度过滤，须按尺寸剔除；0.005 = 图像边长 0.5%，真实目标远大于此。
DEFAULT_MIN_BOX_SIZE = 0.005

# auto_label：类别定义 yaml（顺序决定 class_id，从 0 开始）
DEFAULT_FLORENCE2_CLASSES = r'classes.yaml'

# auto_label：默认置信度阈值（Florence-2 开放词汇检测分数通常偏低，0.35 较合理）
DEFAULT_FLORENCE2_CONF = 0.35

# auto_label：输入目录
DEFAULT_FLORENCE2_INPUT_DIR = r"E:\Files\yolo_prj\runs\hand_distance\新建文件夹 (3)"

# auto_label：结果父目录，每个输入生成独立 expN 子目录
DEFAULT_FLORENCE2_SAVE_DIR = r'runs\florence_labels'