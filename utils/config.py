"""集中管理各 CLI 脚本的本机路径默认值与通用调参。

这些值原本散落在各 argparse 的 default 或各脚本内，导致本机路径泄露、重复定义且难以统一调整。
现统一收口到本文件：需改动时只改这里，脚本通过 utils 导入使用。

分组约定：路径类常量与调参类常量分开，按所属模块聚合。
"""

# ============================================================
# 一、审查检测 / 上色（workflow_decensor / workflow_colorize）
# ============================================================
# 审查检测模型权重（默认）
DEFAULT_CENSOR_MODEL = 'weights/censor_detect_m_0831.pt'

# 审查检测输入目录（workflow_decensor 专用，图片目录 / zip 均可）
DEFAULT_CENSOR_SOURCE = r"E:\Share"

# 上色输入目录（workflow_colorize 专用）
DEFAULT_COLORIZE_SOURCE = r"E:\Share"

# zip 模式下去码输出 zip 的目标目录（文件名保持「源文件名[去码].zip」）
DEFAULT_DECENSOR_OUT_DIR = r"E:\Share\manga_output"

# ============================================================
# 二、hand_distance：手部距离 → 截屏
# ============================================================
# 手部检测模型权重（默认）
DEFAULT_HAND_MODEL = r'weights/cbook-hand.pt'

# 手部距离结果父目录（每段视频一个 expN，expN 内 <视频名>/ 为去重 jpg）
DEFAULT_HAND_SAVE_DIR = r'runs\hand_distance'

# 去重截图的通用命中根目录（extract_hand_dedup 手动筛选用）
DEFAULT_TARGET_ROOT = r'runs'

# 示例视频输入（hand_distance 系列专用，各脚本独立源）
DEFAULT_VIDEO_SOURCE = r"E:\Share"

# --- 推理性能调参 ---
# 视频推理输入的最长边像素（预缩放，0/None 表示不缩放）
DEFAULT_INFER_MAX_EDGE = None
# 模型前向尺寸（imgsz，直接决定网络 FLOPs；None 用模型默认 640）
DEFAULT_INFER_IMGSZ = None
# 静止判定阈值（0-255 帧间平均绝对差；帧间运动低于此值视为静止）。
# >0 时：静止帧复用上次推理并才累计"稳定时长"，运动帧重置计数，保证截图帧清晰静止；0 关闭。
# 统一取 4：在"画面静止才截"与推理复用速度之间平衡；素材对静止门槛要求更高可下调(如2~3)。
DEFAULT_MOTION_THRESHOLD = 4

# ============================================================
# 三、auto_label : Florence-2
# ============================================================
# Florence-2 模型本地仓库目录（含 config.json + model.safetensors，
# 从 HuggingFace 下载 microsoft/Florence-2-base 或 Florence-2-large 后放到这里）
DEFAULT_FLORENCE2_MODEL = r'weights\Florence-2-base'

# OVD 输出中过滤"退化点框"的最小归一化边长。
# 模型对无目标背景偶尔吐出接近图像原点(0,0)、边长<0.1% 的无效小框，score 恒为 1
# 无法用置信度过滤，须按尺寸剔除；0.005 = 图像边长 0.5%，真实目标远大于此。
DEFAULT_MIN_BOX_SIZE = 0.005

# 类别定义 yaml（顺序决定 class_id，从 0 开始）
DEFAULT_FLORENCE2_CLASSES = 'apps/classes.yaml'

# 默认置信度阈值（Florence-2 开放词汇检测分数通常偏低，0.35 较合理）
DEFAULT_FLORENCE2_CONF = 0.35

# 输入目录
DEFAULT_FLORENCE2_INPUT_DIR = r"E:\Share"

# 结果父目录，每个输入生成独立 expN 子目录
DEFAULT_FLORENCE2_SAVE_DIR = r'runs\florence_labels'

# ============================================================
# 三-b、auto_label_yolo：YOLO 自动标注
# ============================================================
# YOLO 自动标注模型权重（默认复用 hand_distance 的手部模型，可按需替换）
DEFAULT_YOLO_LABEL_MODEL = r'weights\censor_detect_v1.0_s_0725.pt'

# 类别过滤 yaml（可选，默认 None 直接使用模型自带类别；
# 提供后仅标注 names 中的类，并按 yaml 顺序重排 class_id）
DEFAULT_YOLO_LABEL_CLASSES = None

# 默认置信度阈值
DEFAULT_YOLO_CONF = 0.4

# 输入目录（复用 Florence-2 的输入源）
DEFAULT_YOLO_INPUT_DIR = r"E:\Share"

# 结果父目录，每个输入生成独立 expN 子目录
DEFAULT_YOLO_SAVE_DIR = r'runs\yolo_labels'

# ============================================================
# 四、scripts 一次性脚本默认路径
# ============================================================
# canny_cut：单图输入
DEFAULT_CANNY_INPUT = r'C:\Users\Administrator\Desktop\1.png'

# copy_detected_images：来源与目标目录
DEFAULT_DETECT_SOURCE_DIR = r'E:\Share\剩下'
DEFAULT_DETECT_TARGET_DIR = r'E:\Share\剩下\detected'

# crop_book 系列：测试画册目录
DEFAULT_BOOK_INPUT = r'E:\储藏室\画册\扫描\testbook'

# replace_font_paths：待处理 json 目录
DEFAULT_FONT_JSON_DIR = r"E:\Share\1\original_images\manga_translator_work\json"

# ============================================================
# 五、公共文件系统常量
# ============================================================
# 帧去重图片输出的统一扩展名（相似度 / 去重逻辑通用）
DEFAULT_EXT = '.jpg'