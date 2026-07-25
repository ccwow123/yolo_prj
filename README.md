# YOLO手部距离计算器

基于YOLOv8的手部距离计算工具，支持视频和图片处理，自动检测双手并计算距离，超过阈值时自动截图。

## 功能特性

- ✅ 手部检测：使用YOLOv8模型检测图像中的手部
- ✅ 距离计算：计算左右手部之间的像素距离
- ✅ 自动截图：当手部分离超过阈值并保持稳定时自动截图
- ✅ 视频处理：支持处理视频文件，输出带距离标注的视频
- ✅ 图像裁剪：支持按比例裁剪图像两边
- ✅ 图像压缩：支持JPEG压缩，可配置压缩质量
- ✅ 截图去重：使用ORB特征匹配去除相似截图
- ✅ 画册比例计算：根据截图时的双手距离计算画册所占比例
- ✅ ComfyUI集成：支持检测后自动调用ComfyUI处理

## 安装依赖

推荐使用 `requirements.txt` 安装以保证可复现环境：

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install ultralytics opencv-python numpy tqdm Pillow
```

## 使用方法

### 1. detect.py - YOLO检测推理

通用YOLO检测脚本，支持图像和视频检测，输出检测框和汇总JSON。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/censor_detect_v1.0_s_0725.pt` | 模型权重文件路径 |
| `--source` | string | `imgs` | 源目录、图像路径或视频文件路径 |
| `--conf` | float | 0.6 | 置信度阈值 |
| `--save-dir` | string | `runs/detections` | 结果保存目录 |
| `--save-json` | flag | False | 保存单个检测json文件（summary.json始终保存） |
| `--save-annotated` | flag | True | 保存带检测框的图片 |
| `--no-annotated` | flag | False | 保存原图，不带检测框 |
| `--device` | string | `cuda` | 推理设备（cpu 或 cuda） |

#### 使用示例

```bash
# 检测目录中的所有图像
python detect.py --source imgs/

# 检测单个图像
python detect.py --source input.jpg

# 检测视频文件
python detect.py --source input.mp4

# 保存原图（不带检测框）
python detect.py --source imgs/ --no-annotated

# 使用CPU推理
python detect.py --source imgs/ --device cpu
```

### 2. hand_distance.py - 手部距离计算器

检测手部并计算双手距离，超过阈值自动截图。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `./weights/ultralytics/hand_yolov8n.pt` | 手部检测模型权重路径 |
| `--source` | string | - | 源目录、图片路径或视频文件路径 |
| `--conf` | float | 0.6 | 置信度阈值 |
| `--save-dir` | string | `runs/hand_distance` | 输出目录 |
| `--no-save-txt` | flag | False | 不保存距离结果为txt文件 |
| `--distance-threshold` | int | 1500 | 触发截图的距离阈值（像素） |
| `--stable-duration` | float | 2.0 | 触发截图所需的稳定时长（秒） |
| `--crop-ratio` | float | 0.2 | 图像两边向中央裁剪的总比例 |
| `--quality` | int | 100 | 图像压缩质量（1-100） |

#### 使用示例

```bash
# 处理视频文件
python hand_distance.py --source input.mp4

# 处理图片文件
python hand_distance.py --source input.jpg

# 处理目录中的所有文件
python hand_distance.py --source ./images/

# 设置自定义距离阈值
python hand_distance.py --source input.mp4 --distance-threshold 1500

# 设置裁剪比例（左右各裁剪15%，总共裁剪30%）
python hand_distance.py --source input.mp4 --crop-ratio 0.3

# 设置压缩质量
python hand_distance.py --source input.mp4 --quality 90
```

### 3. detect_comfyui.py - 检测 + ComfyUI 处理流程

先检测图像，有检测目标的图像通过ComfyUI处理，无检测目标的图像直接复制。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/censor_detect_v1.0_s_0725.pt` | 模型权重文件路径 |
| `--source` | string | `imgs` | 检测输入目录 |
| `--conf` | float | 0.6 | 检测置信度阈值 |
| `--detect-save-dir` | string | `runs/detections` | 检测结果保存目录 |
| `--save-json` | flag | False | 保存单个检测json文件 |
| `--workflow` | string | `workflows/f2k-漫画去码-py.json` | ComfyUI工作流JSON路径 |
| `--comfyui-save-dir` | string | `runs/comfyui_output` | ComfyUI结果保存目录 |
| `--comfyui-server` | string | `http://127.0.0.1:8188` | ComfyUI服务器地址 |

#### 使用示例

```bash
# 运行完整检测+ComfyUI处理流程
python detect_comfyui.py --source imgs/

# 使用自定义工作流
python detect_comfyui.py --source imgs/ --workflow workflows/anima漫画上色-py.json

# 指定ComfyUI服务器地址
python detect_comfyui.py --source imgs/ --comfyui-server http://192.168.1.100:8188
```

#### 处理流程

1. **步骤1**：调用YOLO检测图像
2. **步骤2**：读取汇总JSON，判断哪些图片有检测结果
3. **步骤3**：无检测目标的图像直接复制到输出目录
4. **步骤4**：有检测目标的图像通过ComfyUI处理

### 4. train.py - 模型训练

使用Ultralytics平台训练YOLO模型。

#### 使用说明

```bash
# 运行训练脚本
python train.py
```

#### 训练配置说明

- **数据集**：使用Ultralytics平台上的 `cbook` 数据集
- **模型**：YOLO26n
- **训练轮数**：100 epochs
- **项目名称**：`lemon/my-project`
- **实验名称**：`v2i-3`

> **注意**：需要在Ultralytics平台注册账号并获取API Key，替换脚本中的 `ULTRALYTICS_API_KEY`。

## 输出结果

### detect.py 输出

```
runs/detections/exp*/
├── <image_name>.jpg          # 检测后的图像（带检测框）
├── <image_name>.json         # 单个检测结果（--save-json时生成）
├── det_<video_name>.mp4      # 处理后的视频（仅视频输入）
└── summary.json              # 汇总JSON文件
```

### hand_distance.py 输出

```
runs/hand_distance/exp*/
├── hand_distance_<video_name>.mp4  # 处理后的视频（仅视频输入）
├── screenshots/                    # 截图目录
│   ├── screenshot_000000.png
│   └── ...
├── distance_summary.txt            # 距离统计摘要
└── frame_distance_log.txt          # 每帧距离日志
```

### detect_comfyui.py 输出

```
runs/detections/exp*/           # 检测结果目录
runs/comfyui_output/            # ComfyUI处理结果目录
```

## 裁剪比例说明

`--crop-ratio` 参数表示图像两边向中央裁剪的**总比例**：

- `crop_ratio=0`：不裁剪
- `crop_ratio=0.3`：左右各裁剪15%，总共裁剪30%，保留中间70%
- `crop_ratio=0.4`：左右各裁剪20%，总共裁剪40%，保留中间60%

## 模型权重

| 模型名称 | 路径 | 用途 |
|---------|------|------|
| 手部检测模型 | `./weights/ultralytics/hand_yolov8n.pt` | 检测手部 |
| 审查检测模型 | `./weights/censor_detect_v1.0_s_0725.pt` | 检测审查目标 |
| YOLO26n | `./weights/ultralytics/yolo26/yolo26n/yolo26n.pt` | 通用检测 |
| SAM模型 | `./weights/ultralytics/sam_b.pt` | 分割模型 |

## 原理说明

1. **手部检测**：使用YOLOv8模型检测图像中的手部目标
2. **距离计算**：提取左右手部的中心点坐标，计算欧氏距离
3. **稳定检测**：当距离超过阈值时开始计数，连续稳定指定时长后触发截图
4. **截图去重**：使用ORB特征匹配算法去除相似截图
5. **画册比例**：根据截图时的双手距离平均值除以图像宽度计算画册所占比例

## 许可证

MIT License