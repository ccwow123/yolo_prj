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
- ✅ 加速推理：模型前向尺寸缩放(`--imgsz`) + FP16混合精度(`--no-fp16`关闭)，大幅降低网络FLOPs
- ✅ 静止帧截取：帧间运动检测(`--motion-threshold`)，仅画面静止时才累计稳定时长并触发清晰截图
- ✅ 异步读帧：后台线程预取视频帧，把解码从推理主线程解耦
- ✅ 结果目录隔离：目录/txt批量输入时每个文件输出到独立的 `expN` 目录，互不覆盖

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

检测手部并计算双手距离，超过阈值且画面静止时自动截图。内置多项加速（异步读帧、静止帧复用推理、FP16、前向尺寸缩放）。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/cbook-hand.pt` | 手部检测模型权重路径 |
| `--source` | string | - | 源目录、图片路径或视频文件路径 |
| `--list-file` | string | None | 从txt批量导入源路径（每行一个，支持`#`注释和空行），优先于`--source` |
| `--conf` | float | 0.6 | 置信度阈值 |
| `--save-dir` | string | `runs/hand_distance` | 输出父目录，每个输入文件生成独立的 `expN` 子目录 |
| `--no-save-txt` | flag | False | 不保存距离结果为txt文件 |
| `--distance-threshold` | int | 1200 | 触发截图的距离阈值（像素） |
| `--stable-duration` | float | 1.0 | 触发截图所需的稳定时长（秒） |
| `--crop-ratio` | float | 0.2 | 图像两边向中央裁剪的总比例 |
| `--quality` | int | 100 | 图像压缩质量（1-100） |
| `--max-edge` | int | 1280 | 视频输入帧最长边像素（0 表示不预缩放），坐标映射回原分辨率 |
| `--imgsz` | int | 480 | 模型前向尺寸，直接决定网络FLOPs（0 用模型默认640） |
| `--motion-threshold` | float | 4.0 | 静止判定阈值（0-255帧间MAD，0关闭）；仅静止帧累计稳定时长并触发截图 |
| `--no-fp16` | flag | False | 禁用FP16混合精度推理（仅GPU生效） |
| `--no-video` | flag | False | 不生成输出视频（纯截图，跳过逐帧写入，最快） |
| `--no-annotate-video` | flag | False | 输出视频不叠加检测标注，写原帧（省每帧拷贝/绘制） |

#### 使用示例

```bash
# 处理视频文件
python hand_distance.py --source input.mp4

# 处理图片文件
python hand_distance.py --source input.jpg

# 处理目录中的所有文件（每个文件独立expN目录）
python hand_distance.py --source ./images/

# 从txt文件批量处理
python hand_distance.py --list-file list.txt

# 设置自定义距离阈值
python hand_distance.py --source input.mp4 --distance-threshold 1500

# 设置裁剪比例（左右各裁剪15%，总共裁剪30%）
python hand_distance.py --source input.mp4 --crop-ratio 0.3

# 开启静止帧截取（画面静止才截图，更清晰）
python hand_distance.py --source input.mp4 --motion-threshold 6

# 极致加速：纯截图，不写视频
python hand_distance.py --source input.mp4 --no-video --no-annotate-video
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

### 5. auto_label.py - Florence-2 自动标注

用 Florence-2 开放词汇检测把原始图片批量标注成 YOLO 数据集标签（坐标转归一化 `class_id cx cy w h`），一次共享提示框出全部类别。类别与提示词由 `classes.yaml` 定义。复用 `utils/core.py` 的收集/目录/logging 工具，无重复轮子。

#### 前置准备

1. 安装依赖（torchg 环境：`pip install transformers timm einops`）
2. 手动下载 Florence-2 safetensors 权重（`microsoft/Florence-2-base` 或 `-large`）到 `weights/florence2/` 目录
3. 编辑 `classes.yaml` 填入你的目标类别与提示词（顺序决定 class_id）

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/florence2/Florence-2-base` | Florence-2 模型本地目录 |
| `--source` | string | `imgs` | 待标注图片目录/单图/txt列表 |
| `--classes` | string | `classes.yaml` | 类别定义yaml |
| `--conf` | float | 0.35 | 置信度阈值 |
| `--save-dir` | string | `runs/florence_labels` | 结果父目录，每次生成独立 expN |
| `--device` | string | `cuda` | 推理设备 |
| `--no-fp16` | flag | False | 禁用FP16推理（仅GPU生效） |

#### 使用示例

```bash
python auto_label.py --source imgs --classes classes.yaml
python auto_label.py --source ./raw --model weights/florence2/Florence-2-base --conf 0.3
```

#### 输出结构

```
runs/florence_labels/exp1/
├── labels/<img>.txt      # YOLO 标签（class_id cx cy w h）
├── images/               # 拷贝的原图（与cbook参考数据集结构一致）
├── previews/<img>.png    # 带框预览图（人工抽检）
├── data.yaml             # 供 train.py 直接引用
├── summary.json          # 汇总（含每图检测明细）
└── classes.yaml 语义等同   # 类别映射见 classes 配置
```

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

每个输入文件一个独立的 `expN` 目录（从 `exp1` 开始递增）：

```
runs/hand_distance/exp1/            # 第一个输入文件的结果目录
├── hand_distance_<video_name>.mp4  # 处理后的视频（仅视频输入且未用 --no-video）
├── screenshots/                    # 原始截图目录
│   ├── screenshot_000000.png
│   └── ...
├── <video_name>/                   # 去重后精选截图
│   ├── screenshot_000123.jpg
│   └── ...
├── distance_summary.txt            # 距离统计摘要（--no-save-txt 关闭）
└── frame_distance_log.txt          # 每帧距离日志（--no-save-txt 关闭）
runs/hand_distance/exp2/            # 第二个输入文件的结果目录
...
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
| 手部检测模型 | `./weights/cbook-hand.pt` | 检测手部（hand_distance 默认） |
| 审查检测模型 | `./weights/censor_detect_v1.0_s_0725.pt` | 检测审查目标 |
| YOLO26n | `./weights/ultralytics/yolo26/yolo26n/yolo26n.pt` | 通用检测 |
| SAM模型 | `./weights/ultralytics/sam_b.pt` | 分割模型 |

## 原理说明

1. **手部检测**：使用YOLOv8模型检测图像中的手部目标
2. **距离计算**：提取左右手部的中心点坐标，计算欧氏距离
3. **稳定检测**：当距离超过阈值时开始计数，连续稳定指定时长后触发截图；配合 `--motion-threshold` 仅在画面静止时累计，保证截图帧清晰
4. **截图去重**：使用ORB特征匹配算法去除相似截图
5. **画册比例**：根据截图时的双手距离平均值除以图像宽度计算画册所占比例

### 加速原理

| 手段 | 说明 |
|------|------|
| 前向尺寸缩放（`--imgsz`） | 模型推理FLOPs近似随前向尺寸的平方下降，`imgsz=480` 相比默认640明显提速 |
| FP16混合精度（默认开启） | GPU上用半精度推理，速度接近翻倍；`--no-fp16` 可关闭 |
| 静止帧复用推理 | 画面静止帧跳过重复推理，直接复用上次检测框与标注图 |
| 异步读帧 | 后台线程预取解码帧，把I/O从推理主线程解耦 |
| 关视频输出（`--no-video`） | 跳过逐帧写入，纯截图最快路径 |

> **调参建议**：`motion_threshold` 越大判定越宽松（速度越快），但为保证"画面静止才截图"应配合实际素材微调；默认取 `4`，若素材对静止门槛要求不高可适当上调（如 8~20），追求更严静止则下调（如 2~3）。

## 许可证

MIT License