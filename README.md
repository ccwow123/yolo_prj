# YOLO 手部距离计算与漫画数据流水线

基于 YOLOv8 / Florence-2 的漫画与审查数据流水线：手部距离计算与去重截图、ComfyUI 上色/去码、自动标注、模型检测与训练。

## 项目结构

```
yolo_prj/
├── detect.py            # 业务入口：通用 YOLO 检测推理（保留根目录）
├── hand_distance.py     # 业务入口：手部距离计算 → 去重截图（保留根目录）
├── train.py             # 业务入口：模型训练（保留根目录）
│
├── apps/                # 可执行入口 / 流程脚本
│   ├── auto_label.py          # Florence-2 自动标注
│   ├── hand2label.py          # 两步工作流：手部去重截图 → 自动标注
│   ├── workflow_colorize.py   # 漫画上色（调用 ComfyUI）
│   └── workflow_decensor.py   # 漫画去码（检测 + ComfyUI，原 detect_comfyui.py）
│
├── workflows/           # ComfyUI 流程配置 JSON
│   ├── anima漫画上色-py.json
│   └── f2k-漫画去码-py.json
│
├── scripts/             # 一次性/辅助工具（裁剪、去重、数据集整理等）
├── utils/               # 可复用工具库（core / cv / hands / similarity / florence2 / comfyui / config）
├── tests/               # 单元回归测试（stdlib unittest）
├── classes.yaml         # 类别与提示词定义
├── weights/             # 模型权重
└── runs/                # 运行产物（检测、截图、标注结果，按 expN 隔离）
```

> 所有 `apps/`、`scripts/` 下的脚本建议**从项目根目录运行**（`--workflow` 等相对路径以项目根为基准）。

## 功能特性

- ✅ 手部检测：使用 YOLO 模型检测图像中的手部
- ✅ 距离计算：计算左右手部之间的像素距离
- ✅ 自动截图：当手部分离超过阈值并保持稳定时自动截图
- ✅ 视频处理：支持处理视频文件，输出带距离标注的视频
- ✅ 截图去重：使用 ORB 特征匹配去除相似截图
- ✅ 漫画上色 / 去码：通过 ComfyUI 工作流批量处理
- ✅ 自动标注：Florence-2 开放词汇检测生成 YOLO 标签 + 预览图
- ✅ 两步工作流：手部去重截图直接喂给自动标注
- ✅ 加速推理：前向尺寸缩放(`--imgsz`) + FP16 混合精度，降低网络 FLOPs
- ✅ 静止帧截取：帧间运动检测(`--motion-threshold`)，仅静止帧累计稳定时长并触发明晰截图
- ✅ 异步读帧：后台线程预取视频帧，把解码从推理主线程解耦
- ✅ 结果目录隔离：目录/txt 批量输入时每个文件输出到独立的 `expN` 目录，互不覆盖

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

### 1. detect.py - YOLO 检测推理

通用 YOLO 检测脚本，支持图像和视频检测，输出检测框和汇总 JSON。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/censor_detect_v1.0_s_0725.pt` | 模型权重文件路径 |
| `--source` | string | `imgs` | 源目录、图像路径或视频文件路径 |
| `--conf` | float | 0.6 | 置信度阈值 |
| `--save-dir` | string | `runs/detections` | 结果保存目录 |
| `--save-json` | flag | False | 保存单个检测 json 文件（summary.json 始终保存） |
| `--no-annotated` | flag | False | 保存原图，不带检测框 |
| `--device` | string | `cuda` | 推理设备（cpu 或 cuda） |

#### 使用示例

```bash
# 检测目录中的所有图像
python detect.py --source imgs/

# 检测单个图像 / 视频
python detect.py --source input.jpg
python detect.py --source input.mp4

# 保存原图（不带检测框）
python detect.py --source imgs/ --no-annotated

# 使用 CPU 推理
python detect.py --source imgs/ --device cpu
```

### 2. hand_distance.py - 手部距离计算器

检测手部并计算双手距离，超过阈值且画面静止时自动截图。内置多项加速（异步读帧、静止帧复用推理、FP16、前向尺寸缩放）。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/cbook-hand.pt` | 手部检测模型权重路径 |
| `--source` | string | - | 源目录、图片路径或视频文件路径 |
| `--list-file` | string | None | 从 txt 批量导入源路径（每行一个，支持 `#` 注释和空行），优先于 `--source` |
| `--conf` | float | 0.6 | 置信度阈值 |
| `--save-dir` | string | `runs/hand_distance` | 输出父目录，每个输入文件生成独立的 `expN` 子目录 |
| `--no-save-txt` | flag | False | 不保存距离结果为 txt 文件 |
| `--distance-threshold` | int | 1200 | 触发截图的距离阈值（像素） |
| `--stable-duration` | float | 1.0 | 触发截图所需的稳定时长（秒） |
| `--crop-ratio` | float | 0.2 | 图像两边向中央裁剪的总比例 |
| `--quality` | int | 100 | 图像压缩质量（1-100） |
| `--max-edge` | int | 1280 | 视频输入帧最长边像素（0 表示不预缩放），坐标映射回原分辨率 |
| `--imgsz` | int | 480 | 模型前向尺寸，直接决定网络 FLOPs（0 用模型默认 640） |
| `--motion-threshold` | float | 4.0 | 静止判定阈值（0-255 帧间 MAD，0 关闭）；仅静止帧累计稳定时长并触发截图 |
| `--no-fp16` | flag | False | 禁用 FP16 混合精度推理（仅 GPU 生效） |
| `--no-video` | flag | False | 不生成输出视频（纯截图，跳过逐帧写入，最快） |
| `--no-annotate-video` | flag | False | 输出视频不叠加检测标注，写原帧（省每帧拷贝/绘制） |

#### 使用示例

```bash
# 处理视频 / 图片
python hand_distance.py --source input.mp4
python hand_distance.py --source input.jpg

# 处理目录中的所有文件（每个文件独立 expN 目录）
python hand_distance.py --source ./images/

# 从 txt 文件批量处理
python hand_distance.py --list-file list.txt

# 自定义距离阈值 / 裁剪比例
python hand_distance.py --source input.mp4 --distance-threshold 1500
python hand_distance.py --source input.mp4 --crop-ratio 0.3

# 开启静止帧截取（画面静止才截图，更清晰）
python hand_distance.py --source input.mp4 --motion-threshold 4

# 极致加速：纯截图，不写视频
python hand_distance.py --source input.mp4 --no-video --no-annotate-video
```

### 3. apps/auto_label.py - Florence-2 自动标注

用 Florence-2 开放词汇检测把原始图片批量标注成 YOLO 数据集标签（坐标转归一化 `class_id cx cy w h`），一次共享提示框出全部类别。类别与提示词由 `classes.yaml` 定义。复用 `utils/core.py` 的收集/目录/logging 工具，无重复轮子。

#### 前置准备

1. 安装依赖（torchg 环境：`pip install transformers timm einops`）
2. 手动下载 Florence-2 safetensors 权重（`microsoft/Florence-2-base` 或 `-large`）到 `weights/florence2/` 目录
3. 编辑 `classes.yaml` 填入你的目标类别与提示词（顺序决定 class_id）

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/florence2/Florence-2-base` | Florence-2 模型本地目录 |
| `--source` | string | `imgs` | 待标注图片目录/单图/txt 列表 |
| `--classes` | string | `classes.yaml` | 类别定义 yaml |
| `--conf` | float | 0.35 | 置信度阈值 |
| `--save-dir` | string | `runs/florence_labels` | 结果父目录，每次生成独立 expN |
| `--device` | string | `cuda` | 推理设备 |
| `--no-fp16` | flag | False | 禁用 FP16 推理（仅 GPU 生效） |

#### 使用示例

```bash
python apps/auto_label.py --source imgs --classes classes.yaml
python apps/auto_label.py --source ./raw --model weights/florence2/Florence-2-base --conf 0.3
```

#### 输出结构

```
runs/florence_labels/exp1/
├── labels/<img>.txt      # YOLO 标签（class_id cx cy w h）
├── images/               # 拷贝的原图（与 cbook 参考数据集结构一致）
├── previews/<img>.png    # 带框预览图（人工抽检）
├── data.yaml             # 供 train.py 直接引用
└── summary.json          # 汇总（含每图检测明细）
```

### 4. apps/workflow_decensor.py - 漫画去码（检测 + ComfyUI）

先检测图像，有检测目标的图像通过 ComfyUI 去码流程处理，无检测目标的图像直接复制。对应 `workflows/f2k-漫画去码-py.json`。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `weights/censor_detect_v1.0_s_0725.pt` | 模型权重文件路径 |
| `--source` | string | `imgs` | 检测输入目录 |
| `--conf` | float | 0.6 | 检测置信度阈值 |
| `--detect-save-dir` | string | `runs/detections` | 检测结果保存目录 |
| `--save-json` | flag | False | 保存单个检测 json 文件 |
| `--workflow` | string | `workflows/f2k-漫画去码-py.json` | ComfyUI 工作流 JSON 路径 |
| `--comfyui-save-dir` | string | `runs/comfyui_output` | ComfyUI 结果保存目录 |
| `--comfyui-server` | string | `http://127.0.0.1:8188` | ComfyUI 服务器地址 |

#### 使用示例与处理流程

```bash
# 运行完整检测+ComfyUI 处理流程
python apps/workflow_decensor.py --source imgs/

# 指定 ComfyUI 服务器地址
python apps/workflow_decensor.py --source imgs/ --comfyui-server http://192.168.1.100:8188
```

处理流程：**步骤1** YOLO 检测图像 → **步骤2** 读取汇总 JSON，判断哪些图片有检测目标 → **步骤3** 无检测目标的图像直接复制到输出 → **步骤4** 有检测目标的图像通过 ComfyUI 处理。

### 5. apps/workflow_colorize.py - 漫画上色（ComfyUI）

将灰度漫画图批量送入 ComfyUI 上色流程着色。对应 `workflows/anima漫画上色-py.json`。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--source` | string | 配置默认 | 待上色图像目录 |
| `--output` | string | `runs` | 结果输出目录 |
| `--workflow` | string | `workflows/anima漫画上色-py.json` | ComfyUI 工作流 JSON 路径 |
| `--comfyui-server` | string | `http://127.0.0.1:8188` | ComfyUI 服务器地址 |

#### 使用示例

```bash
python apps/workflow_colorize.py --source imgs/ --output runs/colorized
```

### 6. apps/hand2label.py - 两步工作流（手部截图 → 自动标注）

串联 `hand_distance`（去重截图）与 `auto_label`（Florence-2 标注）：先对每段视频得到去重图目录，再汇总所有去重 jpg 喂给自动标注，生成 YOLO 标签 + 预览图 + summary.json。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--source` | string | 配置默认 | 源目录/视频/图片 |
| `--hand-model` | string | `weights/cbook-hand.pt` | 手部检测模型路径 |
| `--hand-save-dir` | string | `runs/hand_distance` | 手部截图结果父目录 |
| `--conf` / `--distance-threshold` / `--stable-duration` | - | 0.6 / 1200 / 1 | 手部截图阶段参数（同 hand_distance） |
| `--max-edge` / `--imgsz` / `--motion-threshold` | - | 1280 / 480 / 4 | 推理加速与静止判定参数 |
| `--no-fp16` / `--no-video` / `--no-annotate-video` | flag | False | 关闭 FP16 / 关视频 / 关标注写原帧 |
| `--florence-model` | string | `weights/florence2/Florence-2-base` | Florence-2 模型 |
| `--classes` | string | `classes.yaml` | 类别定义 |
| `--label-conf` | float | 0.35 | 标注置信度阈值 |
| `--label-save-dir` | string | `runs/florence_labels` | 标注结果父目录 |
| `--device` | string | `cuda` | Florence-2 推理设备 |
| `--no-label-fp16` | flag | False | 关闭标注阶段 FP16 |
| `--copy-undetected` | flag | False | 无检测目标时也复制图片并写空标签（默认跳过不复制） |
| `--export-max-edge` | int | None | 标注预处理最长边（0 表示不缩放） |

#### 使用示例

```bash
python apps/hand2label.py --source ./videos
python apps/hand2label.py --source ./videos --distance-threshold 1200 --stable-duration 2
```

### 7. train.py - 模型训练

使用 Ultralytics 平台训练 YOLO 模型。

```bash
python train.py
```

- **数据集**：Ultralytics 平台上的 `cbook` 数据集
- **模型**：YOLO26n
- **训练轮数**：100 epochs
- **项目/实验**：`lemon/my-project` / `v2i-3`

> **注意**：需要在 Ultralytics 平台注册账号并获取 API Key，替换脚本中的 `ULTRALYTICS_API_KEY`。

## 输出结果

### detect.py 输出

```
runs/detections/exp*/
├── <image_name>.jpg          # 检测后的图像（带检测框）
├── <image_name>.json         # 单个检测结果（--save-json 时生成）
├── det_<video_name>.mp4      # 处理后的视频（仅视频输入）
└── summary.json              # 汇总 JSON 文件
```

### hand_distance.py 输出

每个输入文件一个独立的 `expN` 目录（从 `exp1` 开始递增）：

```
runs/hand_distance/exp1/            # 第一个输入文件的结果目录
├── hand_distance_<video_name>.mp4  # 处理后的视频（仅视频输入且未用 --no-video）
├── screenshots/                    # 原始截图目录
│   └── screenshot_000000.png
├── <video_name>/                   # 去重后精选截图
│   └── screenshot_000123.jpg
├── distance_summary.txt            # 距离统计摘要（--no-save-txt 关闭）
└── frame_distance_log.txt          # 每帧距离日志（--no-save-txt 关闭）
```

### workflow_decensor.py 输出

```
runs/detections/exp*/           # 检测结果目录
runs/comfyui_output/            # ComfyUI 处理结果目录
```

## 裁剪比例说明

`--crop-ratio` 参数表示图像两边向中央裁剪的**总比例**：

- `crop_ratio=0`：不裁剪
- `crop_ratio=0.3`：左右各裁剪 15%，总共裁剪 30%，保留中间 70%
- `crop_ratio=0.4`：左右各裁剪 20%，总共裁剪 40%，保留中间 60%

## 模型权重

| 模型名称 | 路径 | 用途 |
|---------|------|------|
| 手部检测模型 | `./weights/cbook-hand.pt` | 检测手部（hand_distance 默认） |
| 审查检测模型 | `./weights/censor_detect_v1.0_s_0725.pt` | 检测审查目标 |
| YOLO26n | `./weights/ultralytics/yolo26/yolo26n/yolo26n.pt` | 通用检测 |
| SAM 模型 | `./weights/ultralytics/sam_b.pt` | 分割模型 |
| Florence-2 | `./weights/florence2/Florence-2-base` | 自动标注 |

## 原理说明

1. **手部检测**：使用 YOLO 模型检测图像中的手部目标
2. **距离计算**：提取左右手部的中心点坐标，计算欧氏距离
3. **稳定检测**：当距离超过阈值时开始计数，连续稳定指定时长后触发截图；配合 `--motion-threshold` 仅在画面静止时累计，保证截图帧清晰
4. **截图去重**：使用 ORB 特征匹配算法去除相似截图
5. **画册比例**：根据截图时的双手距离平均值除以图像宽度计算画册所占比例

### 加速原理

| 手段 | 说明 |
|------|------|
| 前向尺寸缩放（`--imgsz`） | 模型推理 FLOPs 近似随前向尺寸的平方下降，`imgsz=480` 相比默认 640 明显提速 |
| FP16 混合精度（默认开启） | GPU 上用半精度推理，速度接近翻倍；`--no-fp16` 可关闭 |
| 静止帧复用推理 | 画面静止帧跳过重复推理，直接复用上次检测框与标注图 |
| 异步读帧 | 后台线程预取解码帧，把 I/O 从推理主线程解耦 |
| 关视频输出（`--no-video`） | 跳过逐帧写入，纯截图最快路径 |

> **调参建议**：`motion_threshold` 越大判定越宽松（速度越快），但为保证"画面静止才截图"应配合实际素材微调；默认取 `4`，若素材对静止门槛要求不高可适当上调（如 8~20），追求更严静止则下调（如 2~3）。

## 许可证

MIT License