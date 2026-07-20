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

## 安装依赖

```bash
pip install ultralytics opencv-python numpy tqdm
```

## 使用方法

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | string | `./weights/ultralytics/hand_yolov8n.pt` | 手部检测模型权重路径 |
| `--source` | string | `videoes/video (1).mp4` | 源目录、图片路径或视频文件路径 |
| `--conf` | float | 0.6 | 置信度阈值 |
| `--save-dir` | string | None | 输出目录（未指定时自动递增） |
| `--save-txt` | flag | True | 保存距离结果为txt文件 |
| `--distance-threshold` | int | 1400 | 触发截图的距离阈值（像素） |
| `--stable-duration` | float | 2.0 | 触发截图所需的稳定时长（秒） |
| `--crop-ratio` | float | 0 | 图像两边向中央裁剪的总比例 |
| `--quality` | int | 80 | 图像/视频压缩质量（1-100） |

### 使用示例

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

## 输出结果

### 目录结构

```
runs/hand_distance/exp*/
├── hand_distance_<video_name>.mp4  # 处理后的视频（仅视频输入）
├── screenshots/                    # 截图目录
│   ├── screenshot_000000.jpg
│   ├── screenshot_000245.jpg
│   └── ...
├── distance_summary.txt            # 距离统计摘要
└── frame_distance_log.txt          # 每帧距离日志
```

### 统计信息

处理完成后会输出以下统计信息：

```
=== 视频处理完成 ===
输出视频: runs/hand_distance/exp1/hand_distance_input.mp4
截图保存到: runs/hand_distance/exp1/screenshots

=== 截图统计 ===
捕获截图总数: 15
去重后截图数: 8
已移除重复截图: 7

平均距离: 1450.5 px

=== 截图时双手距离统计 ===
截图时距离列表: 1500.5, 1480.3, 1520.1 px
截图时平均距离: 1500.3 px
图像宽度: 1920 px
画册所占比例: 0.7814 (78.14%)
```

## 裁剪比例说明

`--crop-ratio` 参数表示图像两边向中央裁剪的**总比例**：

- `crop_ratio=0`：不裁剪
- `crop_ratio=0.3`：左右各裁剪15%，总共裁剪30%，保留中间70%
- `crop_ratio=0.4`：左右各裁剪20%，总共裁剪40%，保留中间60%

## 模型权重

请将YOLOv8手部检测模型权重文件放置在 `./weights/ultralytics/` 目录下，命名为 `hand_yolov8n.pt`。

## 原理说明

1. **手部检测**：使用YOLOv8模型检测图像中的手部目标
2. **距离计算**：提取左右手部的中心点坐标，计算欧氏距离
3. **稳定检测**：当距离超过阈值时开始计数，连续稳定指定时长后触发截图
4. **截图去重**：使用ORB特征匹配算法去除相似截图
5. **画册比例**：根据截图时的双手距离平均值除以图像宽度计算画册所占比例

## 许可证

MIT License