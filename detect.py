import argparse
import os
import cv2
import json
import logging
from tqdm import tqdm

from utils import get_next_exp_dir, collect_source_items, save_detection_results, load_yolo_model, configure_logging
from utils.config import DEFAULT_CENSOR_MODEL

# logger
logger = logging.getLogger(__name__)

def detect_single_frame(model, frame, conf, save_dir=None, filename=None, save_json=False, save_annotated=True, annotate=True):
    """
    检测单帧图像（公共函数）
    
    Args:
        model: YOLO模型对象
        frame: 图像帧（numpy数组）或图像路径
        conf: 置信度阈值
        save_dir: 保存目录（可选）
        filename: 保存文件名（可选）
        save_json: 是否保存json
        save_annotated: 是否保存带检测框的图像
        annotate: 是否计算标注图像（False时跳过耗时的 plot()，返回None）
    
    Returns:
        dict: 检测结果数据
        numpy.ndarray: 标注后的图像（annotate=False 或不需要时返回 None）
    """
    results = model(frame, conf=conf, save=False, verbose=False)
    result = results[0]
    
    data = {
        "image_filename": filename,
        "detection_count": 0,
        "detections": []
    }
    
    if result.boxes is not None and len(result.boxes) > 0:
        detections = []
        for box in result.boxes:
            cls = int(box.cls)
            conf_val = float(box.conf)
            xywh = box.xywh[0].tolist()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls,
                "confidence": conf_val,
                "bbox_xywh": xywh,
                "bbox_xyxy": [x1, y1, x2, y2]
            })
        
        data["detection_count"] = len(detections)
        data["detections"] = detections
        
        if save_dir and filename and save_json:
            json_path = os.path.join(save_dir, os.path.splitext(filename)[0] + '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    if save_dir and filename:
        save_detection_results(result, save_dir, filename, save_json, save_annotated)
    
    return data, result.plot() if annotate else None

def run_image_detection(model, source, conf, save_dir, save_json, save_annotated=True):
    """运行图像检测（使用已加载的 model 对象）"""
    model.eval()

    results = model(
        source=source,
        conf=conf,
        save=False,
        verbose=False
    )

    logger.info("正在保存检测结果...")
    for result in tqdm(results, desc="处理图像", unit="张"):
        img_name = os.path.basename(result.path)
        save_detection_results(result, save_dir, img_name, save_json, save_annotated)

    logger.info(f"\n所有结果已保存到 {save_dir}")

def run_video_detection(model, source, conf, save_dir, save_json, save_annotated=True, annotate_video=None, sample_interval=1):
    """运行视频检测（使用已加载的 model 对象）

    sample_interval>1 时开启抽帧：仅每隔 N 帧做一次推理，未抽中的帧跳过推理，
    沿用最近一次标注结果（annotate）或直接复用原帧（不标注）写回输出视频。
    输出视频始终写入全部帧、保持原帧率；JSON 与汇总只记录实际抽帧推理的真实数据，
    避免因沿用上一帧结果而重复计数。
    """
    model.eval()

    # annotate_video 默认跟随 save_annotated：不保存标注时跳过逐帧 plot()，只输出原帧
    if annotate_video is None:
        annotate_video = save_annotated
    sample_interval = max(1, int(sample_interval))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"错误：无法打开视频文件 {source}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(save_dir, exist_ok=True)
    
    video_name = os.path.basename(source)
    output_video_path = os.path.join(save_dir, f"det_{video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    video_results = []
    last_annotated = None  # 供未抽中帧沿用最近的标注结果
    
    logger.info(f"正在处理视频: {video_name} (抽帧间隔={sample_interval})")
    with tqdm(total=total_frames, desc="处理帧", unit="帧") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = None
            if frame_count % sample_interval == 0:
                # 抽中的帧：真正执行推理
                frame_data, annotated_frame = detect_single_frame(
                    model, frame, conf, 
                    save_dir=save_dir if save_json else None,
                    filename=f'frame_{frame_count:04d}.jpg' if save_json else None,
                    save_json=save_json,
                    save_annotated=save_annotated,
                    annotate=annotate_video
                )
                video_results.append(frame_data)
                if annotated_frame is not None:
                    last_annotated = annotated_frame
            else:
                # 未抽中的帧：跳过推理，沿用最近的标注结果（缺失时退化为原帧）
                annotated_frame = last_annotated
            
            out.write(annotated_frame if annotated_frame is not None else frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    logger.info(f"\n视频检测完成。输出已保存到 {output_video_path}")
    
    return video_results

def run_detection(model, model_path, source, conf, save_dir, save_json, save_annotated=True, sample_interval=1, image_only=False, annotated_dir=None):
    """运行检测主函数（model 已加载）

    image_only=True 时仅处理图片，跳过视频文件（供 detect_comfyui 等仅图片流程复用）。
    annotated_dir=True 时，图片分支额外保存一份带检测框的预览图到 <save_dir>/annotated，
    不影响 save_dir（folder2/源图）的正常保存，用于人工核对检测结果。
    返回的第三项为实际带框预览目录（未启用时为 None）。

    Returns:
        (save_dir, all_results, annotated_dir): 实际输出目录、逐图检测结果、带框预览目录；失败返回 (None, None, None)
    """
    save_dir = get_next_exp_dir(save_dir or 'runs/detect')
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"结果将保存到: {save_dir}")

    if annotated_dir:
        annotated_dir = os.path.join(save_dir, 'annotated')
        os.makedirs(annotated_dir, exist_ok=True)
        logger.info(f"带检测框预览将保存到: {annotated_dir}")

    items, error = collect_source_items(source, image_only=image_only)
    if error:
        logger.error(f"错误：{error}")
        return None, None, None

    all_results = []
    with tqdm(total=len(items), desc="处理文件", unit="个") as pbar:
        for path, is_video in items:
            if is_video:
                video_results = run_video_detection(model, path, conf, save_dir, save_json, save_annotated, sample_interval=sample_interval)
                if video_results:
                    all_results.extend(video_results)
            else:
                results = model(path, conf=conf, save=False, verbose=False)
                base_name = os.path.basename(path)
                img_results = save_detection_results(results[0], save_dir, base_name, save_json, save_annotated)
                if annotated_dir:
                    # 独立目录再存一份带框预览，供人工检查，不影响源图
                    save_detection_results(results[0], annotated_dir, base_name, save_json=False, save_annotated=True)
                if img_results:
                    all_results.append(img_results)
            pbar.update(1)
    logger.info(f"\n所有文件处理完成。结果已保存到 {save_dir}")

    # 生成汇总json文件（始终保存）
    if all_results:
        summary = {
            "model_path": model_path,
            "source": source,
            "confidence_threshold": conf,
            "total_files_processed": len(all_results),
            "total_detections": sum(r["detection_count"] for r in all_results),
            "results": all_results
        }

        summary_path = os.path.join(save_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"\n汇总文件已保存: {summary_path}")

    return save_dir, all_results, annotated_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO手部检测推理脚本')
    parser.add_argument('--model', type=str, default=DEFAULT_CENSOR_MODEL, help='模型权重文件路径（手部跟踪使用手部检测权重）')
    parser.add_argument('--source', type=str, default=r'imgs', help='源目录、图像路径或视频文件路径')
    parser.add_argument('--conf', type=float, default=0.6, help='置信度阈值')
    parser.add_argument('--save-dir', type=str, default='runs\detections', help='结果保存目录')
    parser.add_argument('--save-json', default=False, action='store_true', help='保存单个检测json文件（summary.json始终保存）')
    parser.add_argument('--save-annotated', default=True, action='store_true', help='保存带检测框的图片（默认启用）')
    parser.add_argument('--no-annotated', dest='save_annotated', action='store_false', help='保存原图，不带检测框')
    parser.add_argument('--device', type=str, default='cuda', help='推理设备（例如 cpu 或 cuda）')
    parser.add_argument('--sample-interval', type=int, default=1, help='视频抽帧间隔，>1 时每隔 N 帧做一次推理，其余帧沿用上一帧标注（仅视频检测生效）')

    args = parser.parse_args()

    # 初始化日志
    configure_logging()
    logger.info(f"使用模型: {args.model}")

    # 使用统一的模型加载函数
    model, device_info = load_yolo_model(args.model)
    if model is None:
        logger.error("模型加载失败")
        exit(1)
    
    run_detection(model, args.model, args.source, args.conf, args.save_dir, args.save_json, args.save_annotated, args.sample_interval)