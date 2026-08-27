import argparse
import os
import cv2
import json
import logging
from tqdm import tqdm

from utils import get_next_exp_dir, is_video_file, is_image_file, save_detection_results, load_yolo_model

# logger
logger = logging.getLogger(__name__)

def detect_single_frame(model, frame, conf, save_dir=None, filename=None, save_json=False, save_annotated=True):
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
    
    Returns:
        dict: 检测结果数据
        numpy.ndarray: 标注后的图像（如果需要）
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
    
    return data, result.plot()

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

def run_video_detection(model, source, conf, save_dir, save_json, save_annotated=True):
    """运行视频检测（使用已加载的 model 对象）"""
    model.eval()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {source}")
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
    
    logger.info(f"正在处理视频: {video_name}")
    with tqdm(total=total_frames, desc="处理帧", unit="帧") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_data, annotated_frame = detect_single_frame(
                model, frame, conf, 
                save_dir=save_dir if save_json else None,
                filename=f'frame_{frame_count:04d}.jpg' if save_json else None,
                save_json=save_json
            )
            
            video_results.append(frame_data)
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    logger.info(f"\n视频检测完成。输出已保存到 {output_video_path}")
    
    return video_results

def run_detection(model, model_path, source, conf, save_dir, save_json, save_annotated=True):
    """运行检测主函数（model 已加载）"""
    if save_dir is None:
        save_dir = get_next_exp_dir('runs/detect')
    else:
        save_dir = get_next_exp_dir(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"结果将保存到: {save_dir}")

    all_results = []

    if os.path.isfile(source):
        if is_video_file(source):
            video_results = run_video_detection(model, source, conf, save_dir, save_json, save_annotated)
            if video_results:
                all_results.extend(video_results)
        else:
            logger.info(f"处理单张图像: {os.path.basename(source)}")
            results = model(source, conf=conf, save=False, verbose=False)
            img_results = save_detection_results(results[0], save_dir, os.path.basename(source), save_json, save_annotated)
            if img_results:
                all_results.append(img_results)
            logger.info(f"已保存 {os.path.basename(source)} 的结果")

    elif os.path.isdir(source):
        files = [f for f in os.listdir(source)
                 if os.path.isfile(os.path.join(source, f))
                 and (is_image_file(f) or is_video_file(f))]

        with tqdm(total=len(files), desc="处理文件", unit="个") as pbar:
            for filename in files:
                filepath = os.path.join(source, filename)
                if is_video_file(filepath):
                    video_results = run_video_detection(model, filepath, conf, save_dir, save_json, save_annotated)
                    if video_results:
                        all_results.extend(video_results)
                else:
                    results = model(filepath, conf=conf, save=False, verbose=False)
                    img_results = save_detection_results(results[0], save_dir, filename, save_json, save_annotated)
                    if img_results:
                        all_results.append(img_results)
                pbar.update(1)
        logger.info(f"\n所有文件处理完成。结果已保存到 {save_dir}")

    else:
        logger.error(f"错误：未找到源文件/目录 {source}")
        return None

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

    return save_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO手部检测推理脚本')
    parser.add_argument('--model', type=str, default=r'weights\censor_detect_v1.0_s_0725.pt', help='模型权重文件路径（手部跟踪使用手部检测权重）')
    parser.add_argument('--source', type=str, default=r'imgs', help='源目录、图像路径或视频文件路径')
    parser.add_argument('--conf', type=float, default=0.6, help='置信度阈值')
    parser.add_argument('--save-dir', type=str, default='runs\detections', help='结果保存目录')
    parser.add_argument('--save-json', default=False, action='store_true', help='保存单个检测json文件（summary.json始终保存）')
    parser.add_argument('--save-annotated', default=True, action='store_true', help='保存带检测框的图片（默认启用）')
    parser.add_argument('--no-annotated', dest='save_annotated', action='store_false', help='保存原图，不带检测框')
    parser.add_argument('--device', type=str, default='cuda', help='推理设备（例如 cpu 或 cuda）')

    args = parser.parse_args()

    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    logger.info(f"使用模型: {args.model}")

    # 使用统一的模型加载函数
    model, device_info = load_yolo_model(args.model)
    if model is None:
        logger.error("模型加载失败")
        exit(1)
    
    run_detection(model, args.model, args.source, args.conf, args.save_dir, args.save_json, args.save_annotated)