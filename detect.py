import argparse
import os
import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm

from utils import get_next_exp_dir, is_video_file, save_detection_results

def run_image_detection(model_path, source, conf, save_dir, save_json):
    model = YOLO(model_path)
    
    results = model(
        source=source,
        conf=conf,
        save=False,
        verbose=False
    )
    
    print("Saving detection results...")
    for result in tqdm(results, desc="Processing images", unit="image"):
        img_name = os.path.basename(result.path)
        save_detection_results(result, save_dir, img_name, save_json)
    
    print(f"\nAll results saved to {save_dir}")

def run_video_detection(model_path, source, conf, save_dir, save_json):
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {source}")
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
    
    # 收集视频帧的检测结果
    video_results = []
    
    print(f"Processing video: {video_name}")
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(
                frame,
                conf=conf,
                save=False,
                verbose=False
            )
            
            result = results[0]
            annotated_frame = result.plot()
            
            frame_data = {
                "image_filename": f'frame_{frame_count:04d}.jpg',
                "detection_count": 0,
                "detections": []
            }
            
            if save_json and result.boxes is not None:
                json_path = os.path.join(save_dir, f'frame_{frame_count:04d}.json')
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
                
                frame_data["detection_count"] = len(detections)
                frame_data["detections"] = detections
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(frame_data, f, indent=2, ensure_ascii=False)
            
            video_results.append(frame_data)
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"\nVideo detection completed. Output saved to {output_video_path}")
    
    return video_results

def run_detection(model_path, source, conf, save_dir, save_json):
    if save_dir is None:
        save_dir = get_next_exp_dir('runs/detect')
    else:
        save_dir = get_next_exp_dir(save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # 收集所有检测结果用于汇总
    all_results = []
    
    if os.path.isfile(source):
        if is_video_file(source):
            video_results = run_video_detection(model_path, source, conf, save_dir, save_json)
            if video_results:
                all_results.extend(video_results)
        else:
            print(f"Processing single image: {os.path.basename(source)}")
            model = YOLO(model_path)
            results = model(source, conf=conf, save=False, verbose=False)
            img_results = save_detection_results(results[0], save_dir, os.path.basename(source), save_json)
            if img_results:
                all_results.append(img_results)
            print(f"Saved result for {os.path.basename(source)}")
    
    elif os.path.isdir(source):
        files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
        
        with tqdm(total=len(files), desc="Processing files", unit="file") as pbar:
            for filename in files:
                filepath = os.path.join(source, filename)
                if is_video_file(filepath):
                    video_results = run_video_detection(model_path, filepath, conf, save_dir, save_json)
                    if video_results:
                        all_results.extend(video_results)
                else:
                    model = YOLO(model_path)
                    results = model(filepath, conf=conf, save=False, verbose=False)
                    img_results = save_detection_results(results[0], save_dir, filename, save_json)
                    if img_results:
                        all_results.append(img_results)
                pbar.update(1)
        print(f"\nAll files processed. Results saved to {save_dir}")
    
    else:
        print(f"Error: Source {source} not found")
        return
    
    # 生成汇总json文件
    if save_json and all_results:
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
        print(f"\n汇总文件已保存: {summary_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Hand Detection Inference Script')
    parser.add_argument('--model', type=str, default=r'weights\erax-anti-nsfw-yolo11s-v1.1.pt', help='Path to model weights (use hand detection weights for hand tracking)')
    parser.add_argument('--source', type=str, default=r'E:\Share\111', help='Source directory, image path, or video file path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-dir', type=str, default='runs\detections', help='Output directory for saving results')
    parser.add_argument('--save-json', default=True, action='store_true', help='Save detection results as json files')
    
    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    run_detection(args.model, args.source, args.conf, args.save_dir, args.save_json)