import argparse
import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def get_next_exp_dir(base_dir='runs/detect'):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith('exp')]
    
    if not existing_dirs:
        return os.path.join(base_dir, 'exp')
    
    max_num = 0
    for d in existing_dirs:
        if d == 'exp':
            num = 1
        elif d.startswith('exp') and d[3:].isdigit():
            num = int(d[3:])
        else:
            continue
        if num > max_num:
            max_num = num
    
    if max_num == 0:
        return os.path.join(base_dir, 'exp')
    else:
        return os.path.join(base_dir, f'exp{max_num + 1}')

def run_image_detection(model_path, source, conf, save_dir, save_txt):
    model = YOLO(model_path)
    
    results = model(
        source=source,
        conf=conf,
        save=False,
        verbose=False
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving detection results...")
    for result in tqdm(results, desc="Processing images", unit="image"):
        img_path = result.path
        img_name = os.path.basename(img_path)
        
        if save_txt:
            txt_path = os.path.join(save_dir, os.path.splitext(img_name)[0] + '.txt')
            with open(txt_path, 'w') as f:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    xywh = box.xywh[0].tolist()
                    line = f"{cls} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]} {conf}\n"
                    f.write(line)
        
        result.save(os.path.join(save_dir, img_name))
    
    print(f"\nAll results saved to {save_dir}")

def run_video_detection(model_path, source, conf, save_dir, save_txt):
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(save_dir, exist_ok=True)
    
    output_video_path = os.path.join(save_dir, 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print("Processing video frames...")
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
            
            if save_txt:
                txt_path = os.path.join(save_dir, f'frame_{frame_count:04d}.txt')
                with open(txt_path, 'w') as f:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        xywh = box.xywh[0].tolist()
                        line = f"{cls} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]} {conf}\n"
                        f.write(line)
            
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"\nVideo detection completed. Output saved to {output_video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO26 Inference Script')
    parser.add_argument('--video', default=False,  help='Enable video detection mode') # 是否启用视频检测模式


    parser.add_argument('--model', type=str, default=r'.\weights\ultralytics\yolo26\yolo26n\yolo26_v2i.pt', help='Path to model weights')
    parser.add_argument('--source', type=str, default=r'E:\Files\video_to_imgs\all_frames', help='Source directory, image path, or video file path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-dir', type=str, default=None, help='Output directory (auto-incrementing if not specified)')
    parser.add_argument('--save-txt', default=True, help='Save detection results as txt files') # 是否保存检测结果的txt文件


    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = get_next_exp_dir()
    
    print(f"Results will be saved to: {args.save_dir}")
    
    if args.video:
        run_video_detection(args.model, args.source, args.conf, args.save_dir, args.save_txt)
    else:
        run_image_detection(args.model, args.source, args.conf, args.save_dir, args.save_txt)