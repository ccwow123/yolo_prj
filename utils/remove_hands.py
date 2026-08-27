import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from .utils import get_next_exp_dir, load_yolo_model, imread_unicode

def remove_hands(image_path, model, conf=0.4, book_width_ratio=0.0, min_area=5000, 
                 max_area=100000, aspect_ratio_range=(0.3, 3.0), visualize_mask=False):
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    h_img, w_img = img.shape[:2]
    
    book_area = None
    if book_width_ratio > 0:
        book_width = int(w_img * book_width_ratio)
        book_x1 = (w_img - book_width) // 2
        book_x2 = book_x1 + book_width
        book_area = (book_x1, 0, book_x2, h_img)
        print(f"  自动计算书页区域: ({book_x1}, 0) - ({book_x2}, {h_img})")
    
    results = model.predict(image_path, conf=conf, iou=0.45, verbose=False)
    
    hand_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    for res in results:
        if res.masks is None:
            continue
        
        for mask, box in zip(res.masks.data, res.boxes):
            cls_id = int(box.cls)
            cls_name = res.names[cls_id]
            
            seg_mask = cv2.resize(mask.cpu().numpy(), (w_img, h_img))
            
            area = np.sum(seg_mask > 0.5)
            
            if area < min_area or area > max_area:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            aspect_ratio = bbox_w / bbox_h
            
            if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                continue
            
            moments = cv2.moments((seg_mask > 0.5).astype(np.uint8))
            if moments["m00"] == 0:
                continue
            
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            if book_area is not None:
                book_x1, book_y1, book_x2, book_y2 = book_area
                if not (book_x1 < cx < book_x2 and book_y1 < cy < book_y2):
                    hand_mask[seg_mask > 0.5] = 255
            else:
                hand_mask[seg_mask > 0.5] = 255
    
    img_no_hand = img.copy()
    
    if np.any(hand_mask == 255):
        img_no_hand[hand_mask == 255] = [255, 255, 255]
        print("  手部区域已填充白色")
    else:
        print("  未检测到手部")
    
    plot_img = None
    if len(results) > 0:
        plot_img = res.plot()
    
    return img_no_hand, hand_mask, plot_img

def create_comparison(img_original, img_result, mask):
    h, w = img_original.shape[:2]
    
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_rgb[mask == 255] = [0, 0, 255]
    
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    combined[:, :w] = img_original
    combined[:, w:w*2] = mask_rgb
    combined[:, w*2:] = img_result
    
    cv2.putText(combined, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Hand Mask", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Result", (w * 2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined

def process_folder(input_dir, output_dir, model_path, conf=0.4, book_width_ratio=0.0, 
                   min_area=5000, max_area=100000, aspect_ratio_range=(0.3, 3.0), visualize=False):
    model, device_info = load_yolo_model(model_path)
    if model is None:
        raise RuntimeError(f"无法加载模型: {model_path}")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(image_extensions)])
    
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    if visualize:
        compare_dir = os.path.join(output_dir, 'comparison')
        os.makedirs(compare_dir, exist_ok=True)
        seg_dir = os.path.join(output_dir, 'segmentation')
        os.makedirs(seg_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    print(f"发现 {len(files)} 张图片")
    print(f"使用模型: {model_path}")
    print(f"置信度阈值: {conf}")
    print(f"面积范围: {min_area} - {max_area}")
    print(f"宽高比范围: {aspect_ratio_range[0]} - {aspect_ratio_range[1]}")
    if book_width_ratio > 0:
        print(f"书页宽度比例: {book_width_ratio * 100:.1f}% (自动居中)")
        print("仅处理书页区域外的手部")
    
    with tqdm(total=len(files), desc="处理图片", unit="张") as pbar:
        for filename in files:
            try:
                input_path = os.path.join(input_dir, filename)
                name, ext = os.path.splitext(filename)
                
                img_no_hand, mask, plot_img = remove_hands(
                    input_path, model, conf, book_width_ratio, min_area, max_area, aspect_ratio_range
                )
                
                output_path = os.path.join(output_dir, f"{name}_nohand{ext}")
                if ext.lower() in ['.jpg', '.jpeg']:
                    cv2.imwrite(output_path, img_no_hand, [cv2.IMWRITE_JPEG_QUALITY, 85])
                else:
                    cv2.imwrite(output_path, img_no_hand)
                
                mask_path = os.path.join(mask_dir, f"{name}_mask.png")
                cv2.imwrite(mask_path, mask)
                
                if visualize:
                    original = imread_unicode(input_path)
                    comparison = create_comparison(original, img_no_hand, mask)
                    comparison_path = os.path.join(compare_dir, f"{name}_compare.png")
                    cv2.imwrite(comparison_path, comparison)
                    
                    if plot_img is not None:
                        seg_path = os.path.join(seg_dir, f"{name}_seg.png")
                        cv2.imwrite(seg_path, plot_img)
                
                status = "含手" if np.any(mask > 0) else "无手"
                success_count += 1
                print(f"  ✓ {filename} -> {name}_nohand{ext} [{status}]")
                
            except Exception as e:
                fail_count += 1
                print(f"  ✗ {filename} -> 失败: {str(e)}")
            
            pbar.update(1)
    
    report_path = os.path.join(output_dir, 'processing_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 手部移除处理报告 ===\n\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"使用模型: {model_path}\n")
        f.write(f"置信度阈值: {conf}\n")
        f.write(f"面积范围: {min_area} - {max_area}\n")
        f.write(f"宽高比范围: {aspect_ratio_range[0]} - {aspect_ratio_range[1]}\n")
        if book_width_ratio > 0:
            f.write(f"书页宽度比例: {book_width_ratio * 100:.1f}%\n")
        f.write(f"处理总数: {len(files)}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {fail_count}\n")
        f.write(f"成功率: {success_count / len(files) * 100:.1f}%\n")
    
    print(f"\n=== 处理完成 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理总数: {len(files)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='使用YOLO手部分割模型移除图片中的手部')
    parser.add_argument('--input', type=str, default=r'./runs/hand_distance/exp9/screenshots', help='输入图片目录')
    parser.add_argument('--output', type=str, default=r'./runs/remove_hands/', help='输出目录')
    parser.add_argument('--model', type=str, default=r'./weights/ultralytics/yolov8s-seg.pt', help='手部分割模型路径')
    parser.add_argument('--conf', type=float, default=0.4, help='置信度阈值，越高误检越少')
    parser.add_argument('--min-area', type=int, default=5000, help='最小面积阈值，过滤小面积误检')
    parser.add_argument('--max-area', type=int, default=100000, help='最大面积阈值，过滤大面积误检')
    parser.add_argument('--aspect-ratio', type=float, nargs=2, default=[0.3, 3.0], 
                        help='宽高比范围，过滤畸形检测框')
    parser.add_argument('--book-width-ratio', type=float, default=0.7, 
                        help='书页宽度占原图宽度的比例，默认0.7，自动居中')
    parser.add_argument('--visualize', action='store_true', help='生成对比图和分割可视化')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    output_dir = get_next_exp_dir(args.output)
    print(f"输出目录: {output_dir}")
    
    process_folder(
        args.input,
        output_dir,
        args.model,
        conf=args.conf,
        book_width_ratio=args.book_width_ratio,
        min_area=args.min_area,
        max_area=args.max_area,
        aspect_ratio_range=tuple(args.aspect_ratio),
        visualize=args.visualize
    )

if __name__ == '__main__':
    main()