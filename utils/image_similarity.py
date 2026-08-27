import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from utils import get_next_exp_dir, imread_unicode

def crop_center(image, ratio=0.5):
    height, width = image.shape[:2]
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    return image[start_y:start_y+new_height, start_x:start_x+new_width]

def orb_distance(image1, image2, use_center=False, center_ratio=0.5):
    if use_center:
        image1 = crop_center(image1, center_ratio)
        image2 = crop_center(image2, center_ratio)
    
    orb = cv2.ORB_create(nfeatures=500)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 1.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) == 0:
        return 1.0
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = [m for m in matches if m.distance < 50]
    
    if max(len(kp1), len(kp2)) == 0:
        return 1.0
    
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    
    return 1.0 - similarity

def compare_images(img_path1, img_path2, threshold=0.7, use_center=False, center_ratio=0.5):
    img1 = imread_unicode(img_path1)
    img2 = imread_unicode(img_path2)
    
    if img1 is None:
        print(f"错误：无法读取图片 {img_path1}")
        return None
    if img2 is None:
        print(f"错误：无法读取图片 {img_path2}")
        return None
    
    distance = orb_distance(img1, img2, use_center, center_ratio)
    similarity = 100.0 * (1 - distance)
    
    return {
        'distance': distance,
        'similarity': similarity,
        'is_similar': distance <= threshold
    }

def filter_unique_images(input_dir, output_dir, threshold=0.7, use_center=False, center_ratio=0.5):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(image_extensions)])
    
    print(f"在输入目录中找到 {len(files)} 张图片")
    if use_center:
        print(f"使用中心区域检测，比例: {center_ratio}")
    
    if len(files) == 0:
        print("未找到图片")
        return [], []
    
    images = {}
    for filename in tqdm(files, desc="正在加载图片", unit="张"):
        filepath = os.path.join(input_dir, filename)
        img = imread_unicode(filepath)
        if img is not None:
            images[filename] = img
    
    unique_images = []
    removed_images = []
    
    if files:
        unique_images.append(files[0])
        last_kept = files[0]
        
        for filename in tqdm(files[1:], desc="正在比较图片", unit="张"):
            if last_kept not in images or filename not in images:
                unique_images.append(filename)
                last_kept = filename
                continue
                
            distance = orb_distance(images[last_kept], images[filename], use_center, center_ratio)
            
            if distance <= threshold:
                removed_images.append(filename)
                # print(f"  Remove {filename} (similar to {last_kept}, distance={distance:.4f})")
            else:
                unique_images.append(filename)
                last_kept = filename
                # print(f"  Keep {filename} (different from {last_kept}, distance={distance:.4f})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    for filename in tqdm(unique_images, desc="Saving unique images", unit="image"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        if os.path.exists(src_path):
            img = imread_unicode(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)
                saved_count += 1
    
    txt_path = os.path.join(output_dir, 'filter_report.txt')
    with open(txt_path, 'w') as f:
        f.write(f"算法: ORB\n")
        f.write(f"阈值: {threshold}\n")
        f.write(f"处理图片总数: {len(files)}\n")
        f.write(f"去重后保存: {len(unique_images)}\n")
        f.write(f"已移除图片: {len(removed_images)}\n")
        
        if removed_images:
            f.write("\n已移除的图片（与前一张相似）:\n")
            for filename in removed_images:
                f.write(f"  {filename}\n")
        
        f.write("\n保留的图片:\n")
        for filename in unique_images:
            f.write(f"  {filename}\n")
    
    print(f"\n=== 去重完成 ===")
    print(f"算法: ORB")
    print(f"阈值: {threshold}")
    print(f"处理图片总数: {len(files)}")
    print(f"去重后保存: {len(unique_images)}")
    print(f"已移除图片: {len(removed_images)}")
    print(f"结果保存到: {output_dir}")
    
    return unique_images, removed_images

def run_similarity_check(args):
    if args.compare:
        if len(args.compare) != 2:
            print("请提供恰好两张图片进行比较")
            return
        
        result = compare_images(args.compare[0], args.compare[1], args.threshold, args.use_center, args.center_ratio)
        if result:
            print(f"\n比较结果:")
            print(f"图片1: {args.compare[0]}")
            print(f"图片2: {args.compare[1]}")
            print(f"距离: {result['distance']:.4f}")
            print(f"相似度: {result['similarity']:.2f}%")
            print(f"是否相似: {result['is_similar']}")
    
    elif args.filter_unique:
        output_dir = args.output or get_next_exp_dir('runs/filter_unique')
        filter_unique_images(args.filter_unique, output_dir, args.threshold, args.use_center, args.center_ratio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用 ORB 算法进行图片相似度检测')
    
    parser.add_argument('--compare', nargs=2, metavar=('图片1', '图片2'),
                        help='比较两张图片的相似度')
    
    parser.add_argument('--filter-unique', type=str, default=None,
                        help='输入图片目录')
    
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='相似度距离阈值 (0.0=完全相同, 1.0=完全不同)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出目录，默认 runs/filter_unique（自动递增防覆盖）')
    
    parser.add_argument('--use-center', action='store_true',
                        help='使用图片中心区域进行相似度检测')
    
    parser.add_argument('--center-ratio', type=float, default=0.5,
                        help='中心区域比例，默认0.5表示使用宽高各一半的中心区域')
    
    args = parser.parse_args()
    
    if not (args.compare or args.filter_unique):
        parser.print_help()
        print("\n请指定以下参数之一: --compare 或 --filter-unique")
    else:
        run_similarity_check(args)