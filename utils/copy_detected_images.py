import argparse
import os
import json
import shutil

def load_summary(summary_path):
    """加载summary.json文件"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def copy_detected_images(summary_path, source_dir, target_dir, classes=None, min_detections=1):
    """
    从summary.json中筛选图片并复制到目标文件夹
    
    Args:
        summary_path: summary.json文件路径
        source_dir: 原始图片所在目录
        target_dir: 目标文件夹
        classes: 要筛选的类别名称或类别ID列表（可选，默认为None表示检测到任何结果都复制）
        min_detections: 最小检测数量（可选，默认1，即至少检测到一个目标）
    """
    # 加载summary.json
    summary = load_summary(summary_path)
    
    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)
    
    # 统计复制的文件数量
    copied_count = 0
    
    print(f"开始处理summary.json: {summary_path}")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    if classes:
        print(f"筛选类别: {', '.join(classes)}")
    else:
        print(f"筛选条件: 检测到至少 {min_detections} 个目标")
    
    # 遍历所有检测结果
    for result in summary['results']:
        image_filename = result['image_filename']
        detection_count = result['detection_count']
        detections = result['detections']
        
        # 检查是否满足条件
        should_copy = False
        
        if classes:
            # 检查是否检测到指定类别中的任何一个
            for det in detections:
                det_class_name = det.get('class_name', '')
                det_class_id = str(det.get('class_id', ''))
                
                for cls in classes:
                    if det_class_name == cls or det_class_id == cls:
                        should_copy = True
                        break
                if should_copy:
                    break
        else:
            # 检查检测数量是否满足
            if detection_count >= min_detections:
                should_copy = True
        
        if should_copy:
            # 构建源文件路径
            source_path = os.path.join(source_dir, image_filename)
            
            if os.path.exists(source_path):
                # 构建目标文件路径
                target_path = os.path.join(target_dir, image_filename)
                
                # 处理文件名冲突
                if os.path.exists(target_path):
                    base_name, ext = os.path.splitext(image_filename)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
                        counter += 1
                
                # 复制文件
                shutil.copy2(source_path, target_path)
                copied_count += 1
                print(f"复制: {image_filename}")
            else:
                print(f"警告: 文件不存在 - {source_path}")
    
    print(f"\n处理完成！共复制 {copied_count} 个文件")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从summary.json筛选并复制检测到目标的图片')
    parser.add_argument('--summary', type=str, default=r'runs\detections\exp\summary.json', help='summary.json文件路径')
    parser.add_argument('--source-dir', type=str, default=r'E:\Share\剩下', help='原始图片所在目录')
    parser.add_argument('--target-dir', type=str, default=r'E:\Share\剩下\detected', help='目标文件夹路径')
    parser.add_argument('--classes', type=str, default=None, help='要筛选的类别，多个类别用[x,y,z]格式，None表示检测到任何结果都复制')
    parser.add_argument('--min-detections', type=int, default=1, help='最小检测数量（默认1）')
    
    args = parser.parse_args()
    
    copy_detected_images(
        summary_path=args.summary,
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        classes=args.classes,
        min_detections=args.min_detections
    )