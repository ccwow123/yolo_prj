import os
import json
import cv2


def get_next_exp_dir(base_dir='runs/exp'):
    """
    获取下一个实验目录路径（自动递增）
    
    Args:
        base_dir: 基础目录路径
    
    Returns:
        下一个实验目录的完整路径，格式为 base_dir/exp 或 base_dir/exp1, exp2...
    """
    os.makedirs(base_dir, exist_ok=True)
    exp_num = 0
    while True:
        exp_dir = os.path.join(base_dir, f"exp{exp_num}" if exp_num > 0 else "exp")
        if not os.path.exists(exp_dir):
            return exp_dir
        exp_num += 1


def is_video_file(filepath):
    """
    判断文件是否为视频文件
    
    Args:
        filepath: 文件路径
    
    Returns:
        True 如果是视频文件，否则 False
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')
    return filepath.lower().endswith(video_extensions)


def is_image_file(filepath):
    """
    判断文件是否为图片文件
    
    Args:
        filepath: 文件路径
    
    Returns:
        True 如果是图片文件，否则 False
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return filepath.lower().endswith(image_extensions)


def is_grayscale(image_path, saturation_threshold=10, variance_threshold=10):
    """
    判断图像是否为灰度图（使用HSV饱和度分析和通道方差分析）
    
    Args:
        image_path: 图像文件路径
        saturation_threshold: 饱和度阈值（0-255），低于此值认为是灰度图
        variance_threshold: 通道方差阈值，低于此值认为三个通道差异很小
    
    Returns:
        bool: True表示灰度图（需要上色），False表示彩色图
    """
    import numpy as np
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # 如果是单通道图像，直接认为是灰度图
        if len(img.shape) == 2:
            return True
        
        # 如果通道数不是3，无法判断，返回False
        if img.shape[2] != 3:
            return False
        
        # 方法1: 使用HSV颜色空间分析饱和度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # 计算平均饱和度
        mean_saturation = np.mean(saturation)
        
        # 方法2: 计算RGB通道之间的差异（方差）
        b, g, r = cv2.split(img)
        diff_br = np.mean(np.abs(b - r))  # B-R通道差异
        diff_bg = np.mean(np.abs(b - g))  # B-G通道差异
        diff_gr = np.mean(np.abs(g - r))  # G-R通道差异
        avg_diff = (diff_br + diff_bg + diff_gr) / 3
        
        # 方法3: 计算每个像素的颜色方差
        pixel_var = np.var(img, axis=2).mean()
        
        # 判断逻辑：满足以下任一条件认为是灰度图
        is_gray = False
        
        # 条件1: 平均饱和度低于阈值
        if mean_saturation < saturation_threshold:
            is_gray = True
        
        # 条件2: 通道间平均差异很小
        if avg_diff < variance_threshold:
            is_gray = True
        
        # 条件3: 像素颜色方差很小（所有通道值接近）
        if pixel_var < variance_threshold:
            is_gray = True
        
        return is_gray
        
    except Exception as e:
        print(f"判断图像类型时出错: {e}")
        return False


def get_files_by_extension(directory, extensions):
    """
    获取目录中指定扩展名的所有文件
    
    Args:
        directory: 目录路径
        extensions: 扩展名元组，如 ('.jpg', '.png')
    
    Returns:
        符合条件的文件路径列表（排序后）
    """
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and f.lower().endswith(extensions)
    ]
    return sorted(files)


def save_detection_results(result, save_dir, filename, save_json=False, save_annotated=True):
    """
    保存检测结果到文件
    
    Args:
        result: YOLO检测结果对象
        save_dir: 保存目录
        filename: 文件名
        save_json: 是否保存单个json标注文件（汇总json始终保存）
        save_annotated: 是否保存带检测框的图片，False则保存原图
    
    Returns:
        dict: 检测结果数据（用于汇总），如果没有检测结果返回None
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图片结果
    if save_annotated:
        # 使用YOLO自带的保存方法，保持颜色正确
        result.save(os.path.join(save_dir, filename))
    else:
        # 直接保存原始图像，避免颜色空间转换问题
        # YOLO的orig_img是RGB格式，使用PIL保存更可靠
        from PIL import Image
        img = Image.fromarray(result.orig_img)
        img.save(os.path.join(save_dir, filename))
    
    # 准备检测结果数据
    data = {
        "image_filename": filename,
        "detection_count": 0,
        "detections": []
    }
    
    # 保存单个json标注（仅当save_json=True时）
    if save_json and result.boxes is not None and len(result.boxes) > 0:
        json_path = os.path.join(save_dir, os.path.splitext(filename)[0] + '.json')
        detections = []
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xywh = box.xywh[0].tolist()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls,
                "class_name": result.names[cls] if hasattr(result, 'names') else "unknown",
                "confidence": conf,
                "bbox_xywh": xywh,
                "bbox_xyxy": [x1, y1, x2, y2]
            })
        
        data["detection_count"] = len(detections)
        data["detections"] = detections
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif result.boxes is not None and len(result.boxes) > 0:
        # 不保存单个json，但仍收集检测数据用于汇总
        detections = []
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xywh = box.xywh[0].tolist()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls,
                "class_name": result.names[cls] if hasattr(result, 'names') else "unknown",
                "confidence": conf,
                "bbox_xywh": xywh,
                "bbox_xyxy": [x1, y1, x2, y2]
            })
        
        data["detection_count"] = len(detections)
        data["detections"] = detections
    
    return data


def validate_parameters(params):
    """
    校验参数是否在有效范围内
    
    Args:
        params: 参数字典，包含需要校验的参数
    
    Returns:
        错误信息列表（如果没有错误则为空列表）
    """
    errors = []
    
    if 'crop_ratio' in params and not (0 <= params['crop_ratio'] < 1):
        errors.append(f"裁剪比例 crop_ratio 必须在 [0, 1) 范围内，当前值: {params['crop_ratio']}")
    
    if 'quality' in params and not (1 <= params['quality'] <= 100):
        errors.append(f"压缩质量 quality 必须在 [1, 100] 范围内，当前值: {params['quality']}")
    
    if 'distance_threshold' in params and params['distance_threshold'] < 0:
        errors.append(f"距离阈值 distance_threshold 必须大于等于0，当前值: {params['distance_threshold']}")
    
    if 'stable_duration' in params and params['stable_duration'] <= 0:
        errors.append(f"稳定时长 stable_duration 必须大于0，当前值: {params['stable_duration']}")
    
    if 'conf' in params and not (0 <= params['conf'] <= 1):
        errors.append(f"置信度 conf 必须在 [0, 1] 范围内，当前值: {params['conf']}")
    
    return errors