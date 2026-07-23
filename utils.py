import os


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


def save_detection_results(result, save_dir, filename, save_txt=False):
    """
    保存检测结果到文件
    
    Args:
        result: YOLO检测结果对象
        save_dir: 保存目录
        filename: 文件名
        save_txt: 是否保存txt标注文件
    
    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图片结果
    result.save(os.path.join(save_dir, filename))
    
    # 保存txt标注
    if save_txt and result.boxes is not None:
        txt_path = os.path.join(save_dir, os.path.splitext(filename)[0] + '.txt')
        with open(txt_path, 'w') as f:
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xywh = box.xywh[0].tolist()
                line = f"{cls} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]} {conf}\n"
                f.write(line)


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