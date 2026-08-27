import argparse
import os
import shutil
from detect import run_detection
from utils import ComfyUIClient, load_yolo_model
from utils.config import DEFAULT_ALBUM_SOURCE

'''
这个脚本用于检测漫画输入目录（文件夹1）中的图片，
并使用ComfyUI去码处理。
检测结果保存在detect-save-dir目录（文件夹2）中，
ComfyUI处理保存在comfyui-save-dir目录（文件夹3）中。
json文件保存在save-json目录（文件夹4）中。
'''




def main():
    parser = argparse.ArgumentParser(description='检测 + ComfyUI 处理流程')
    
    # Detect 参数
    parser.add_argument('--model', type=str, default=r'weights\censor_detect_v1.0_s_0725.pt', 
                        help='模型权重文件路径')
    parser.add_argument('--source', type=str, default=DEFAULT_ALBUM_SOURCE, 
                        help='检测输入目录（文件夹1）')
    parser.add_argument('--conf', type=float, default=0.6, 
                        help='检测置信度阈值')
    parser.add_argument('--detect-save-dir', type=str, default=r'runs\detections', 
                        help='检测结果保存目录（文件夹2）')
    parser.add_argument('--save-json', action='store_true', default=False, 
                        help='保存单个检测json文件（汇总json始终保存）')
    
    # ComfyUI 参数
    parser.add_argument('--workflow', type=str, default=r'workflows\f2k-漫画去码-py.json', 
                        help='ComfyUI工作流JSON路径（工作流1）')
    parser.add_argument('--comfyui-save-dir', type=str, default=r'runs\comfyui_output\[Cuvie] Bitter Addiction [DL版][机翻][去码]', 
                        help='ComfyUI结果保存目录（文件夹3）')
    parser.add_argument('--comfyui-server', type=str, default='http://127.0.0.1:8188', 
                        help='ComfyUI服务器地址')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("检测 + ComfyUI 处理流程")
    print("=" * 60)
    
    # 步骤1: 调用 detect.py 进行推理（保存原图，不带检测框）
    print("\n[步骤1] 运行 YOLO 检测...")
    print(f"  输入目录: {args.source}")
    print(f"  保存原图（不带检测框）")
    
    model, _ = load_yolo_model(args.model)
    if model is None:
        print("\n错误：模型加载失败")
        return
    
    detect_output_dir, results = run_detection(model, args.model, args.source, args.conf, args.detect_save_dir, args.save_json, save_annotated=False)
    if detect_output_dir is None:
        print("\n错误：检测未完成，未生成输出目录")
        return
    print(f"  输出目录: {detect_output_dir}")
    
    # 步骤2: 直接用 detect.py 返回值判断哪些图片有检测结果（不再重读 summary.json）
    images_with_detection = []
    images_without_detection = []
    
    for result in results:
        img_path = os.path.join(detect_output_dir, result["image_filename"])
        if result["detection_count"] > 0:
            images_with_detection.append(img_path)
        else:
            images_without_detection.append(img_path)
    
    print(f"\n检测完成")
    print(f"  有检测目标的图像: {len(images_with_detection)} 张")
    print(f"  无检测目标的图像: {len(images_without_detection)} 张")
    
    # 步骤3: 创建文件夹3
    os.makedirs(args.comfyui_save_dir, exist_ok=True)
    
    # 步骤4: 无检测目标的图像直接复制到文件夹3
    if images_without_detection:
        print(f"\n[步骤2] 复制无检测目标的图像到文件夹3...")
        for img_path in images_without_detection:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(args.comfyui_save_dir, filename)
            shutil.copy2(img_path, dest_path)
            print(f"  复制: {filename}")
    
    # 步骤5: 有检测目标的图像导入ComfyUI处理
    if images_with_detection:
        print(f"\n[步骤3] 运行 ComfyUI 处理有检测目标的图像...")
        print(f"  工作流: {args.workflow}")
        print(f"  输出目录: {args.comfyui_save_dir}")
        
        comfy_client = ComfyUIClient(args.comfyui_server)
        
        if not os.path.exists(args.workflow):
            print(f"错误：工作流文件不存在: {args.workflow}")
            return
        
        print(f"批量处理 {len(images_with_detection)} 张图像...")
        saved_paths = comfy_client.process_batch_images(images_with_detection, args.workflow, args.comfyui_save_dir)
        print(f"  ComfyUI处理完成，保存了 {len(saved_paths)} 张图像")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"  检测结果: {detect_output_dir}")
    print(f"  ComfyUI结果: {args.comfyui_save_dir}")
    print(f"  有检测目标: {len(images_with_detection)} 张（已通过ComfyUI处理）")
    print(f"  无检测目标: {len(images_without_detection)} 张（直接复制）")
    print("=" * 60)

if __name__ == '__main__':
    main()