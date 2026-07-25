import argparse
import os
import shutil
from utils import ComfyUIClient, is_image_file, is_grayscale


def main():
    parser = argparse.ArgumentParser(description='图像上色处理流程')
    
    # 输入输出参数
    parser.add_argument('--source', type=str, default=r'imgs', 
                        help='输入目录（文件夹1）')
    parser.add_argument('--output', type=str, default=r'runs\colorize_output', 
                        help='输出目录（文件夹2）')
    
    # ComfyUI 参数
    parser.add_argument('--workflow', type=str, default=r'workflows\anima漫画上色-py.json', 
                        help='ComfyUI上色工作流JSON路径')
    parser.add_argument('--comfyui-server', type=str, default='http://127.0.0.1:8188', 
                        help='ComfyUI服务器地址')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("图像上色处理流程")
    print("=" * 60)
    
    # 验证输入目录
    if not os.path.exists(args.source):
        print(f"错误：输入目录不存在: {args.source}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = [
        os.path.join(args.source, f)
        for f in os.listdir(args.source)
        if os.path.isfile(os.path.join(args.source, f)) and is_image_file(f)
    ]
    
    if not image_paths:
        print(f"错误：输入目录中没有有效的图片文件")
        return
    
    print(f"\n共发现 {len(image_paths)} 张图片")
    
    # 分类图像
    grayscale_images = []
    color_images = []
    
    print("\n[步骤1] 分析图像类型...")
    for img_path in image_paths:
        if is_grayscale(img_path):
            grayscale_images.append(img_path)
            print(f"  灰度图: {os.path.basename(img_path)}")
        else:
            color_images.append(img_path)
            print(f"  彩色图: {os.path.basename(img_path)}")
    
    print(f"\n分析完成")
    print(f"  灰度图（需要上色）: {len(grayscale_images)} 张")
    print(f"  彩色图（直接复制）: {len(color_images)} 张")
    
    # 步骤2: 彩色图直接复制到输出目录
    if color_images:
        print(f"\n[步骤2] 复制彩色图像到输出目录...")
        for img_path in color_images:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(args.output, filename)
            shutil.copy2(img_path, dest_path)
            print(f"  复制: {filename}")
    
    # 步骤3: 灰度图通过ComfyUI上色
    if grayscale_images:
        print(f"\n[步骤3] 运行 ComfyUI 上色处理...")
        print(f"  工作流: {args.workflow}")
        print(f"  输出目录: {args.output}")
        
        # 验证工作流文件
        if not os.path.exists(args.workflow):
            print(f"错误：工作流文件不存在: {args.workflow}")
            return
        
        comfy_client = ComfyUIClient(args.comfyui_server)
        
        print(f"批量处理 {len(grayscale_images)} 张灰度图...")
        saved_paths = comfy_client.process_batch_images(grayscale_images, args.workflow, args.output)
        print(f"  ComfyUI上色完成，保存了 {len(saved_paths)} 张图像")
    
    # 总结
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"  输出目录: {args.output}")
    print(f"  灰度图（已上色）: {len(grayscale_images)} 张")
    print(f"  彩色图（直接复制）: {len(color_images)} 张")
    print("=" * 60)


if __name__ == '__main__':
    main()