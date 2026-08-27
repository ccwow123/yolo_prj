import argparse
import os
import cv2
import numpy as np

from utils import get_next_exp_dir, is_image_file


def detect_book_bbox(image_gray, min_ratio=0.15, blur_kernel=5,
                     canny_low=None, canny_high=None, contour_min_ratio=0.01,
                     close_kernel=15):
    """
    检测张开书籍的区域（画面内最大内容区并集）。

    展开的书在 Canny 下通常被拆成左右页两块内容区，且可能与桌面反光/
    边框噪声粘连。这里取"不触贴图像边界、面积达标的轮廓"包围框并集，
    即可覆盖整本书的展开区间。

    Args:
        image_gray: 灰度图
        min_ratio: 最终包围框相对图像的最小面积占比，低于则视为检测失败
        blur_kernel: 高斯模糊核大小（应为奇数）
        canny_low: Canny 低阈值，None 则基于中值自适应
        canny_high: Canny 高阈值，None 则基于中值自适应
        contour_min_ratio: 单个轮廓相对图像的最小面积占比，过滤小噪声
        close_kernel: 形态学闭运算核大小（应为奇数）

    Returns:
        (x, y, w, h) 包围框；失败返回 None
    """
    h_img, w_img = image_gray.shape
    image_area = float(w_img * h_img)
    contour_min_area = image_area * contour_min_ratio

    # 高斯降噪
    if blur_kernel > 0 and blur_kernel % 2 == 1:
        gray = cv2.GaussianBlur(image_gray, (blur_kernel, blur_kernel), 0)
    else:
        gray = image_gray

    # Canny 阈值自适应
    if canny_low is None or canny_high is None:
        median = int(np.median(gray))
        low = max(0, int(0.66 * median))
        high = min(255, int(1.33 * median))
        canny_low = low if canny_low is None else canny_low
        canny_high = high if canny_high is None else canny_high

    edges = cv2.Canny(gray, canny_low, canny_high)

    # 闭运算连成实体内容块
    if close_kernel > 0 and close_kernel % 2 == 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    else:
        closed = edges

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 取画面内、面积达标的轮廓包围框并集
    # 仅排除"同时贴左右两边"或"同时贴上下两边"的横幅/竖幅反光噪声；
    # 单边贴边（如书页延伸到照片边缘）属于合法内容，予以保留
    margin_px = 2
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < contour_min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        left = x <= margin_px
        right = x + w >= w_img - margin_px
        top = y <= margin_px
        bottom = y + h >= h_img - margin_px
        if (left and right) or (top and bottom):
            continue  # 跨画面横幅/竖幅，多为桌面反光/边框噪声
        boxes.append((x, y, w, h))

    if not boxes:
        return None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1

    # 面积占比校验
    if float(w * h) / image_area < min_ratio:
        return None

    return x1, y1, w, h


def crop_book_image(image, bbox, margin=0):
    """
    按包围框轴对齐裁剪图片，并做边距外扩与边界钳制。

    Args:
        image: BGR 图像
        bbox: (x, y, w, h) 包围框
        margin: 外扩边距像素

    Returns:
        裁剪后的图像
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox

    # 外扩边距
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)

    return image[y1:y2, x1:x2]


def process_single(image_path, output_dir, margin=None, min_ratio=0.15, debug=False):
    """
    处理单张图片：检测书籍并裁剪保存。

    Args:
        margin: 外扩边距像素，None 则按图像宽度的 2.5% 自适应外扩（留出书页白边）
        debug: True 时额外保存一张叠加检测框的原图便于校准

    Returns:
        bool: True 表示成功
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ✗ {os.path.basename(image_path)} - 无法读取图像")
        return False

    if margin is None:
        margin = int(image.shape[1] * 0.025)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bbox = detect_book_bbox(gray, min_ratio=min_ratio)
    if bbox is None:
        print(f"  ✗ {os.path.basename(image_path)} - 未检测到书籍区域")
        return False

    if debug:
        vis = image.copy()
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        dbg_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{dbg_name}_debug_box.jpg"), vis)

    cropped = crop_book_image(image, bbox, margin=margin)

    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{name}_cropped.jpg")
    cv2.imwrite(out_path, cropped)
    print(f"  ✓ {os.path.basename(image_path)} -> {os.path.basename(out_path)} "
          f"({cropped.shape[1]}x{cropped.shape[0]})")
    return True


def main():
    parser = argparse.ArgumentParser(description='图片裁剪优化：识别张开的书籍并轴对齐裁剪输出')
    parser.add_argument('--input',default=r'E:\储藏室\画册\扫描\testbook',
                        help='单张图片路径或一个目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，默认 runs/crop_book（自动递增防覆盖）')
    parser.add_argument('--margin', type=int, default=None,
                        help='裁框外扩边距像素；不传则按图像宽度2.5%%自适应外扩留白边')
    parser.add_argument('--min-ratio', type=float, default=0.15,
                        help='检测框相对图像的最小面积占比，低于则视为误检')
    parser.add_argument('--debug', default=True,
                        help='保存叠加检测框的原图便于校准')
    args = parser.parse_args()

    if args.output is None:
        output_dir = get_next_exp_dir('runs/crop_book')
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {output_dir}")

    sources = []
    if os.path.isdir(args.input):
        sources = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.isfile(os.path.join(args.input, f)) and is_image_file(f)
        ]
    elif os.path.isfile(args.input):
        sources = [args.input]
    else:
        print(f"[ERROR] 未找到输入路径: {args.input}")
        return

    if not sources:
        print(f"[WARN] 未找到可处理的图片: {args.input}")
        return

    print(f"[INFO] 待处理图片: {len(sources)} 张")
    ok = sum(process_single(s, output_dir, args.margin, args.min_ratio, args.debug) for s in sources)
    print(f"\n[DONE] 成功 {ok}/{len(sources)} 张，输出目录: {output_dir}")


if __name__ == '__main__':
    main()