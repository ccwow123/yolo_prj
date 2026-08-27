import argparse
import os
import cv2

from utils import get_next_exp_dir, collect_source_items, imread_unicode
from utils.cv import book_contour_bbox


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
    image = imread_unicode(image_path)
    if image is None:
        print(f"  ✗ {os.path.basename(image_path)} - 无法读取图像")
        return False

    if margin is None:
        margin = int(image.shape[1] * 0.025)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bbox = book_contour_bbox(gray, min_ratio=min_ratio)
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

    sources, error = collect_source_items(args.input, image_only=True)
    if error:
        print(f"[ERROR] {error}")
        return

    if not sources:
        print(f"[WARN] 未找到可处理的图片: {args.input}")
        return

    print(f"[INFO] 待处理图片: {len(sources)} 张")
    ok = sum(process_single(s, output_dir, args.margin, args.min_ratio, args.debug) for s, _ in sources)
    print(f"\n[DONE] 成功 {ok}/{len(sources)} 张，输出目录: {output_dir}")


if __name__ == '__main__':
    main()