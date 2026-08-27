import argparse
import os
import cv2
import numpy as np

from utils import get_next_exp_dir, is_image_file


def build_content_mask(gray, blur_kernel=5, canny_low=None, canny_high=None,
                       contour_min_ratio=0.01, close_kernel=15):
    """
    提取画面内书籍内容实心掩码（排除跨画面横幅/竖幅反光噪声）。
    """
    h_img, w_img = gray.shape
    image_area = float(w_img * h_img)
    contour_min_area = image_area * contour_min_ratio

    if blur_kernel > 0 and blur_kernel % 2 == 1:
        gray_b = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        gray_b = gray

    if canny_low is None or canny_high is None:
        median = int(np.median(gray_b))
        canny_low = max(0, int(0.66 * median))
        canny_high = min(255, int(1.33 * median))

    edges = cv2.Canny(gray_b, canny_low, canny_high)

    if close_kernel > 0 and close_kernel % 2 == 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    else:
        closed = edges

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    mask = np.zeros(gray.shape, dtype=np.uint8)
    margin_px = 2
    kept = False
    for cnt in contours:
        if cv2.contourArea(cnt) < contour_min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        left = x <= margin_px
        right = x + w >= w_img - margin_px
        top = y <= margin_px
        bottom = y + h >= h_img - margin_px
        if (left and right) or (top and bottom):
            continue
        kept = True
        cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask if kept else None


def find_edges(mask, min_support=0.5, band_ratio=0.24):
    """
    逆时针逐边寻找书页外边缘：顶、右、底、左。

    对每条边，从内容掩码在该方向的最外轮廓点取样，用覆盖度判断边的可信，
    用分位数（顶/左下分位、底/右上分位）定该边位置，规避双页内容顶部参差。

    Args:
        mask: 内容掩码（0/255）
        min_support: 该边有效取样点的最小覆盖度，低于则视为不可信（缺失）
        band_ratio: 取样时剔除四角的横向/纵向比例

    Returns:
        dict: {"top": y, "right": x, "bottom": y, "left": x}，缺失的边不含键
    """
    h_img, w_img = mask.shape

    def col_min_max(col):
        col_data = mask[:, col]
        nz = np.nonzero(col_data)[0]
        if nz.size == 0:
            return None, None
        return nz.min(), nz.max()

    def row_min_max(row):
        row_data = mask[row, :]
        nz = np.nonzero(row_data)[0]
        if nz.size == 0:
            return None, None
        return nz.min(), nz.max()

    band_w = int(w_img * band_ratio)
    band_h = int(h_img * band_ratio)

    edges_found = {}

    # 顶边：中间列带的每列最上行，取下分位
    cols = list(range(band_w, w_img - band_w))
    tops = []
    for c in cols:
        t, _ = col_min_max(c)
        if t is not None:
            tops.append(t)
    if len(tops) / len(cols) >= min_support and tops:
        edges_found["top"] = int(np.percentile(tops, 10))

    # 右边：中间行带的每行最右列，取上分位
    rows = list(range(band_h, h_img - band_h))
    rights = []
    for r in rows:
        _, rr = row_min_max(r)
        if rr is not None:
            rights.append(rr)
    if len(rights) / len(rows) >= min_support and rights:
        edges_found["right"] = int(np.percentile(rights, 90))

    # 底边：中间列带的每列最下行，取上分位
    bottoms = []
    for c in cols:
        _, b = col_min_max(c)
        if b is not None:
            bottoms.append(b)
    if len(bottoms) / len(cols) >= min_support and bottoms:
        edges_found["bottom"] = int(np.percentile(bottoms, 90))

    # 左边：中间行带的每行最左列，取下分位
    lefts = []
    for r in rows:
        l, _ = row_min_max(r)
        if l is not None:
            lefts.append(l)
    if len(lefts) / len(rows) >= min_support and lefts:
        edges_found["left"] = int(np.percentile(lefts, 10))

    return edges_found


def form_crop_box(edges_found, image_shape, margin=0):
    """
    由找到的边生成轴对齐裁剪框；缺失边以图像边界兜底并外扩 margin。
    """
    h_img, w_img = image_shape[:2]

    x1 = 0 if "left" not in edges_found else max(0, edges_found["left"] - margin)
    y1 = 0 if "top" not in edges_found else max(0, edges_found["top"] - margin)
    x2 = w_img if "right" not in edges_found else min(w_img, edges_found["right"] + margin)
    y2 = h_img if "bottom" not in edges_found else min(h_img, edges_found["bottom"] + margin)

    return x1, y1, x2, y2


def process_single(image_path, output_dir, margin=0, min_support=0.5, debug=False):
    """
    处理单张图片：按书页外边缘裁剪；找不到任何边则保留原图。

    Returns:
        bool: True 表示处理成功（含保留原图的情况）
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ✗ {os.path.basename(image_path)} - 无法读取图像")
        return False

    name = os.path.splitext(os.path.basename(image_path))[0]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = build_content_mask(gray)
    n_sides = 0
    out_path = os.path.join(output_dir, f"{name}_edge.jpg")

    if mask is None:
        # 兜底：无法提取掩码视为没找到边，保留原图
        cv2.imwrite(out_path, image)
        print(f"  ○ {os.path.basename(image_path)} 掩码为空，保留原图")
        return True

    edges_found = find_edges(mask, min_support=min_support)
    n_sides = len(edges_found)

    if debug:
        vis = image.copy()
        if "top" in edges_found:
            cv2.line(vis, (0, edges_found["top"]), (image.shape[1], edges_found["top"]), (0, 0, 255), 2)
        if "bottom" in edges_found:
            cv2.line(vis, (0, edges_found["bottom"]), (image.shape[1], edges_found["bottom"]), (0, 0, 255), 2)
        if "left" in edges_found:
            cv2.line(vis, (edges_found["left"], 0), (edges_found["left"], image.shape[0]), (0, 0, 255), 2)
        if "right" in edges_found:
            cv2.line(vis, (edges_found["right"], 0), (edges_found["right"], image.shape[0]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{name}_debug_edges.jpg"), vis)

    if n_sides == 0:
        # 一条边都找不到：保留原图
        cv2.imwrite(out_path, image)
        print(f"  ○ {os.path.basename(image_path)} 未找到边(共{n_sides}条)，保留原图")
        return True

    x1, y1, x2, y2 = form_crop_box(edges_found, image.shape, margin=margin)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        cv2.imwrite(out_path, image)
        print(f"  ○ {os.path.basename(image_path)} 裁剪框无效，保留原图")
        return True
    cv2.imwrite(out_path, cropped)
    sides_label = "、".join(edges_found.keys())
    print(f"  ✓ {os.path.basename(image_path)} [边:{sides_label}] -> {os.path.basename(out_path)} "
          f"({cropped.shape[1]}x{cropped.shape[0]})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='图片裁剪优化：逆时针逐边找书页外边缘并裁剪（找不到边则保留原图）')
    parser.add_argument('--input', default=r'E:\储藏室\画册\扫描\testbook',
                        help='单张图片路径或一个目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，默认 runs/crop_book_edge（自动递增防覆盖）')
    parser.add_argument('--margin', type=int, default=0,
                        help='从找到的边向外扩的像素，缺失边以图像边界兜底')
    parser.add_argument('--min-support', type=float, default=0.5,
                        help='单边有效取样点最小覆盖度(0-1)，低于视为该边缺失')
    parser.add_argument('--debug', default=True,
                        help='保存边线叠加图便于校准')
    args = parser.parse_args()

    if args.output is None:
        output_dir = get_next_exp_dir('runs/crop_book_edge')
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
    ok = sum(process_single(s, output_dir, margin=args.margin,
                            min_support=args.min_support, debug=args.debug) for s in sources)
    print(f"\n[DONE] 成功 {ok}/{len(sources)} 张，输出目录: {output_dir}")


if __name__ == '__main__':
    main()