import argparse
import os
import cv2
import numpy as np

from utils import get_next_exp_dir, collect_source_items, imread_unicode
from utils.cv import build_book_mask


def order_corners(pts):
    """
    将角点排序为 左上、右上、右下、左下。
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    diff = diff[:, 0]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_book_corners(gray, min_ratio=0.15, epsilon_ratio=0.02, **mask_kwargs):
    """
    检测书籍外轮廓角点。

    优先从实心掩码求凸包并简化为四边形得到 4 角；
    若不足 4 角，回退为外接框的 左上/右下 两角。

    Args:
        gray: 灰度图
        min_ratio: 掩码最小面积占比，不足则视为检测失败
        epsilon_ratio: 凸包多边形逼近精度比例（相对周长）

    Returns:
        (corners, n_corners): corners 为 n 个角点(Nx2)；
                              失败返回 (None, 0)
    """
    h_img, w_img = gray.shape
    mask = build_book_mask(gray, **mask_kwargs)
    if mask is None:
        return None, 0

    mask_area = float(cv2.countNonZero(mask))
    if mask_area < w_img * h_img * min_ratio:
        return None, 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_pts = np.vstack(contours)
    hull = cv2.convexHull(all_pts)

    peri = cv2.arcLength(hull, True)

    # 多档 epsilon 搜索，取最贴合的四边形角点（面积最大者）
    epsilons = [epsilon_ratio * m for m in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)]
    best_quad = None
    best_area = 0.0
    for eps in epsilons:
        approx = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2)
        if len(approx) != 4:
            continue
        area = abs(cv2.contourArea(approx))
        if area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is not None:
        return order_corners(best_quad), 4

    # 回退：不足四角，取外接框的两对角（左上、右下）
    x, y, w, h = cv2.boundingRect(hull)
    corners = np.array([[x, y], [x + w, y + h]], dtype=np.float32)
    return corners, 2


def rectify_quad(image, quad, margin=0):
    """
    将四边形书页透视校正为平整矩形。

    Args:
        image: BGR 图像
        quad: 4 个角点(Nx2, 已按 tl,tr,br,bl 有序)
        margin: 目标矩形四周外扩像素

    Returns:
        校正后的平整书页图
    """
    tl, tr, br, bl = quad
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [margin, margin],
        [max_width - 1 + margin, margin],
        [max_width - 1 + margin, max_height - 1 + margin],
        [margin, max_height - 1 + margin],
    ], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    out_w = max_width + 2 * margin
    out_h = max_height + 2 * margin
    return cv2.warpPerspective(image, m, (out_w, out_h))


def crop_book_image(image, bbox, margin=0):
    """
    按轴对齐包围框裁剪（用于两角回退等场景）。
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    return image[y1:y2, x1:x2]


def process_single(image_path, output_dir, rectify=True, margin=None,
                   min_ratio=0.15, debug=False):
    """
    处理单张图片：按书籍角点裁剪并保存。

    Args:
        rectify: True 且四角齐备时透视校正为平整书页；否则轴对齐裁剪
        margin: 外扩边距像素，None 则按图像宽度 2.5% 自适应
        debug: True 时保存角点叠加图

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
    corners, n_corners = detect_book_corners(gray, min_ratio=min_ratio)
    if corners is None:
        print(f"  ✗ {os.path.basename(image_path)} - 未检测到书籍区域")
        return False

    name = os.path.splitext(os.path.basename(image_path))[0]

    if debug:
        vis = image.copy()
        for i, (cx, cy) in enumerate(corners.astype(int)):
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(vis, str(i), (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{name}_debug_corners.jpg"), vis)

    if n_corners == 4 and rectify:
        cropped = rectify_quad(image, corners, margin=margin)
        mode = "透视校正"
    else:
        x, y, w, h = cv2.boundingRect(corners)
        cropped = crop_book_image(image, (x, y, w, h), margin=margin)
        mode = f"轴对齐({n_corners}角)"

    out_path = os.path.join(output_dir, f"{name}_corner.jpg")
    cv2.imwrite(out_path, cropped)
    print(f"  ✓ {os.path.basename(image_path)} [{mode}] -> {os.path.basename(out_path)} "
          f"({cropped.shape[1]}x{cropped.shape[0]})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='图片裁剪优化：按书籍四角（不足则两角）裁剪，可透视校正')
    parser.add_argument('--input', default=r'E:\储藏室\画册\扫描\testbook',
                        help='单张图片路径或一个目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，默认 runs/crop_book_corner（自动递增防覆盖）')
    parser.add_argument('--axis', default=False,
                        help='关闭透视校正，仅按角点外接框轴对齐裁剪')
    parser.add_argument('--margin', type=int, default=None,
                        help='外扩边距像素；不传则按图像宽度2.5%%自适应')
    parser.add_argument('--min-ratio', type=float, default=0.15,
                        help='掩码相对图像的最小面积占比，不足则视为误检')
    parser.add_argument('--epsilon', type=float, default=0.02,
                        help='凸包多边形逼近精度比例，用于提取四角')
    parser.add_argument('--debug', default=True,
                        help='保存角点叠加图便于校准')
    args = parser.parse_args()

    if args.output is None:
        output_dir = get_next_exp_dir('runs/crop_book_corner')
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
    ok = sum(process_single(s, output_dir, rectify=not args.axis,
                            margin=args.margin, min_ratio=args.min_ratio,
                            debug=args.debug) for s, _ in sources)
    print(f"\n[DONE] 成功 {ok}/{len(sources)} 张，输出目录: {output_dir}")


if __name__ == '__main__':
    main()