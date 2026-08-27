import cv2
import numpy as np


def extract_content_contours(gray, blur_kernel=5, canny_low=None, canny_high=None,
                             contour_min_ratio=0.01, close_kernel=15):
    """
    提取画面内"书籍内容"轮廓，供书籍掩码/包围框复用。

    展开的书在 Canny 下通常被拆成左右页两块内容区，且可能与桌面反光/
    边框噪声粘连。这里做高斯降噪 → 自适应 Canny → 闭运算连成实体块，
    再过滤掉"同时贴左右两边"或"同时贴上下两边"的横幅/竖幅反光噪声；
    单边贴边（如书页延伸到照片边缘）属于合法内容，予以保留。

    Args:
        gray: 灰度图
        blur_kernel: 高斯模糊核大小（应为奇数）
        canny_low: Canny 低阈值，None 则基于中值自适应
        canny_high: Canny 高阈值，None 则基于中值自适应
        contour_min_ratio: 单轮廓相对图像的最小面积占比，过滤小噪声
        close_kernel: 形态学闭运算核大小（应为奇数）

    Returns:
        list: 保留的轮廓列表；无内容返回空列表
    """
    h_img, w_img = gray.shape
    contour_min_area = float(w_img * h_img) * contour_min_ratio

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
        return []

    margin_px = 2
    kept = []
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
        kept.append(cnt)

    return kept


def build_book_mask(gray, **kwargs):
    """
    由内容轮廓填充为二值掩码（0/255）；无内容返回 None。
    """
    contours = extract_content_contours(gray, **kwargs)
    if not contours:
        return None
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask


def book_contour_bbox(gray, min_ratio=0.15, **kwargs):
    """
    由内容轮廓包围框并集得书籍包围框 (x, y, w, h)；失败返回 None。

    Args:
        gray: 灰度图
        min_ratio: 最终包围框相对图像的最小面积占比，低于则视为检测失败
        **kwargs: 透传给 extract_content_contours 的参数
    """
    contours = extract_content_contours(gray, **kwargs)
    if not contours:
        return None

    h_img, w_img = gray.shape
    boxes = [cv2.boundingRect(c) for c in contours]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1

    if float(w * h) / (float(w_img) * h_img) < min_ratio:
        return None
    return x1, y1, w, h