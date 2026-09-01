import cv2
import numpy as np


def resize_max_edge(img, max_edge):
    """
    按最长边等比缩放图像，使最长边不超过 max_edge 像素。
    保持长宽比；max_edge 为空或原图最长边已小于等于阈值时原样返回。

    Args:
        img: BGR 图像
        max_edge: 最长边目标像素；None 表示不缩放

    Returns:
        缩放后的图像（可能为原图对象）
    """
    if img is None or not max_edge:
        return img
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    return cv2.resize(
        img, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def frame_mad(frame_a, frame_b, max_edge=96):
    """
    计算两帧的平均绝对差（0-255），用于画面运动量检测。

    先将两帧等比缩到小尺寸并灰度化再做 absdiff 求均值，开销极小，
    适合每帧调用。任一帧为空时返回正无穷（视为变化剧烈）。

    Args:
        frame_a: BGR 帧 A
        frame_b: BGR 帧 B
        max_edge: 比较用图像的最长边像素，越小越快

    Returns:
        float: 平均绝对差；越大表示运动越明显
    """
    if frame_a is None or frame_b is None:
        return float('inf')

    def prep(img):
        h, w = img.shape[:2]
        if max_edge and max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)

    return float(np.mean(np.abs(prep(frame_a) - prep(frame_b))))


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


def descreen_moire(image, peak_frac=0.25, exclusion_frac=0.05, notch_radius=4, min_peak_dist=10):
    """
    去除扫描漫画条纹（摩尔纹/网纹）。

    印刷网点是规则周期结构，映射到频域是分布在直流(中心)外围的高幅值亮点。
    流程：Hanning 窗抑制边缘伪峰 → FFT → 在中心环带外检测局部极大值峰 →
    峰及其镜像对称点放置小半径陷波置零 → 逆 FFT 还原。
    彩色图仅在 YCrCb 的 Y 通道处理，保留颜色、避免色偏；灰度图直接处理。

    Args:
        image: BGR 或灰度图
        peak_frac: 相对环带最大幅值的峰阈值比例，越低越激进
        exclusion_frac: 中心直流排除半径占最短边比例，防误伤低频主体
        notch_radius: 单个陷波半径（像素），过大伤细节
        min_peak_dist: 相邻峰最小间距，防止同一条纹带重复陷波

    Returns:
        去纹后的图像（BGR/灰度与原输入一致）；未检测到条纹时原样返回
    """
    if not image.ndim in (2, 3):
        raise ValueError("descreen_moire 仅支持灰度(2D)或 BGR(3D) 图像")

    bgr_input = image.ndim == 3
    if bgr_input:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        gray = ycrcb[:, :, 0].copy()
    else:
        gray = image.copy()

    h0, w0 = gray.shape
    h, w = (h0 + 1) & ~1, (w0 + 1) & ~1
    g = cv2.copyMakeBorder(gray, 0, h - h0, 0, w - w0,
                           cv2.BORDER_REFLECT).astype(np.float32)
    win = np.outer(np.hanning(h), np.hanning(w))
    gw = g * win

    F = np.fft.fftshift(np.fft.fft2(gw))
    mag = np.abs(F)

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = min(h, w) * exclusion_frac
    excluded = np.hypot(yy - cy, xx - cx) < radius
    ring = mag.copy()
    ring[excluded] = 0.0

    if ring.max() <= 0:
        return image

    dil = cv2.dilate(ring, np.ones((3, 3), np.uint8))
    thr = peak_frac * ring.max()
    cand = np.argwhere((ring >= dil) & (ring >= thr))
    if cand.size == 0:
        return image
    order = np.argsort(-ring[tuple(cand.T)])

    peaks = []
    for i in order:
        y, x = cand[i]
        if all(np.hypot(y - py, x - px) >= min_peak_dist for py, px in peaks):
            peaks.append((int(y), int(x)))
    if not peaks:
        return image

    mask = np.ones((h, w), np.float32)
    Yy, Xx = np.ogrid[:h, :w]
    for y, x in peaks:
        mask[np.square(Yy - y) + np.square(Xx - x) <= notch_radius ** 2] = 0.0
        my, mx = h - 1 - y, w - 1 - x
        mask[np.square(Yy - my) + np.square(Xx - mx) <= notch_radius ** 2] = 0.0

    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
    win_safe = np.where(win < 1e-3, 1.0, win)
    out = np.clip(out / win_safe, 0, 255).astype(np.uint8)[:h0, :w0]

    if bgr_input:
        ycrcb[:, :, 0] = out
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return out