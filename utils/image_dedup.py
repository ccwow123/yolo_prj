#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_dedup.py — 可导出的图像查重函数库

导出:
    deduplicate_screenshotsV2(screenshots_dir, triggered_frames=None,
                            similarity_threshold=0.8, method="orb", ...)
        -> (filtered_frames, removed_frames)

参考 deduplicate_screenshotsV2 的思路, 但修复了两个坑:
  1. cv2.imread 在 Windows 上无法读取含中文/日文等非 ASCII 路径
     -> 改用 cv2.imdecode(np.fromfile(...))  (imread_unicode)
  2. ORB 距离度量: 用 Lowe 比值测试的入内点比例
     distance = 1 - inlier_ratio, 越小越相似 (与原函数 "<= threshold 视为相似" 一致)

依赖: opencv-python-headless + numpy (orb); Pillow (dhash)
"""

import os

import cv2
import numpy as np
from PIL import Image

DEFAULT_EXT = ".jpg"


# ------------------------- 基础工具 -------------------------
def imread_unicode(path):
    """兼容含非 ASCII 字符路径的读图 (cv2.imread 在 Windows 上对这类路径会静默返回 None)。
    文件不存在/无法解码时返回 None。"""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def orb_distance(img1, img2, nfeatures=1000):
    """两张图的归一化 ORB 距离, 约在 [0,1]。0=完全一致, 越大越不相似。
    基于 Lowe 比值测试的入内点比例: distance = 1 - inlier_ratio。
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 1.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    mm = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in mm if m.distance < 0.75 * n.distance]
    if not good:
        return 1.0
    ratio = len(good) / min(len(kp1), len(kp2))
    return 1.0 - ratio


def dhash(image, hash_size=8):
    """差异哈希 (difference hash), 返回 int 位向量。"""
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = image.load()
    value = 0
    bit = 0
    for row in range(hash_size):
        for col in range(hash_size):
            if px[col, row] > px[col + 1, row]:
                value |= 1 << bit
            bit += 1
    return value


# ------------------------- 帧号 / 文件解析 -------------------------
def _list_frames(screenshots_dir, ext):
    """扫描目录, 返回 [(帧号, 文件名), ...] 按帧号升序。"""
    items = []
    for name in os.listdir(screenshots_dir):
        if name.lower().endswith(ext.lower()):
            stem = name[: -len(ext)] if name.lower().endswith(ext.lower()) else name
            digits = "".join(ch for ch in stem if ch.isdigit())
            frame = int(digits) if digits else hash(name)
            items.append((frame, name))
    items.sort(key=lambda x: x[0])
    return items


def _resolve_path(screenshots_dir, frame, ext):
    """按 'screenshot_{frame:06d}{ext}' 拼出文件路径 (与触发截图命名一致)。"""
    return os.path.join(screenshots_dir, f"screenshot_{frame:06d}{ext}")


# ------------------------- 查重主函数 -------------------------
def deduplicate_screenshotsV2(screenshots_dir, triggered_frames=None,
                             similarity_threshold=0.8, method="orb",
                             ext=DEFAULT_EXT, nfeatures=1000, hash_size=8):
    """
    去除相似截图。

    Args:
        screenshots_dir:      截图目录
        triggered_frames:     帧号列表 (int); 为 None 时自动扫描目录下 '{prefix}{帧号:06d}{ext}'
        similarity_threshold: 相似度阈值, 越低越严格。
                               orb 默认 0.8; dhash 默认 15(汉明距离)
        method:               "orb" (ORB 特征匹配 + 相邻帧顺序比较) 或 "dhash" (全局分组)
        ext:                  文件扩展名, 默认 ".jpg"
        nfeatures:            ORB 关键点数量
        hash_size:            dHash 精度

    Returns:
        filtered_frames: 去重后保留的帧号列表
        removed_frames:  被移除(判定为重复)的帧号列表
    """
    if method not in ("orb", "dhash"):
        raise ValueError("method 必须是 'orb' 或 'dhash'")

    # 确定待处理帧序列
    if triggered_frames is None:
        items = _list_frames(screenshots_dir, ext)
    else:
        items = sorted((int(f), f"screenshot_{int(f):06d}{ext}") for f in triggered_frames)

    if len(items) <= 1:
        return [f for f, _ in items], []

    filtered, removed = [], []

    if method == "dhash":
        # 全局分组: 任意两张汉明距离 <= 阈值 归为一组, 保留每组文件最大的一张
        hashes = {}
        valid = []
        for frame, name in items:
            path = os.path.join(screenshots_dir, name)
            try:
                with Image.open(path) as im:
                    hashes[frame] = dhash(im, hash_size)
                valid.append((frame, name))
            except Exception:
                # 读图失败/文件不存在 -> 当作不同, 直接保留
                filtered.append(frame)
        reps = []  # (frame, hash)
        groups = []
        for frame, name in valid:
            h = hashes[frame]
            placed = False
            for gi, (rf, rh) in enumerate(reps):
                if (h ^ rh).bit_count() <= int(similarity_threshold):
                    groups[gi].append(frame)
                    placed = True
                    break
            if not placed:
                groups.append([frame])
                reps.append((frame, h))
        for gi, grp in enumerate(groups):
            keeper = max(grp, key=lambda f: os.path.getsize(_resolve_path(screenshots_dir, f, ext)))
            filtered.append(keeper)
            for f in grp:
                if f != keeper:
                    removed.append(f)
        filtered.sort()
        removed.sort()
        return filtered, removed

    # ---- method == "orb": 相邻帧顺序比较, 把连续相似帧压缩成一张代表帧 ----
    rep_frame, rep_name = items[0]
    rep_img = imread_unicode(os.path.join(screenshots_dir, rep_name))
    filtered.append(rep_frame)

    for frame, name in items[1:]:
        img = imread_unicode(os.path.join(screenshots_dir, name))
        if img is None or rep_img is None:
            # 读图失败当作不同, 保留
            filtered.append(frame)
            rep_frame, rep_name, rep_img = frame, name, img
            continue
        if orb_distance(rep_img, img, nfeatures) <= similarity_threshold:
            removed.append(frame)
        else:
            filtered.append(frame)
            rep_frame, rep_name, rep_img = frame, name, img

    return filtered, removed


if __name__ == "__main__":
    # 简单自测: 直接运行本文件, 在所在目录上去重并打印结果
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    f, r = deduplicate_screenshotsV2(d, method="orb", similarity_threshold=0.8)
    print(f"保留 {len(f)} 张, 移除 {len(r)} 张")
    print("removed_frames:", r)