"""utils/cv.py 的回归测试（resize/frame_mad/书籍掩码）。

运行方式（项目根目录）：
    python -m unittest discover -s tests
"""
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.cv import (  # noqa: E402
    resize_max_edge, frame_mad, extract_content_contours, build_book_mask,
    book_contour_bbox,
)


class TestResizeMaxEdge(unittest.TestCase):
    def test_none_returns_same(self):
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        self.assertIs(resize_max_edge(img, None), img)

    def test_smaller_returns_same(self):
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        self.assertIs(resize_max_edge(img, 200), img)

    def test_scales_long_edge(self):
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        out = resize_max_edge(img, 500)
        h, w = out.shape[:2]
        self.assertEqual(max(h, w), 500)
        # 等比保持：约 500 x 400
        self.assertAlmostEqual(w / h, 1000 / 800, delta=0.02)
        self.assertLessEqual(h, 500)
        self.assertLessEqual(w, 500)


class TestFrameMad(unittest.TestCase):
    def test_identical_zero(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertAlmostEqual(frame_mad(a, a.copy()), 0.0, delta=0.1)

    def test_none_inf(self):
        self.assertEqual(frame_mad(None, None), float('inf'))

    def test_different_positive(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertGreater(frame_mad(a, b), 0.0)


class TestContentContours(unittest.TestCase):
    def _block(self):
        g = np.zeros((256, 256), dtype=np.uint8)
        g[50:150, 40:130] = 255
        return g

    def test_blank_empty(self):
        self.assertEqual(extract_content_contours(np.zeros((256, 256), dtype=np.uint8)), [])

    def test_block_yields_contour(self):
        g = self._block()
        contours = extract_content_contours(g, canny_low=50, canny_high=150)
        self.assertTrue(contours)

    def test_build_book_mask_blank_none(self):
        self.assertIsNone(build_book_mask(np.zeros((256, 256), dtype=np.uint8)))

    def test_build_book_mask_has_pixels(self):
        g = self._block()
        mask = build_book_mask(g, canny_low=50, canny_high=150)
        self.assertIsNotNone(mask)
        self.assertTrue(np.any(mask))

    def test_book_contour_bbox(self):
        g = self._block()
        bbox = book_contour_bbox(g, min_ratio=0.04, canny_low=50, canny_high=150)
        self.assertIsNotNone(bbox)
        x, y, w, h = bbox
        self.assertEqual(len(bbox), 4)
        # 闭运算只可能让包围框外扩，必须覆盖真实内容块 [40:130, 50:150]
        self.assertLessEqual(x, 40)
        self.assertLessEqual(y, 50)
        self.assertGreaterEqual(x + w, 130)
        self.assertGreaterEqual(y + h, 150)
        self.assertGreaterEqual(w, 90)
        self.assertGreaterEqual(h, 100)

    def test_book_contour_bbox_blank_none(self):
        self.assertIsNone(book_contour_bbox(np.zeros((256, 256), dtype=np.uint8)))


if __name__ == '__main__':
    unittest.main()