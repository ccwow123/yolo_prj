"""utils/core.py 的回归测试。

运行方式（项目根目录）：
    python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import (  # noqa: E402
    get_next_exp_dir, imread_unicode, is_image_file, is_video_file,
    load_source_list, collect_source_items, is_grayscale_v2,
)


def _write_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


class TestGetNextExpDir(unittest.TestCase):
    def test_increments_from_exp1(self):
        with tempfile.TemporaryDirectory() as d:
            first = get_next_exp_dir(d)
            self.assertTrue(first.endswith('exp1'))
            os.makedirs(first, exist_ok=True)  # 真实用法是先建目录才有 exp2
            second = get_next_exp_dir(d)
            self.assertTrue(second.endswith('exp2'))
            self.assertTrue(os.path.isdir(first))


class TestImreadUnicode(unittest.TestCase):
    def test_reads_chinese_filename(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, '中文路径测试.png')
            _write_image(path, np.full((20, 30, 3), 128, dtype=np.uint8))
            img = imread_unicode(path)
            self.assertIsNotNone(img)
            self.assertEqual(img.shape[:2], (20, 30))

    def test_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(imread_unicode(os.path.join(d, '不存在.png')))


class TestMediaFilters(unittest.TestCase):
    def test_is_video_file(self):
        self.assertTrue(is_video_file('a.MP4'))
        self.assertTrue(is_video_file('a/b.mkv'))
        self.assertFalse(is_video_file('a.png'))

    def test_is_image_file(self):
        self.assertTrue(is_image_file('a.JPG'))
        self.assertTrue(is_image_file('a/b.png'))
        self.assertFalse(is_image_file('a.mp4'))


class TestLoadSourceList(unittest.TestCase):
    def test_skips_blank_and_comments(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'list.txt')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('# 注释行\n\n')
                f.write('"C:\\dir\\a.png"\n')
                f.write("'D:\\b\\c.jpg'\n")
            self.assertEqual(load_source_list(path), ['C:\\dir\\a.png', 'D:\\b\\c.jpg'])


class TestCollectSourceItems(unittest.TestCase):
    def test_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, 'a.png')
            _write_image(img, np.zeros((10, 10, 3), dtype=np.uint8))
            items, err = collect_source_items(img)
            self.assertIsNone(err)
            self.assertEqual(items, [(img, False)])

    def test_directory_filters_non_media(self):
        with tempfile.TemporaryDirectory() as d:
            _write_image(os.path.join(d, 'a.png'), np.zeros((10, 10, 3), dtype=np.uint8))
            with open(os.path.join(d, 'note.txt'), 'w', encoding='utf-8') as f:
                f.write('x')
            items, err = collect_source_items(d)
            self.assertIsNone(err)
            self.assertEqual(len(items), 1)
            self.assertTrue(items[0][0].endswith('a.png'))

    def test_missing_source_returns_error(self):
        items, err = collect_source_items('E:/__no_such_path_anyway__')
        self.assertEqual(items, [])
        self.assertIsNotNone(err)


class TestIsGrayscaleV2(unittest.TestCase):
    def test_grayscale_image_true(self):
        with tempfile.TemporaryDirectory() as d:
            gray = np.full((50, 60), 150, dtype=np.uint8)
            gray[:, 20:40] = 60
            path = os.path.join(d, '灰度.png')
            _write_image(path, gray)
            self.assertTrue(is_grayscale_v2(path))

    def test_colorful_image_false(self):
        with tempfile.TemporaryDirectory() as d:
            color = np.zeros((50, 60, 3), dtype=np.uint8)
            color[:, :, 2] = 255  # 纯红
            color[:, 20:40, 0] = 255
            path = os.path.join(d, '彩色.png')
            _write_image(path, color)
            self.assertFalse(is_grayscale_v2(path))


if __name__ == '__main__':
    unittest.main()