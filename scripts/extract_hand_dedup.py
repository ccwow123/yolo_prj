"""提取 hand_distance 产生的去重 jpg 目录，复制到 runs 下的同名目录（支持批量 expN）。

逻辑：
  1. 确定要处理的 expN 列表：--exp 逗号分隔指定多个；--all 处理全部；默认取最近修改的一个
  2. 遍历每个 expN 下所有非 screenshots 的子目录（即各视频的去重图目录 <视频名>/）
  3. 收集各目录下的 screenshot_*.jpg，复制到 <target_root>/<视频名>/（保留原目录名，
     按视频子目录分组避免跨视频帧号重名；同名视频重复出现时同名文件覆盖取最新）

用法示例:
    python extract_hand_dedup.py                           # 本次最新 expN → runs
    python extract_hand_dedup.py --exp exp12,exp13         # 指定多个 exp
    python extract_hand_dedup.py --all                     # 全部 expN
    python extract_hand_dedup.py --exp exp12 --target-root runs\\某画册
"""

import argparse
import logging
import os
import shutil

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from utils.core import configure_logging, get_files_by_extension

logger = logging.getLogger(__name__)

DEFAULT_HAND_SAVE_DIR = r'runs\hand_distance'
DEFAULT_TARGET_ROOT = r'runs'


def find_latest_exp(hand_save_dir):
    """返回目录下最近修改的子目录（通常是本次 expN）。"""
    subdirs = list_subdirs(hand_save_dir)
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def list_subdirs(base_dir):
    return [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]


def resolve_exp_dirs(hand_save_dir, exp_arg, all_exps):
    """把 --exp / --all 参数解析为 expN 目录路径列表。"""
    if all_exps:
        dirs = list_subdirs(hand_save_dir)
        if not dirs:
            logger.warning(f"hand_distance 输出目录下没有子目录: {hand_save_dir}")
        return dirs
    if exp_arg:
        names = [e.strip() for e in exp_arg.split(',') if e.strip()]
        dirs = []
        for name in names:
            path = os.path.join(hand_save_dir, name)
            if os.path.isdir(path):
                dirs.append(path)
            else:
                logger.warning(f"跳过（expN 目录不存在）: {path}")
        return dirs
    latest = find_latest_exp(hand_save_dir)
    return [latest] if latest else []


def collect_dedup_dirs(exp_dir):
    """返回 expN 下所有去重图目录：(来源目录路径, 目录名, [jpg路径...]) 列表。"""
    result = []
    for name in os.listdir(exp_dir):
        src = os.path.join(exp_dir, name)
        if not os.path.isdir(src) or name == 'screenshots':
            continue
        jpgs = get_files_by_extension(src, ('.jpg', '.jpeg'))
        if not jpgs:
            continue
        result.append((src, name, [os.path.join(src, f) for f in jpgs]))
    return result


def copy_dedup_dir(src, name, jpgs, target_root, dry_run):
    if dry_run:
        logger.info(f"  - {name}: {len(jpgs)} 张 → {os.path.join(target_root, name)}")
        return len(jpgs)
    dst = os.path.join(target_root, name)
    os.makedirs(dst, exist_ok=True)
    for jpg in jpgs:
        shutil.copy2(jpg, os.path.join(dst, os.path.basename(jpg)))
    logger.info(f"  - {name}: {len(jpgs)} 张 → {dst}")
    return len(jpgs)


def extract_dedup_images(hand_save_dir, exp_arg=None, all_exps=False,
                         target_root=DEFAULT_TARGET_ROOT, dry_run=False):
    exp_dirs = resolve_exp_dirs(hand_save_dir, exp_arg, all_exps)
    if not exp_dirs:
        logger.error(f"没有可处理的 expN 目录: {hand_save_dir}")
        return

    os.makedirs(target_root, exist_ok=True)
    total = 0
    for exp_dir in exp_dirs:
        logger.info(f"hand_distance 结果目录: {exp_dir}")
        dedup_dirs = collect_dedup_dirs(exp_dir)
        if not dedup_dirs:
            logger.warning(f"{exp_dir} 下没有找到去重 jpg 目录（可能不是视频输出，或未触发截图）。")
            continue
        for src, name, jpgs in dedup_dirs:
            total += copy_dedup_dir(src, name, jpgs, target_root, dry_run)

    verb = "将复制" if dry_run else "已复制"
    logger.info(f"{verb} {total} 张去重图" + ("" if dry_run else f" 到 {target_root}"))


def main():
    parser = argparse.ArgumentParser(description='提取 hand_distance 去重 jpg 目录到 runs（支持批量 expN）')
    parser.add_argument('--hand-save-dir', type=str, default=DEFAULT_HAND_SAVE_DIR,
                        help='hand_distance 输出父目录（默认 runs\\hand_distance）')
    parser.add_argument('--exp', type=str, default=None,
                        help='指定多个 expN，逗号分隔（如 exp12,exp13）；默认自动取最近修改的一个')
    parser.add_argument('--all', dest='all_exps', action='store_true',
                        help='处理所有 expN（与 --exp 互斥，--exp 优先）')
    parser.add_argument('--target-root', type=str, default=DEFAULT_TARGET_ROOT,
                        help='复制目标根目录，每个去重目录保留原名作为其下子目录')
    parser.add_argument('--dry-run', action='store_true', help='只查看将复制的文件，不真正复制')
    args = parser.parse_args()

    configure_logging()
    extract_dedup_images(args.hand_save_dir, args.exp, args.all_exps,
                         args.target_root, args.dry_run)


if __name__ == '__main__':
    main()