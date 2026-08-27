import argparse
import csv
import os
from pathlib import Path

from utils import is_image_file, is_video_file


def collect_media_files(source, recursive):
    """收集目录下所有视频和图片文件的绝对路径"""
    source = Path(source)
    matched = []

    file_iter = source.rglob('*') if recursive else source.iterdir()

    for p in file_iter:
        if not p.is_file():
            continue
        if is_image_file(str(p)) or is_video_file(str(p)):
            matched.append(p)

    # 排序，保证输出稳定
    return sorted(matched, key=lambda p: str(p).lower())


def to_win_path(path_str):
    """转换为 Windows 风格绝对路径（反斜杠 + 盘符）"""
    return os.path.normpath(path_str).replace('/', '\\')


def export_paths(paths, output, fmt, win_format):
    """导出路径列表，支持 txt / csv 两种格式"""
    records = []

    for p in paths:
        path_str = str(p.resolve())
        if win_format:
            path_str = to_win_path(path_str)
        records.append({
            'path': path_str,
            'filename': p.name,
            'extension': p.suffix.lower(),
            'type': 'video' if is_video_file(str(p)) else 'image',
        })

    fmt = fmt.lower()
    if fmt == 'csv':
        with open(output, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'filename', 'extension', 'type'])
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(output, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(r['path'] + '\n')

    return records


def main():
    parser = argparse.ArgumentParser(
        description='扫描文件夹，收集所有视频和图片文件并导出 Windows 绝对路径'
    )
    parser.add_argument('--source', type=str, default='.',
                        help='要扫描的文件夹路径（默认当前目录）')
    parser.add_argument('--recursive', action='store_true',
                        help='递归扫描子文件夹（默认只扫描当前层）')
    parser.add_argument('--output', type=str, default=None,
                        help='导出文件路径（默认 scripts 当前目录下的 media_paths.txt/csv）')
    parser.add_argument('--format', type=str, choices=['txt', 'csv'], default='txt',
                        help='导出格式：txt（每行一个路径）或 csv（含文件名/类型）')
    parser.add_argument('--win-format', action='store_true',
                        help='强制使用 Windows 风格路径（反斜杠）')
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f'错误：未找到文件夹 {source}')
        return 1

    if args.output is None:
        ext = 'csv' if args.format.lower() == 'csv' else 'txt'
        output = source / f'media_paths.{ext}'
    else:
        output = Path(args.output)

    print(f'正在扫描：{source}（递归={args.recursive}）...')
    paths = collect_media_files(source, args.recursive)

    if not paths:
        print('未找到任何视频或图片文件。')
        return 1

    records = export_paths(paths, output, args.format, args.win_format)

    video_count = sum(1 for r in records if r['type'] == 'video')
    image_count = sum(1 for r in records if r['type'] == 'image')

    print(f'共收集 {len(records)} 个文件：视频 {video_count} 个，图片 {image_count} 个')
    print(f'路径已导出到：{output.resolve()}')
    for r in records[:5]:
        print(f'  {r["path"]}')
    if len(records) > 5:
        print(f'  ...（共 {len(records)} 条）')
    return 0


if __name__ == '__main__':
    exit(main())