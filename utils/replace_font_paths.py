#!/usr/bin/env python3
"""Replace font_path values in JSON files.

Examples:
    python replace_font_paths.py Arial Arial-Unicode-Regular 0009_10_translations.json
    python replace_font_paths.py Arial Arial-Unicode-Regular ./json_files --recursive
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


def replace_font_paths(data: Any, old_font: str, new_font: str) -> tuple[bool, int]:
    """Recursively replace font_path values using a regular expression pattern."""
    changed = False
    count = 0

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "font_path" and isinstance(value, str):
                new_value = re.sub(old_font, new_font, value)
                if new_value != value:
                    data[key] = new_value
                    changed = True
                    count += 1
            else:
                child_changed, child_count = replace_font_paths(value, old_font, new_font)
                changed = changed or child_changed
                count += child_count
    elif isinstance(data, list):
        for item in data:
            child_changed, child_count = replace_font_paths(item, old_font, new_font)
            changed = changed or child_changed
            count += child_count

    return changed, count


def process_file(path: Path, old_font: str, new_font: str) -> tuple[bool, int]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    changed, count = replace_font_paths(data, old_font, new_font)
    if changed:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=4)
            fh.write("\n")
    return changed, count


def iter_json_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() == ".json" else []

    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")

    return sorted(p for p in target.glob("*.json") if p.is_file())


def main() -> None:
    """程序主入口函数。

    负责解析命令行参数，遍历JSON文件并批量替换其中的 font_path 字段值。

    命令行参数：
        old_font (str, 可选): 需要被替换的字体名称，默认为 "Arial-Unicode-Regular"
        new_font (str, 可选): 替换后的字体名称，默认为 "Arial"
        target (str, 可选): 目标JSON文件路径或包含JSON文件的目录路径，默认为指定路径

    执行流程：
        1. 解析命令行参数
        2. 获取目标路径下所有JSON文件列表
        3. 遍历每个JSON文件，调用 process_file 进行字体替换
        4. 统计并输出修改结果

    返回：
        None: 无返回值，直接打印执行结果到控制台
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Replace font_path values in JSON files")
    parser.add_argument("old_font", nargs="?", default="Arial-Unicode-Regular", 
                        help="Font name to replace, for example: Arial")
    parser.add_argument("new_font", nargs="?", default="Arial", 
                        help="Replacement font name, for example: Arial-Unicode-Regular")
    parser.add_argument("target", nargs="?", 
                        default=r"E:\Share\1\original_images\manga_translator_work\json", 
                        help="JSON file or directory to process")

    # 解析命令行参数
    args = parser.parse_args()

    # 将目标路径转换为 Path 对象
    target = Path(args.target)
    # 获取所有需要处理的JSON文件列表
    files = iter_json_files(target)

    # 如果未找到任何JSON文件，输出提示并退出
    if not files:
        print("No JSON files found.")
        return

    # 存储被修改的文件信息（路径和替换次数）
    modified_files = []
    # 遍历处理每个JSON文件
    for path in files:
        changed, count = process_file(path, args.old_font, args.new_font)
        if changed:
            modified_files.append((path, count))
            
    # 输出最终处理结果
    if not modified_files:
        print("No matching font_path values were found.")
    else:
        print(f"Total modified files: {len(modified_files)}")
        for path, count in modified_files:
            print(f"- {path}: {count} replacements")


if __name__ == "__main__":
    main()