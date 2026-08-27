#!/usr/bin/env python3
"""替换 JSON 文件中的 font_path 值。

示例：
    python replace_font_paths.py Arial Arial-Unicode-Regular 0009_10_translations.json
    python replace_font_paths.py Arial Arial-Unicode-Regular ./json_files --recursive
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from utils.config import DEFAULT_FONT_JSON_DIR


def replace_font_paths(data: Any, old_font: str, new_font: str) -> tuple[bool, int]:
    """递归使用正则表达式模式替换 font_path 字段的值。"""
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
        raise FileNotFoundError(f"路径不存在：{target}")

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
    parser = argparse.ArgumentParser(description="替换 JSON 文件中的 font_path 值")
    parser.add_argument("old_font", nargs="?", default="Arial-Unicode-Regular", 
                        help="需要被替换的字体名称，例如：Arial")
    parser.add_argument("new_font", nargs="?", default="Arial", 
                        help="替换后的字体名称，例如：Arial-Unicode-Regular")
    parser.add_argument("target", nargs="?", 
                        default=DEFAULT_FONT_JSON_DIR, 
                        help="目标 JSON 文件或目录路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 将目标路径转换为 Path 对象
    target = Path(args.target)
    # 获取所有需要处理的JSON文件列表
    files = iter_json_files(target)

    # 如果未找到任何JSON文件，输出提示并退出
    if not files:
        print("未找到任何 JSON 文件。")
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
        print("未找到匹配的 font_path 值。")
    else:
        print(f"共修改文件数：{len(modified_files)}")
        for path, count in modified_files:
            print(f"- {path}: 替换 {count} 处")


if __name__ == "__main__":
    main()