"""图片相似度检测 CLI（ORB）。

库实现见 utils/similarity.py；本文件仅提供命令行入口。
"""
import argparse
import sys, os as _os

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from utils import compare_images, filter_unique_images, get_next_exp_dir  # noqa: E402


def run_similarity_check(args):
    if args.compare:
        if len(args.compare) != 2:
            print("请提供恰好两张图片进行比较")
            return

        result = compare_images(args.compare[0], args.compare[1], args.threshold, args.use_center, args.center_ratio)
        if result:
            print(f"\n比较结果:")
            print(f"图片1: {args.compare[0]}")
            print(f"图片2: {args.compare[1]}")
            print(f"距离: {result['distance']:.4f}")
            print(f"相似度: {result['similarity']:.2f}%")
            print(f"是否相似: {result['is_similar']}")

    elif args.filter_unique:
        output_dir = args.output or get_next_exp_dir('runs/filter_unique')
        filter_unique_images(args.filter_unique, output_dir, args.threshold, args.use_center, args.center_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用 ORB 算法进行图片相似度检测')

    parser.add_argument('--compare', nargs=2, metavar=('图片1', '图片2'),
                        help='比较两张图片的相似度')

    parser.add_argument('--filter-unique', type=str, default=None,
                        help='输入图片目录')

    parser.add_argument('--threshold', type=float, default=0.8,
                        help='相似度距离阈值 (0.0=完全相同, 1.0=完全不同)')

    parser.add_argument('--output', type=str, default=None,
                        help='结果输出目录，默认 runs/filter_unique（自动递增防覆盖）')

    parser.add_argument('--use-center', action='store_true',
                        help='使用图片中心区域进行相似度检测')

    parser.add_argument('--center-ratio', type=float, default=0.5,
                        help='中心区域比例，默认0.5表示使用宽高各一半的中心区域')

    args = parser.parse_args()

    if not (args.compare or args.filter_unique):
        parser.print_help()
        print("\n请指定以下参数之一: --compare 或 --filter-unique")
    else:
        run_similarity_check(args)