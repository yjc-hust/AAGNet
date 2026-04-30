"""
批量归一化 STEP 数据集。

支持两种输入方式：

1. 输入整个数据集目录：
   dataset/
   ├─ steps/
   ├─ labels/

2. 只输入 steps 目录。

输出结构：
dataset_norm/
├─ steps/
├─ labels/              # 如果 --copy_labels 且原始 labels 存在
├─ norm_reports/
├─ type_overrides/
├─ normalize_summary.json
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm

from brep_normalizer import NormalizeConfig, normalize_one_step


def collect_step_files(step_dir: Path):
    files = []
    files.extend(step_dir.glob("*.step"))
    files.extend(step_dir.glob("*.stp"))
    return sorted(files)


def copy_label_if_exists(input_dataset: Path, output_dataset: Path, stem: str):
    """
    复制标签文件。
    兼容 .json / .seg / .txt 等可能格式。
    """
    label_dir = input_dataset / "labels"
    if not label_dir.exists():
        return False

    output_label_dir = output_dataset / "labels"
    output_label_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        label_dir / f"{stem}.json",
        label_dir / f"{stem}.seg",
        label_dir / f"{stem}.txt",
        label_dir / f"{stem}.npy",
    ]

    copied = False
    for src in candidates:
        if src.exists():
            shutil.copy2(src, output_label_dir / src.name)
            copied = True

    return copied


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dataset",
        type=str,
        default=None,
        help="输入数据集根目录，要求包含 steps 文件夹"
    )
    parser.add_argument(
        "--input_steps",
        type=str,
        default=None,
        help="只输入 STEP 文件夹"
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        required=True,
        help="输出归一化数据集根目录"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="light",
        choices=["light", "strong"],
        help="light 适合 MFCAD++ 有标签训练集；strong 适合 pyOCC/SolidWorks 无标签测试集"
    )
    parser.add_argument(
        "--copy_labels",
        action="store_true",
        help="是否复制 labels 文件夹中的同名标签"
    )
    parser.add_argument(
        "--target_box",
        type=float,
        default=2.0,
        help="归一化后最大包围盒尺寸，默认 2.0，即近似 [-1, 1]^3"
    )

    args = parser.parse_args()

    if args.input_dataset is None and args.input_steps is None:
        raise ValueError("必须指定 --input_dataset 或 --input_steps")

    output_dataset = Path(args.output_dataset)
    output_steps = output_dataset / "steps"
    report_dir = output_dataset / "norm_reports"
    override_dir = output_dataset / "type_overrides"

    output_steps.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    override_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dataset is not None:
        input_dataset = Path(args.input_dataset)
        input_steps = input_dataset / "steps"
    else:
        input_dataset = None
        input_steps = Path(args.input_steps)

    if not input_steps.exists():
        raise FileNotFoundError(f"steps 文件夹不存在: {input_steps}")

    step_files = collect_step_files(input_steps)

    cfg = NormalizeConfig(
        mode=args.mode,
        target_box=args.target_box,
    )

    summary = {
        "input_steps": str(input_steps),
        "output_dataset": str(output_dataset),
        "mode": args.mode,
        "total": len(step_files),
        "success": 0,
        "failed": 0,
        "failed_files": [],
        "label_copied": 0,
        "warnings": [],
    }

    for step_file in tqdm(step_files, desc="Normalizing STEP"):
        stem = step_file.stem
        out_step = output_steps / f"{stem}.step"
        report_path = report_dir / f"{stem}.json"
        override_path = override_dir / f"{stem}.json"

        try:
            report = normalize_one_step(
                input_step=str(step_file),
                output_step=str(out_step),
                report_path=str(report_path),
                type_override_path=str(override_path),
                cfg=cfg,
            )

            summary["success"] += 1

            if report.warning:
                summary["warnings"].append({
                    "file": step_file.name,
                    "warning": report.warning,
                })

            if args.copy_labels and input_dataset is not None:
                if copy_label_if_exists(input_dataset, output_dataset, stem):
                    summary["label_copied"] += 1

        except Exception as e:
            summary["failed"] += 1
            summary["failed_files"].append({
                "file": str(step_file),
                "error": str(e),
            })

    with open(output_dataset / "normalize_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\n========== Normalize Finished ==========")
    print(f"Total   : {summary['total']}")
    print(f"Success : {summary['success']}")
    print(f"Failed  : {summary['failed']}")
    print(f"Output  : {output_dataset}")
    print("========================================")


if __name__ == "__main__":
    main()