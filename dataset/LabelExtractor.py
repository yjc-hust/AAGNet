import os
import re
import json
from pathlib import Path

def extract_labels_from_step_text(step_path):
    """
    从直接编辑过名称的 STEP 文件中提取面标签。
    假设每个 ADVANCED_FACE 行的第一个单引号字符串为 'class_数字'。
    返回 {面索引: 类别ID} 的字典，顺序按文件中 ADVANCED_FACE 出现顺序（0 起始）。
    """
    with open(step_path, 'r') as f:
        lines = f.readlines()

    # 匹配 ADVANCED_FACE(...) 中第一个单引号里的内容
    pattern = re.compile(r"^\s*#\d+\s*=\s*ADVANCED_FACE\s*\(\s*'([^']*)'")
    seg = {}
    face_idx = 0
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            name = match.group(1)
            if name.startswith("class_"):
                try:
                    class_id = int(name.split("_")[1])
                    seg[str(face_idx)] = class_id
                except ValueError:
                    pass  # 解析失败则忽略
            face_idx += 1
    return seg

def create_sw_label_json(step_path, output_json_path, model_name):
    seg = extract_labels_from_step_text(step_path)
    json_data = [
        [
            model_name,
            {
                "seg": seg,
                "inst": [],      # 如果需要，可从 OCCT 标签复制
                "bottom": {}
            }
        ]
    ]
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"{model_name}: 提取到 {len(seg)} 个面标签 → {output_json_path.name}")

# ========== 配置路径 ==========
sw_labeled_dir = Path(r"E:\data_training\dataset_sw33\steps")   # 存放带标签 STEP 的文件夹
output_label_dir = Path(r"E:\data_training\dataset_sw33\labels")        # 输出 JSON 文件夹
output_label_dir.mkdir(parents=True, exist_ok=True)

step_files = list(sw_labeled_dir.glob("*.step")) + list(sw_labeled_dir.glob("*.stp"))
if not step_files:
    print("未找到带标签的 STEP 文件！")
    exit(1)

for step_path in step_files:
    base = step_path.stem
    # 去掉 _labeled 后缀，得到原始模型名（例如 20260414_160844_0）
    if base.endswith("_labeled"):
        model_name = base[:-len("_labeled")]
    else:
        model_name = base
    out_path = output_label_dir / f"{model_name}.json"
    create_sw_label_json(step_path, out_path, model_name)