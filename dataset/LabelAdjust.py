import json
from pathlib import Path

def fix_label_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 结构是 [ [filename, { "seg": {...}, ... } ] ]
    filename, info = data[0]
    seg = info.get("seg", {})
    num_faces = len(seg)
    if num_faces == 0:
        print(f"  跳过 {filename}：seg 为空")
        return

    # 补全 inst
    inst = info.get("inst", [])
    if not isinstance(inst, list) or len(inst) != 1 or len(inst[0]) != num_faces:
        inst = [[0] * num_faces]

    # 补全 bottom
    bottom = info.get("bottom", {})
    new_bottom = {str(i): bottom.get(str(i), 0) for i in range(num_faces)}

    info["inst"] = inst
    info["bottom"] = new_bottom
    data[0] = [filename, info]

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"已修复：{json_path.name}")

if __name__ == "__main__":
    labels_dir = Path(r"E:/data_training/dataset_sw31/labels")
    for json_file in labels_dir.glob("*.json"):
        fix_label_file(json_file)