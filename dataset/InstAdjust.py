import json
from pathlib import Path

labels_dir = Path("E:/data_training/dataset_sw31/labels")
for json_path in labels_dir.glob("*.json"):
    with open(json_path) as f:
        data = json.load(f)
    _, info = data[0]
    seg = info["seg"]
    n = len(seg)
    # 关键修改：改为简单一维列表，长度等于面数
    info["inst"] = [0] * n
    # 同时确保 bottom 键连续
    info["bottom"] = {str(i): info["bottom"].get(str(i), 0) for i in range(n)}
    data[0] = [data[0][0], info]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
print("inst 已改为一维列表格式")