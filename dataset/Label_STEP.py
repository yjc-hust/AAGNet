import os
import re
import json
from pathlib import Path
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TopoDS import topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Pnt

# ================= 指纹工具 =================
def get_face_fingerprint(face, sample_u=0.5, sample_v=0.5):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass()
    centre = props.CentreOfMass()
    cx, cy, cz = centre.Coord()
    try:
        adaptor = BRepAdaptor_Surface(face, True)
        u = adaptor.FirstUParameter() + (adaptor.LastUParameter() - adaptor.FirstUParameter()) * sample_u
        v = adaptor.FirstVParameter() + (adaptor.LastVParameter() - adaptor.FirstVParameter()) * sample_v
        pt = gp_Pnt()
        adaptor.D0(u, v, pt)
        return (round(cx, 6), round(cy, 6), round(cz, 6), round(area, 4),
                round(pt.X(), 6), round(pt.Y(), 6), round(pt.Z(), 6))
    except:
        return (round(cx, 6), round(cy, 6), round(cz, 6), round(area, 4), 0.0, 0.0, 0.0)

def match_faces(occt_fps, sw_fps):
    matches = {}
    used_sw = set()
    for oi, fp in enumerate(occt_fps):
        best_sw, best_dist = -1, float('inf')
        for si, sw_fp in enumerate(sw_fps):
            if si in used_sw: continue
            dist = ((fp[0]-sw_fp[0])**2 + (fp[1]-sw_fp[1])**2 + (fp[2]-sw_fp[2])**2) ** 0.5
            area_diff = abs(fp[3] - sw_fp[3])
            if dist < 1e-4 and area_diff < 1e-4:
                if dist < best_dist:
                    best_dist, best_sw = dist, si
        if best_sw >= 0:
            matches[oi] = best_sw
            used_sw.add(best_sw)
    return matches

def load_faces_from_step(step_path):
    """返回 (faces_fps, faces_topo)"""
    reader = STEPCAFControl_Reader()
    reader.ReadFile(str(step_path))
    doc = TDocStd_Document("MDTV_XCAF")
    reader.Transfer(doc)
    tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    labels = TDF_LabelSequence()
    tool.GetFreeShapes(labels)
    faces_fps, faces_topo = [], []
    for i in range(1, labels.Length() + 1):
        shape = tool.GetShape(labels.Value(i))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods.Face(exp.Current())
            faces_fps.append(get_face_fingerprint(face))
            faces_topo.append(face)
            exp.Next()
    return faces_fps, faces_topo

# ================= 标签嵌入（文本编辑） =================
def embed_labels_into_step(sw_step_path, sw_fps, matches, seg_labels, output_path):
    """
    读取 SW 原始 STEP 文本，按顺序将标签写入 ADVANCED_FACE 行。
    """
    with open(sw_step_path, 'r') as f:
        lines = f.readlines()

    # 正则匹配 ADVANCED_FACE 行，捕获其 NAME 字段（第一个单引号字符串）
    # 示例：#123 = ADVANCED_FACE('',(#124,#125),#130,.F.);
    adv_face_pattern = re.compile(r"^(#\d+\s*=\s*ADVANCED_FACE\s*\(\s*)'([^']*)'")

    # 构建面索引 -> 标签名
    sw_label = {}
    for oi, si in matches.items():
        label_id = seg_labels[oi]
        sw_label[si] = f"class_{label_id}"

    # 当前处理的 ADVANCED_FACE 索引（0 base）
    face_idx = 0
    new_lines = []
    for line in lines:
        match = adv_face_pattern.match(line.strip())
        if match:
            # 这是一个 ADVANCED_FACE 行
            if face_idx in sw_label:
                new_name = sw_label[face_idx]
                # 替换名称部分
                new_line = match.group(1) + "'" + new_name + "'" + line.strip()[match.end(2)+1:]  # 原行剩余部分
                new_lines.append(new_line + "\n")
            else:
                new_lines.append(line)  # 无标签，保留原样（或可写'Unknown'）
            face_idx += 1
        else:
            new_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

# ================= 主批量处理 =================
if __name__ == "__main__":
    occt_step_dir = Path(r"C:\Users\Lenovo\AAGNet\data\steps")
    occt_label_dir = Path(r"C:\Users\Lenovo\AAGNet\data\labels")
    sw_step_dir = Path(r"C:\Users\Lenovo\AAGNet\data\sw_steps_norm\steps")
    output_dir = Path(r"E:\data_training\dataset_sw33\steps")
    output_dir.mkdir(parents=True, exist_ok=True)

    occt_files = list(occt_step_dir.glob("*.step")) + list(occt_step_dir.glob("*.stp"))
    if not occt_files:
        print("错误：OCCT STEP文件夹中无文件！")
        exit(1)

    success, fails = 0, []
    for occt_path in occt_files:
        base = occt_path.stem
        label_path = occt_label_dir / f"{base}.json"
        sw_path = sw_step_dir / f"{base}.step"
        if not sw_path.exists():
            sw_path = sw_step_dir / f"{base}.stp"
        if not label_path.exists() or not sw_path.exists():
            print(f"跳过 {base}：缺少标签或SW STEP")
            fails.append(base)
            continue

        print(f"处理 {base} ...")
        # 1. 加载标签
        with open(label_path) as f:
            data = json.load(f)
        seg = data[0][1]["seg"]
        seg_labels = {int(k): v for k, v in seg.items()}

        # 2. 加载 OCCT 面指纹
        occt_fps, _ = load_faces_from_step(occt_path)
        if len(occt_fps) != len(seg_labels):
            print(f"  跳过：OCCT面数({len(occt_fps)})与标签数({len(seg_labels)})不匹配")
            fails.append(base)
            continue

        # 3. 加载 SW 面指纹
        sw_fps, _ = load_faces_from_step(sw_path)

        # 4. 匹配
        matches = match_faces(occt_fps, sw_fps)
        print(f"  匹配：{len(matches)}/{len(seg_labels)}")

        # 5. 用文本方式嵌入标签
        output_path = output_dir / f"{base}.step"
        embed_labels_into_step(sw_path, sw_fps, matches, seg_labels, output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"  成功生成：{output_path.name}")
            success += 1
        else:
            print(f"  生成失败")
            fails.append(base)

    print(f"\n完成！成功 {success} 个，失败 {len(fails)} 个")
    if fails:
        print("失败文件：", fails)