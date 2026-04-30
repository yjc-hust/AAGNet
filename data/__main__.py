# -*- coding: utf-8 -*-
"""
Brep归一化模块（主程序）
适用于SolidWorks与OCC生成的STEP文件归一化处理
包含：解析、几何归一化、拓扑清洗、拓扑重构、导出

依赖：pythonOCC
"""
import os
import csv
# ==========================
# 1. STEP解析模块
# ==========================
from OCC.Extend.DataExchange import read_step_file, write_step_file
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.gp import gp_Pnt


class BRepModel:
    """统一B-rep中间表示"""
    def __init__(self):
        self.faces = []
        self.edges = []
        self.vertices = []
        # 面邻接表：face_id -> 相邻face_id列表
        self.adjacency = {}
        # 邻接边表：face_i -> {face_j: shared_edge}
        self.adjacency_edges = {}


# ==========================
# 2. STEP解析函数
# ==========================

def parse_step(file_path):
    """
    读取STEP文件并转换为BRepModel
    """
    shape = read_step_file(file_path)
    model = BRepModel()

    # 解析面
    exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    while exp_face.More():
        face = exp_face.Current()
        surf = BRepAdaptor_Surface(face)
        model.faces.append({
            "type": surf.GetType(),
            "face": face
        })
        exp_face.Next()

    # 解析边
    exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp_edge.More():
        edge = exp_edge.Current()
        curve = BRepAdaptor_Curve(edge)
        model.edges.append({
            "type": curve.GetType(),
            "edge": edge
        })
        exp_edge.Next()

    # 解析顶点
    exp_vertex = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp_vertex.More():
        vertex = exp_vertex.Current()
        model.vertices.append(vertex)
        exp_vertex.Next()

        # 构建真实面邻接关系
    model = build_adjacency(model, shape)

    return model, shape


# ==========================
# 3. 几何归一化
# ==========================

def normalize_geometry(shape):
    """
    几何归一化：
    - 单位统一
    - 平移至原点
    """
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add
    from OCC.Core.gp import gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    # 计算包围盒
    box = Bnd_Box()
    brepbndlib_Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

    # 计算中心
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2

    # 平移到原点
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Pnt(cx, cy, cz), gp_Pnt(0, 0, 0))

    transformer = BRepBuilderAPI_Transform(shape, trsf)
    new_shape = transformer.Shape()

    return new_shape


# ==========================
# 4. 拓扑清洗
# ==========================

def clean_topology(shape, tol=1e-6):
    """
    拓扑清洗：
    - 删除短边
    - 合并重复顶点（简化版）
    """
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing

    sewing = BRepBuilderAPI_Sewing(tol)
    sewing.Add(shape)
    sewing.Perform()

    return sewing.SewedShape()


# ==========================
# 5. 拓扑关系重构
# ==========================

def build_adjacency(model, shape):
    """
    构建真实的面邻接关系（兼容版）

    方法：
    1. 遍历每个面；
    2. 提取该面的所有边；
    3. 建立“边hash -> 面编号列表”的映射；
    4. 若两张面共享同一条边，则判定为邻接。
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopoDS import topods

    # 初始化
    adjacency = {i: [] for i in range(len(model.faces))}
    adjacency_edges = {i: {} for i in range(len(model.faces))}

    # 建立“边 -> 面集合”的映射
    edge_to_faces = {}

    for face_id, face_info in enumerate(model.faces):
        face = face_info["face"]

        exp_edge = TopExp_Explorer(face, TopAbs_EDGE)
        while exp_edge.More():
            edge = topods.Edge(exp_edge.Current())

            # 用 hash(edge) 作为边的唯一标识
            edge_key = hash(edge)

            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = {
                    "edge": edge,
                    "faces": []
                }

            edge_to_faces[edge_key]["faces"].append(face_id)
            exp_edge.Next()

    # 根据共享边建立邻接关系
    for edge_key, info in edge_to_faces.items():
        current_faces = list(set(info["faces"]))   # 去重
        edge = info["edge"]

        if len(current_faces) >= 2:
            for i in range(len(current_faces)):
                for j in range(i + 1, len(current_faces)):
                    f1 = current_faces[i]
                    f2 = current_faces[j]

                    if f2 not in adjacency[f1]:
                        adjacency[f1].append(f2)
                    if f1 not in adjacency[f2]:
                        adjacency[f2].append(f1)

                    adjacency_edges[f1][f2] = edge
                    adjacency_edges[f2][f1] = edge

    # 排序，保证输出稳定
    for k in adjacency:
        adjacency[k] = sorted(adjacency[k])

    model.adjacency = adjacency
    model.adjacency_edges = adjacency_edges
    return model


# ==========================
# 6. 属性剥离（占位）
# ==========================

def strip_metadata(shape):
    """
    STEP属性剥离（当前简化）
    """
    return shape


# ==========================
# 7. 差异量化（增强版）
# ==========================

def count_face_types(model):
    """
    统计模型中的面类型分布
    返回：{面类型: 数量}
    """
    type_count = {}
    for face_info in model.faces:
        face_type = str(face_info["type"])
        type_count[face_type] = type_count.get(face_type, 0) + 1
    return type_count


def face_type_diff(model1, model2):
    """
    计算面类型分布差异

    说明：
    - 即使两个模型面数相同，其面类型构成也可能不同；
    - 该指标反映几何表示层面的差异。
    """
    t1 = count_face_types(model1)
    t2 = count_face_types(model2)
    all_keys = set(t1.keys()) | set(t2.keys())

    diff_sum = 0
    total = 0
    for k in all_keys:
        a = t1.get(k, 0)
        b = t2.get(k, 0)
        diff_sum += abs(a - b)
        total += max(a, b, 1)

    return diff_sum / total if total > 0 else 0


def adjacency_diff(model1, model2):
    """
    计算邻接结构差异

    说明：
    - 统计两模型中的面邻接边数量；
    - 用于衡量拓扑关系层面的差异。
    """
    a1 = sum(len(v) for v in model1.adjacency.values()) // 2
    a2 = sum(len(v) for v in model2.adjacency.values()) // 2
    return abs(a1 - a2) / max(a1, a2, 1)


def vertex_position_diff(model1, model2):
    """
    计算顶点集合的最近邻平均偏差（改进版）

    方法：
    - 不再按顶点遍历顺序直接配对；
    - 对model1中每个顶点，在model2中寻找最近邻顶点；
    - 再对model2中每个顶点，在model1中寻找最近邻顶点；
    - 取双向平均，降低顶点顺序不同带来的误差。

    说明：
    - 该方法更适合比较不同来源STEP/B-rep模型的几何位置差异；
    - 可视为简化版Chamfer Distance。
    """
    from OCC.Core.BRep import BRep_Tool

    if len(model1.vertices) == 0 or len(model2.vertices) == 0:
        return 0.0

    pts1 = []
    pts2 = []

    for v in model1.vertices:
        p = BRep_Tool.Pnt(v)
        pts1.append((p.X(), p.Y(), p.Z()))

    for v in model2.vertices:
        p = BRep_Tool.Pnt(v)
        pts2.append((p.X(), p.Y(), p.Z()))

    def point_dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 +
                (p1[1] - p2[1]) ** 2 +
                (p1[2] - p2[2]) ** 2) ** 0.5

    # model1 -> model2 最近邻距离
    dist_1_to_2 = 0.0
    for p1 in pts1:
        min_d = min(point_dist(p1, p2) for p2 in pts2)
        dist_1_to_2 += min_d
    dist_1_to_2 /= len(pts1)

    # model2 -> model1 最近邻距离
    dist_2_to_1 = 0.0
    for p2 in pts2:
        min_d = min(point_dist(p2, p1) for p1 in pts1)
        dist_2_to_1 += min_d
    dist_2_to_1 /= len(pts2)

    return (dist_1_to_2 + dist_2_to_1) / 2.0


def compute_difference(model1, model2):
    """
    计算增强版差异指标

    指标包括：
    1. face_diff：面数量差异
    2. edge_diff：边数量差异
    3. vertex_diff：顶点数量差异
    4. face_type_diff：面类型分布差异
    5. adjacency_diff：面邻接关系差异
    6. vertex_position_diff：顶点平均坐标偏差
    7. total_diff_score：综合差异得分
    """
    F1, F2 = len(model1.faces), len(model2.faces)
    E1, E2 = len(model1.edges), len(model2.edges)
    V1, V2 = len(model1.vertices), len(model2.vertices)

    def diff(a, b):
        return abs(a - b) / max(a, b, 1)

    face_diff_value = diff(F1, F2)
    edge_diff_value = diff(E1, E2)
    vertex_diff_value = diff(V1, V2)
    face_type_diff_value = face_type_diff(model1, model2)
    adjacency_diff_value = adjacency_diff(model1, model2)
    vertex_position_diff_value = vertex_position_diff(model1, model2)

    # 综合差异评分
    total_diff_score = (
        0.20 * face_diff_value +
        0.15 * edge_diff_value +
        0.15 * vertex_diff_value +
        0.20 * face_type_diff_value +
        0.20 * adjacency_diff_value +
        0.10 * vertex_position_diff_value
    )

    return {
        "face_diff": face_diff_value,
        "edge_diff": edge_diff_value,
        "vertex_diff": vertex_diff_value,
        "face_type_diff": face_type_diff_value,
        "adjacency_diff": adjacency_diff_value,
        "vertex_position_diff": vertex_position_diff_value,
        "total_diff_score": total_diff_score
    }

# ==========================
# 8. 主流程函数
# ==========================

def normalize_step(input_path, output_path):
    """
    STEP归一化主函数
    """
    # 解析
    model, shape = parse_step(input_path)

    # 几何归一化
    shape = normalize_geometry(shape)

    # 拓扑清洗
    shape = clean_topology(shape)

    # 属性剥离
    shape = strip_metadata(shape)

    # 导出
    write_step_file(shape, output_path)

    print(f"归一化完成: {output_path}")

import os


def batch_normalize_folder(input_dir, output_dir):
    """
    批量归一化一个文件夹中的所有STEP文件

    参数：
    input_dir  : 输入文件夹
    output_dir : 输出文件夹
    """
    # 如果输出文件夹不存在，则自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入文件夹下所有文件
    for file_name in os.listdir(input_dir):
        # 只处理.step或.stp文件
        if file_name.lower().endswith(".step") or file_name.lower().endswith(".stp"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            print(f"开始处理: {input_path}")
            try:
                normalize_step(input_path, output_path)
            except Exception as e:
                print(f"处理失败: {input_path}")
                print(f"错误信息: {e}")

def batch_compare_folders_to_csv(folder1, folder2, csv_output_path):
    """
    批量比较两个文件夹中同名STEP文件的差异，并导出CSV结果表

    参数：
    folder1         : 第一个文件夹
    folder2         : 第二个文件夹
    csv_output_path : CSV输出路径
    """
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    common_files = sorted(files1 & files2)

    print(f"共找到 {len(common_files)} 个同名文件")

    results = []

    for file_name in common_files:
        if file_name.lower().endswith(".step") or file_name.lower().endswith(".stp"):
            path1 = os.path.join(folder1, file_name)
            path2 = os.path.join(folder2, file_name)

            try:
                model1, _ = parse_step(path1)
                model2, _ = parse_step(path2)
                diff = compute_difference(model1, model2)

                row = {
                    "file_name": file_name,
                    "face_diff": diff["face_diff"],
                    "edge_diff": diff["edge_diff"],
                    "vertex_diff": diff["vertex_diff"],
                    "face_type_diff": diff["face_type_diff"],
                    "adjacency_diff": diff["adjacency_diff"],
                    "vertex_position_diff": diff["vertex_position_diff"],
                    "total_diff_score": diff["total_diff_score"]
                }
                results.append(row)

                print(f"{file_name} -> {diff}")

            except Exception as e:
                print(f"{file_name} 比较失败: {e}")
                results.append({
                    "file_name": file_name,
                    "face_diff": "",
                    "edge_diff": "",
                    "vertex_diff": "",
                    "face_type_diff": "",
                    "adjacency_diff": "",
                    "vertex_position_diff": "",
                    "total_diff_score": "",
                    "error": str(e)
                })

    # 写出CSV
    fieldnames = [
        "file_name",
        "face_diff",
        "edge_diff",
        "vertex_diff",
        "face_type_diff",
        "adjacency_diff",
        "vertex_position_diff",
        "total_diff_score",
        "error"
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            if "error" not in row:
                row["error"] = ""
            writer.writerow(row)

    print(f"CSV结果表已导出: {csv_output_path}")

# ==========================
# 9. 运行
# ==========================
if __name__ == "__main__":
    # ========= 1. 输入文件夹 =========
    sw_input_dir = r"C:\Users\Lenovo\AAGNet\data\sw_steps"
    occ_input_dir = r"C:\Users\Lenovo\AAGNet\data\steps"

    # ========= 2. 归一化输出文件夹 =========
    sw_output_dir = r"C:\Users\Lenovo\AAGNet\data\sw_steps_norm"
    occ_output_dir = r"C:\Users\Lenovo\AAGNet\data\occ_steps_norm"

    # ========= 3. CSV输出路径 =========
    before_csv = r"C:\Users\Lenovo\AAGNet\data\before_norm_diff.csv"
    after_csv = r"C:\Users\Lenovo\AAGNet\data\after_norm_diff.csv"

    before_summary_csv = r"C:\Users\Lenovo\AAGNet\data\before_norm_summary.csv"
    after_summary_csv = r"C:\Users\Lenovo\AAGNet\data\after_norm_summary.csv"

    print("===== 任务开始 =====")
    print("SW输入文件夹:", sw_input_dir)
    print("OCC输入文件夹:", occ_input_dir)
    print("SW归一化输出文件夹:", sw_output_dir)
    print("OCC归一化输出文件夹:", occ_output_dir)

    # ========= 4. 归一化前差异分析 =========
    print("\n===== 比较归一化前差异 =====")
    batch_compare_folders_to_csv(sw_input_dir, occ_input_dir, before_csv)
    export_csv_summary(before_csv, before_summary_csv)

    # ========= 5. 批量归一化 SW 数据集 =========
    print("\n===== 开始批量归一化 SolidWorks 数据集 =====")
    batch_normalize_folder(sw_input_dir, sw_output_dir)

    # ========= 6. 批量归一化 OCC 数据集 =========
    print("\n===== 开始批量归一化 OCC 数据集 =====")
    batch_normalize_folder(occ_input_dir, occ_output_dir)

    # ========= 7. 归一化后差异分析 =========
    print("\n===== 比较归一化后差异 =====")
    batch_compare_folders_to_csv(sw_output_dir, occ_output_dir, after_csv)

    print("\n===== 全部处理完成 =====")
    print("归一化前差异表:", before_csv)
    print("归一化后差异表:", after_csv)
  