"""
BRepNorm: 面向 AAGNet 输入适配的 B-Rep / STEP 归一化模块

核心功能：
1. STEP 读取与有效实体筛选
2. 几何信息归一化：平移到原点 + 缩放到统一尺度
3. 曲线 / 曲面类型标准化：对 BSpline / Other 类型进行几何再判定
4. 拓扑结构清洗：基础修复、退化结构处理、同域合并，可选
5. 非几何信息剥离：重新写出 STEP，去除颜色、图层、名称等不稳定信息
6. 输出归一化报告：供论文实验统计与后续 AAGNet 分析使用

说明：
- 对 pyOCC 训练集，建议使用 --mode light，避免改变 face 数量导致 label 错位。
- 对SolidWorks无标签测试集，可以使用 --mode strong，允许更强拓扑清洗。
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties
from OCC.Core.GProp import GProp_GProps

from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods

from OCC.Core.Bnd import Bnd_Box
try:
    from OCC.Core.BRepBndLib import brepbndlib
    def add_to_bbox(shape, box):
        brepbndlib.Add(shape, box)
except Exception:
    from OCC.Core.BRepBndLib import brepbndlib_Add
    def add_to_bbox(shape, box):
        brepbndlib_Add(shape, box)

from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt

from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Parabola,
    GeomAbs_BezierCurve,
    GeomAbs_BSplineCurve,
    GeomAbs_OffsetCurve,
    GeomAbs_OtherCurve,
)

try:
    from OCC.Core.ShapeFix import ShapeFix_Shape
except Exception:
    ShapeFix_Shape = None

try:
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
except Exception:
    ShapeUpgrade_UnifySameDomain = None


# =========================
# 1. 数据结构定义
# =========================

@dataclass
class NormalizeConfig:
    """
    归一化配置

    mode:
        light  : 轻量模式，适合 MFCAD++ 有标签训练集，尽量不改变拓扑实体数量
        strong : 强修复模式，适合 pyOCC / SolidWorks 无标签测试集
    target_box:
        模型归一化后的最大包围盒尺寸。
        target_box=2.0 表示缩放到近似 [-1, 1]^3。
    """
    mode: str = "light"
    target_box: float = 2.0
    tol_ratio: float = 1e-6
    plane_tol: float = 1e-4
    sphere_tol: float = 2e-3
    cylinder_tol: float = 3e-3
    min_edge_ratio: float = 1e-6
    min_face_area_ratio: float = 1e-8
    sample_u: int = 5
    sample_v: int = 5


@dataclass
class ShapeReport:
    file_name: str
    valid_before: bool
    valid_after: bool
    face_count_before: int
    face_count_after: int
    edge_count_before: int
    edge_count_after: int
    bbox_before: List[float]
    bbox_after: List[float]
    scale_factor: float
    center_before: List[float]
    surface_type_before: Dict[str, int]
    surface_type_after: Dict[str, int]
    curve_type_before: Dict[str, int]
    curve_type_after: Dict[str, int]
    warning: List[str]


# =========================
# 2. STEP 读写
# =========================

def read_step(step_path: str):
    """
    读取 STEP 文件，返回 TopoDS_Shape。
    """
    step_path = str(step_path)
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP 读取失败: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP 文件为空或无法转换: {step_path}")
    return shape


def write_step(shape, out_path: str):
    """
    写出 STEP 文件。
    重新写出会自然剥离大部分非几何信息，如颜色、图层、历史名称等。
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # AP214 相对兼容，适合 CAD 几何交换
    Interface_Static.SetCVal("write.step.schema", "AP214")

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(out_path)

    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP 写出失败: {out_path}")


# =========================
# 3. 基础拓扑与几何工具
# =========================

def iter_faces(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield topods.Face(exp.Current())
        exp.Next()


def iter_edges(shape):
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        yield topods.Edge(exp.Current())
        exp.Next()


def iter_solids(shape):
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield topods.Solid(exp.Current())
        exp.Next()


def count_faces(shape) -> int:
    return sum(1 for _ in iter_faces(shape))


def count_edges(shape) -> int:
    return sum(1 for _ in iter_edges(shape))


def is_valid_shape(shape) -> bool:
    """
    使用 OpenCascade 自带 BRepCheck_Analyzer 检查拓扑合法性。
    """
    try:
        analyzer = BRepCheck_Analyzer(shape)
        return bool(analyzer.IsValid())
    except Exception:
        return False


def bbox_info(shape) -> Tuple[List[float], List[float], float]:
    """
    返回：
    bbox = [xmin, ymin, zmin, xmax, ymax, zmax]
    center = [cx, cy, cz]
    diagonal = 包围盒对角线长度
    """
    box = Bnd_Box()
    box.SetGap(0.0)
    add_to_bbox(shape, box)

    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    bbox = [xmin, ymin, zmin, xmax, ymax, zmax]
    center = [
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        0.5 * (zmin + zmax),
    ]
    diagonal = math.sqrt(
        (xmax - xmin) ** 2 +
        (ymax - ymin) ** 2 +
        (zmax - zmin) ** 2
    )
    return bbox, center, diagonal


def max_bbox_length(bbox: List[float]) -> float:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return max(xmax - xmin, ymax - ymin, zmax - zmin)


# =========================
# 4. 几何信息归一化
# =========================

def normalize_geometry(shape, target_box: float = 2.0):
    """
    几何归一化：
    1. 平移，使包围盒中心位于原点
    2. 缩放，使最大包围盒边长为 target_box

    target_box=2.0 时，模型大致落入 [-1, 1]^3。
    """
    bbox, center, diagonal = bbox_info(shape)
    length = max_bbox_length(bbox)

    if length <= 1e-12:
        raise RuntimeError("模型包围盒尺寸过小，无法归一化")

    scale_factor = target_box / length

    # Step 1: 平移到原点
    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(gp_Vec(-center[0], -center[1], -center[2]))
    moved_shape = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()

    # Step 2: 按统一尺度缩放
    trsf_scale = gp_Trsf()
    trsf_scale.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale_factor)
    scaled_shape = BRepBuilderAPI_Transform(moved_shape, trsf_scale, True).Shape()

    return scaled_shape, scale_factor, center


# =========================
# 5. 拓扑结构清洗
# =========================

def repair_shape_light(shape):
    """
    轻量修复：
    - 适合有标签训练集
    - 尽量不改变拓扑结构
    - 只做基础 ShapeFix，不做强同域合并
    """
    if ShapeFix_Shape is None:
        return shape

    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    return fixer.Shape()


def repair_shape_strong(shape):
    """
    强修复：
    - 适合无标签模型
    - 可能改变面、边数量
    - 不建议直接用于带 face label 的训练集
    """
    repaired = repair_shape_light(shape)

    if ShapeUpgrade_UnifySameDomain is not None:
        try:
            # unify_edges=True, unify_faces=True, concat_bspline=True
            unifier = ShapeUpgrade_UnifySameDomain(repaired, True, True, True)
            unifier.Build()
            repaired = unifier.Shape()
        except Exception:
            pass

    return repaired


def repair_shape_by_mode(shape, mode: str):
    if mode.lower() == "strong":
        return repair_shape_strong(shape)
    return repair_shape_light(shape)


# =========================
# 6. 曲面类型标准化
# =========================

SURFACE_TYPE_NAME = {
    GeomAbs_Plane: "Plane",
    GeomAbs_Cylinder: "Cylinder",
    GeomAbs_Cone: "Cone",
    GeomAbs_Sphere: "Sphere",
    GeomAbs_Torus: "Torus",
    GeomAbs_BezierSurface: "BezierSurface",
    GeomAbs_BSplineSurface: "BSplineSurface",
}


def get_occ_surface_type(face) -> str:
    try:
        surf = BRepAdaptor_Surface(face)
        return SURFACE_TYPE_NAME.get(surf.GetType(), "OtherSurface")
    except Exception:
        return "InvalidSurface"


def sample_face_points(face, nu: int = 5, nv: int = 5) -> np.ndarray:
    """
    在 face 的参数域内采样点。
    注意：这里是用于类型再判定，不用于替代 AAGNet 的 UV-grid。
    """
    surf = BRepAdaptor_Surface(face)

    u0 = surf.FirstUParameter()
    u1 = surf.LastUParameter()
    v0 = surf.FirstVParameter()
    v1 = surf.LastVParameter()

    # 防止无限参数域导致异常
    if not np.isfinite([u0, u1, v0, v1]).all():
        return np.empty((0, 3), dtype=np.float64)

    if abs(u1 - u0) < 1e-12 or abs(v1 - v0) < 1e-12:
        return np.empty((0, 3), dtype=np.float64)

    points = []
    for u in np.linspace(u0, u1, nu):
        for v in np.linspace(v0, v1, nv):
            try:
                p = surf.Value(float(u), float(v))
                points.append([p.X(), p.Y(), p.Z()])
            except Exception:
                continue

    if len(points) < 6:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def fit_plane_residual(points: np.ndarray) -> float:
    """
    用 PCA 拟合平面，返回点到平面的平均残差。
    """
    centroid = points.mean(axis=0)
    pts = points - centroid
    _, s, vh = np.linalg.svd(pts, full_matrices=False)
    normal = vh[-1, :]
    distances = np.abs(pts @ normal)
    scale = max(np.linalg.norm(points.max(axis=0) - points.min(axis=0)), 1e-12)
    return float(distances.mean() / scale)


def fit_sphere_residual(points: np.ndarray) -> float:
    """
    最小二乘拟合球面。
    返回归一化半径残差。
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    A = np.column_stack([2 * x, 2 * y, 2 * z, np.ones_like(x)])
    b = x ** 2 + y ** 2 + z ** 2

    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, cz, d = sol
        center = np.array([cx, cy, cz], dtype=np.float64)
        radii = np.linalg.norm(points - center, axis=1)
        r_mean = max(float(radii.mean()), 1e-12)
        return float(radii.std() / r_mean)
    except Exception:
        return 1e9


def fit_cylinder_residual(points: np.ndarray) -> float:
    """
    粗略圆柱判定：
    - 用 PCA 主方向近似圆柱轴线
    - 计算所有点到轴线距离是否稳定
    返回半径相对标准差。
    """
    centroid = points.mean(axis=0)
    pts = points - centroid

    try:
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        axis = vh[0, :]
        axis = axis / max(np.linalg.norm(axis), 1e-12)

        proj_len = pts @ axis
        proj = np.outer(proj_len, axis)
        radial_vec = pts - proj
        radii = np.linalg.norm(radial_vec, axis=1)

        r_mean = max(float(radii.mean()), 1e-12)
        return float(radii.std() / r_mean)
    except Exception:
        return 1e9


def standardize_surface_type(face, cfg: NormalizeConfig) -> str:
    """
    曲面类型标准化。
    如果 OCC 已经识别为解析曲面，直接保留。
    如果是 BSpline / Bezier / Other，则通过采样拟合进行二次判定。
    """
    raw_type = get_occ_surface_type(face)

    # 已经是标准解析曲面，直接返回
    if raw_type in {"Plane", "Cylinder", "Cone", "Sphere", "Torus"}:
        return raw_type

    points = sample_face_points(face, cfg.sample_u, cfg.sample_v)
    if points.shape[0] < 6:
        return raw_type

    plane_res = fit_plane_residual(points)
    if plane_res < cfg.plane_tol:
        return "Plane"

    sphere_res = fit_sphere_residual(points)
    if sphere_res < cfg.sphere_tol:
        return "Sphere"

    cylinder_res = fit_cylinder_residual(points)
    if cylinder_res < cfg.cylinder_tol:
        return "Cylinder"

    return raw_type


# =========================
# 7. 曲线类型标准化
# =========================

CURVE_TYPE_NAME = {
    GeomAbs_Line: "Line",
    GeomAbs_Circle: "Circle",
    GeomAbs_Ellipse: "Ellipse",
    GeomAbs_Hyperbola: "Hyperbola",
    GeomAbs_Parabola: "Parabola",
    GeomAbs_BezierCurve: "BezierCurve",
    GeomAbs_BSplineCurve: "BSplineCurve",
    GeomAbs_OffsetCurve: "OffsetCurve",
    GeomAbs_OtherCurve: "OtherCurve",
}


def get_occ_curve_type(edge) -> str:
    try:
        curve = BRepAdaptor_Curve(edge)
        return CURVE_TYPE_NAME.get(curve.GetType(), "OtherCurve")
    except Exception:
        return "InvalidCurve"


def edge_length(edge) -> float:
    props = GProp_GProps()
    brepgprop_LinearProperties(edge, props)
    return float(props.Mass())


def sample_edge_points(edge, n: int = 12) -> np.ndarray:
    curve = BRepAdaptor_Curve(edge)
    u0 = curve.FirstParameter()
    u1 = curve.LastParameter()

    if not np.isfinite([u0, u1]).all():
        return np.empty((0, 3), dtype=np.float64)

    if abs(u1 - u0) < 1e-12:
        return np.empty((0, 3), dtype=np.float64)

    pts = []
    for u in np.linspace(u0, u1, n):
        try:
            p = curve.Value(float(u))
            pts.append([p.X(), p.Y(), p.Z()])
        except Exception:
            continue

    if len(pts) < 4:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(pts, dtype=np.float64)


def fit_line_residual(points: np.ndarray) -> float:
    centroid = points.mean(axis=0)
    pts = points - centroid

    try:
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        direction = vh[0, :]
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        proj = np.outer(pts @ direction, direction)
        dist = np.linalg.norm(pts - proj, axis=1)
        scale = max(np.linalg.norm(points.max(axis=0) - points.min(axis=0)), 1e-12)
        return float(dist.mean() / scale)
    except Exception:
        return 1e9


def fit_circle_residual(points: np.ndarray) -> float:
    """
    简化圆弧判定：
    - 先用 PCA 找近似平面
    - 投影到二维坐标后拟合圆
    """
    centroid = points.mean(axis=0)
    pts = points - centroid

    try:
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        e1 = vh[0, :]
        e2 = vh[1, :]

        x = pts @ e1
        y = pts @ e2

        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x ** 2 + y ** 2

        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, d = sol
        r = np.sqrt(max(d + cx ** 2 + cy ** 2, 1e-12))

        radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return float(radii.std() / max(r, 1e-12))
    except Exception:
        return 1e9


def standardize_curve_type(edge, cfg: NormalizeConfig) -> str:
    raw_type = get_occ_curve_type(edge)

    if raw_type in {"Line", "Circle", "Ellipse", "Hyperbola", "Parabola"}:
        return raw_type

    pts = sample_edge_points(edge, n=12)
    if pts.shape[0] < 4:
        return raw_type

    line_res = fit_line_residual(pts)
    if line_res < 1e-5:
        return "Line"

    circle_res = fit_circle_residual(pts)
    if circle_res < 5e-3:
        return "Circle"

    return raw_type


# =========================
# 8. 类型统计与 override 输出
# =========================

def count_dict(values: List[str]) -> Dict[str, int]:
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def collect_surface_types(shape, cfg: NormalizeConfig, standardized: bool = True) -> List[str]:
    values = []
    for face in iter_faces(shape):
        if standardized:
            values.append(standardize_surface_type(face, cfg))
        else:
            values.append(get_occ_surface_type(face))
    return values


def collect_curve_types(shape, cfg: NormalizeConfig, standardized: bool = True) -> List[str]:
    values = []
    for edge in iter_edges(shape):
        if standardized:
            values.append(standardize_curve_type(edge, cfg))
        else:
            values.append(get_occ_curve_type(edge))
    return values


def build_type_override(shape, cfg: NormalizeConfig) -> Dict:
    """
    输出给 AAGExtractor 可选使用的类型标准化结果。
    注意：
    - face/edge 用顺序编号。
    - 对 MFCAD++ 训练集，必须保证归一化没有改变 face 顺序。
    """
    faces = {}
    for i, face in enumerate(iter_faces(shape)):
        faces[str(i)] = standardize_surface_type(face, cfg)

    edges = {}
    for i, edge in enumerate(iter_edges(shape)):
        edges[str(i)] = standardize_curve_type(edge, cfg)

    return {
        "faces": faces,
        "edges": edges,
    }


# =========================
# 9. 主归一化函数
# =========================

def normalize_one_step(
    input_step: str,
    output_step: str,
    report_path: Optional[str] = None,
    type_override_path: Optional[str] = None,
    cfg: Optional[NormalizeConfig] = None,
):
    """
    归一化单个 STEP 文件。

    参数：
    input_step:
        输入 STEP 文件路径
    output_step:
        输出归一化 STEP 文件路径
    report_path:
        归一化报告 JSON 路径
    type_override_path:
        曲线曲面类型标准化结果 JSON 路径
    cfg:
        归一化配置
    """
    if cfg is None:
        cfg = NormalizeConfig()

    input_step = Path(input_step)
    output_step = Path(output_step)

    warnings = []

    # ---------- 读取 ----------
    shape = read_step(str(input_step))

    valid_before = is_valid_shape(shape)
    face_count_before = count_faces(shape)
    edge_count_before = count_edges(shape)
    bbox_before, center_before, _ = bbox_info(shape)

    surface_type_before = count_dict(collect_surface_types(shape, cfg, standardized=False))
    curve_type_before = count_dict(collect_curve_types(shape, cfg, standardized=False))

    # ---------- 拓扑修复 ----------
    # light 模式尽量不改变拓扑；strong 模式允许更强修复
    if not valid_before or cfg.mode.lower() == "strong":
        shape = repair_shape_by_mode(shape, cfg.mode)

    # ---------- 几何归一化 ----------
    shape, scale_factor, center = normalize_geometry(shape, cfg.target_box)

    # ---------- strong 模式下二次修复 ----------
    if cfg.mode.lower() == "strong":
        shape = repair_shape_strong(shape)

    valid_after = is_valid_shape(shape)
    face_count_after = count_faces(shape)
    edge_count_after = count_edges(shape)
    bbox_after, _, _ = bbox_info(shape)

    if not valid_after:
        warnings.append("normalized shape is still invalid")

    if cfg.mode.lower() == "light":
        if face_count_after != face_count_before:
            warnings.append(
                f"face count changed in light mode: {face_count_before} -> {face_count_after}; "
                f"label alignment may be broken"
            )

    surface_type_after = count_dict(collect_surface_types(shape, cfg, standardized=True))
    curve_type_after = count_dict(collect_curve_types(shape, cfg, standardized=True))

    # ---------- 写出 STEP ----------
    write_step(shape, str(output_step))

    # ---------- 写出报告 ----------
    report = ShapeReport(
        file_name=input_step.name,
        valid_before=valid_before,
        valid_after=valid_after,
        face_count_before=face_count_before,
        face_count_after=face_count_after,
        edge_count_before=edge_count_before,
        edge_count_after=edge_count_after,
        bbox_before=bbox_before,
        bbox_after=bbox_after,
        scale_factor=scale_factor,
        center_before=center_before,
        surface_type_before=surface_type_before,
        surface_type_after=surface_type_after,
        curve_type_before=curve_type_before,
        curve_type_after=curve_type_after,
        warning=warnings,
    )

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=4)

    # ---------- 写出类型 override ----------
    if type_override_path is not None:
        type_override_path = Path(type_override_path)
        type_override_path.parent.mkdir(parents=True, exist_ok=True)
        type_override = build_type_override(shape, cfg)
        with open(type_override_path, "w", encoding="utf-8") as f:
            json.dump(type_override, f, ensure_ascii=False, indent=4)

    return report