# smart_packing_streamlit.py
# streamlit run smart_packing_streamlit.py

import math
import random
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------
# 基本配置
# -----------------------
random.seed(123456)


# -----------------------
# 数据模型
# -----------------------
class Cargo:
    def __init__(self,
                 transport_mode: str,
                 route: str,
                 supplier: str,
                 customer_order: str,
                 length: float,
                 width: float,
                 height: float,
                 quantity: int,
                 gross_weight: float,
                 order_time: datetime,
                 package_type: str,
                 stackable: str,  # 'Y' or 'N'
                 load_dir: str,  # 'L', 'W', 'L&W'
                 attachment_id: Optional[str] = None,
                 pallet_id: Optional[str] = None,
                 uid: Optional[int] = None):
        self.transport_mode = transport_mode
        self.route = route
        self.supplier = supplier
        self.customer_order = customer_order
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.quantity = int(quantity)
        self.gross_weight = float(gross_weight)
        self.order_time = order_time
        self.package_type = package_type
        self.stackable = stackable
        self.load_dir = load_dir  # 装载长度
        self.attachment_id = attachment_id
        self.pallet_id = pallet_id
        self.uid = uid

    def volume(self) -> float:
        return self.length * self.width * self.height

    def orientations(self, container) -> List[Tuple[float, float, float, str]]:
        """返回允许的朝向：(l,w,h,mode) mode 表示装载长度 'L' 或 'W'"""
        res = []
        diag = (self.length ** 2 + self.width ** 2) ** 0.5
        if self.load_dir == 'L&W':
            res.append((self.length, self.width, self.height, 'L'))
            res.append((self.width, self.length, self.height, 'W'))

        if self.load_dir == 'L':
            res.append((self.length, self.width, self.height, 'L'))
            if diag < container.width:  # 可以九十度旋转
                res.append((self.width, self.length, self.height, 'L'))

        if self.load_dir == 'W':
            res.append((self.width, self.length, self.height, 'W'))
            if diag < container.width:  # 可以九十度旋转
                res.append((self.length, self.width, self.height, 'W'))

        # if self.load_dir in ('L', 'L&W'):
        #     res.append((self.length, self.width, self.height, 'L'))
        # if self.load_dir in ('W', 'L&W'):
        #     res.append((self.width, self.length, self.height, 'W'))

        uniq = []
        seen = set()
        for a, b, c, m in res:
            key = (round(a, 6), round(b, 6), round(c, 6), m)
            if key not in seen:
                seen.add(key)
                uniq.append((a, b, c, m))
        return uniq


class Container:
    def __init__(self, name: str, transport_mode: str, route: Optional[str],
                 length: float, width: float, height: float, max_weight: float,
                 safe_margin=(0.0, 0.0, 0.0)):
        self.name = name
        self.transport_mode = transport_mode
        self.route = route
        self.length = length - safe_margin[0]
        self.width = width - safe_margin[1]
        self.height = height - safe_margin[2]
        self.max_weight = max_weight
        self.safe_margin = safe_margin

    def volume(self):
        return self.length * self.width * self.height


# -----------------------
# 时间单元与分桶
# -----------------------
def week_unit_for_route(route: str, transport_mode: str, dt: datetime):
    """
    返回用于比较的时间单元标识：
    - Ground -> ('G', year, iso_week)
    - Ocean US -> 每周二 ~ 下周一 为一个时间段
    - Ocean SG -> (SG_A: 周三~周五, SG_B: 周六~下周二)
    - Ocean MY -> 没有额外限制，用 iso_week
    """
    dow = dt.weekday()  # Monday=0, Sunday=6

    if transport_mode == 'Ocean':
        if route.upper() == 'US':
            # 以周二为 anchor
            base_monday = dt - timedelta(days=dow)  # 本周一
            base_tuesday = base_monday + timedelta(days=1)  # 本周二
            if dt >= base_tuesday:
                anchor = base_tuesday
            else:
                anchor = base_tuesday - timedelta(days=7)
            return 'US', anchor.strftime('%Y-%m-%d')

        elif route.upper() == 'SG':
            # SG_A: 周三~周五
            # SG_B: 周六~下周二
            base_monday = dt - timedelta(days=dow)  # 本周一
            if dow in (2, 3, 4):  # Wed-Thu-Fri
                anchor = base_monday + timedelta(days=2)  # Wed
                return 'SG_A', anchor.strftime('%Y-%m-%d')
            else:
                anchor = base_monday + timedelta(days=5)  # Sat
                if dt < anchor:  # 本周一二属于上一个 SG_B 段
                    anchor -= timedelta(days=7)
                return 'SG_B', anchor.strftime('%Y-%m-%d')

        elif route.upper() == 'MY':
            y, w, _ = dt.isocalendar()
            return 'MY', y, w

    # Ground 默认按 ISO 周
    y, w, _ = dt.isocalendar()
    return 'G', y, w


def within_days_span(times: List[datetime], max_days: int) -> bool:
    if not times:
        return True
    return (max(times) - min(times)).days <= max_days


def group_by_transport_and_route(cargos: List[Cargo]) -> Dict[str, List[Cargo]]:
    buckets = {'Ground': [], 'Ocean-US': [], 'Ocean-SG': [], 'Ocean-MY': []}
    for c in cargos:
        if c.transport_mode == 'Ground':
            buckets['Ground'].append(c)
        elif c.transport_mode == 'Ocean':
            r = (c.route or '').upper()
            if r == 'US':
                buckets['Ocean-US'].append(c)
            elif r == 'SG':
                buckets['Ocean-SG'].append(c)
            elif r == 'MY':
                buckets['Ocean-MY'].append(c)
            else:
                buckets['Ocean-MY'].append(c)
        else:
            buckets['Ground'].append(c)
    return buckets


def create_binding_groups(cargos: List[Cargo]) -> List[List[Cargo]]:
    binding = {}
    unbound = []
    for c in cargos:
        key = None
        if c.attachment_id:
            key = ('ATT', str(c.attachment_id))
        elif c.pallet_id:
            key = ('PAL', str(c.pallet_id))
        elif c.customer_order:
            key = ('CO', str(c.customer_order))
        if key:
            binding.setdefault(key, []).append(c)
        else:
            unbound.append(c)
    groups = [v[:] for v in binding.values()]
    for c in unbound:
        groups.append([c])
    return groups


def sort_groups_for_packing(groups: List[List[Cargo]], transport_mode: str) -> List[List[Cargo]]:
    pkg_rank = {'Crate': 0, 'Pallet': 1, 'Box': 2}
    for g in groups:
        g.sort(key=lambda c: (pkg_rank.get(c.package_type, 3), c.customer_order or '', -c.height, c.uid))
    # Ground 优先 Rockwell
    if transport_mode == 'Ground':
        rock = [g for g in groups if any(c.supplier == 'Rockwell' for c in g)]
        others = [g for g in groups if not any(c.supplier == 'Rockwell' for c in g)]
        groups = rock + others
    else:
        groups.sort(key=lambda x: (pkg_rank.get(x[0].package_type, 3), x[0].customer_order or '', -x[0].height))
    return groups


# -----------------------
# 规则检查：堆叠、时效、进叉面、左右平衡
# -----------------------
def validate_time_rules_for_container(cargos: List[Cargo], transport_mode_hint: Optional[str] = None) -> bool:
    """
    检查容器内货物是否满足时效规则：
    - Ground:
        i. 同一容器内最早和最晚 order_time ≤ 2 天
        ii. 当供应商包含 FUYAO 和 FOXSEMICON 时，其他供应商的 order_time 不得晚于 min(FUYAO/FOXSEMICON)+1 天
        iii. 不允许跨 ISO 周
    - Ocean-US: 周二 ~ 下周一 为一个时间段，不能跨段
    - Ocean-SG:
        i. SG_A 段：周三~周五
        ii. SG_B 段：周六~下周二
        同一容器不能跨段
    - Ocean-MY: 无时效限制
    """
    if not cargos:
        return True

    # 确认运输方式
    modes = set(c.transport_mode for c in cargos)
    if transport_mode_hint:
        if any(m != transport_mode_hint for m in modes):
            return False
    transport_mode = transport_mode_hint if transport_mode_hint else cargos[0].transport_mode

    # ============ Ground ============ #
    if transport_mode == 'Ground':
        times = [c.order_time for c in cargos]
        if not times:
            return True

        # i. 时间跨度 ≤ 2 天
        if (max(times) - min(times)).days > 2:
            return False

        # ii. FUYAO/FOXSEMICON 特殊供应商规则
        suppliers = set(c.supplier for c in cargos)
        if 'FUYAO' in suppliers and 'FOXSEMICON' in suppliers:
            special_times = [c.order_time for c in cargos if c.supplier in ('FUYAO', 'FOXSEMICON')]
            if special_times:
                min_sp = min(special_times)
                for c in cargos:
                    if c.supplier not in ('FUYAO', 'FOXSEMICON'):
                        if (c.order_time - min_sp).days > 1:
                            return False

        # iii. 不允许跨 ISO 周
        weeks = set((t.isocalendar()[0], t.isocalendar()[1]) for t in times)
        if len(weeks) > 1:
            return False

        return True

    # ============ Ocean ============ #
    elif transport_mode == 'Ocean':
        routes = set((c.route or '').upper() for c in cargos)
        if len(routes) > 1:
            return False
        route = next(iter(routes))

        if route == 'MY':
            return True  # 无规则，直接合法

        # US/SG 依赖 week_unit_for_route
        units = set(week_unit_for_route(route, 'Ocean', c.order_time) for c in cargos)
        if len(units) > 1:
            return False
        return True

    return True


# -----------------------
# 几何函数、极点方法 (EPP)
# placed 项统一为 (x,y,z,(l,w,h), cargo, mode)
# -----------------------

def check_stack_rules_on_stack(stack: List[Cargo]) -> bool:
    """
    给定一个垂直堆叠的货物列表（从下到上），检查是否满足堆叠规则
    """
    if not stack:
        return True

    # 如果任意货物不可叠，直接失败
    if any(c.stackable == 'N' for c in stack):
        return False

    # 取顶层货物，决定规则
    top = stack[-1]

    # ---------- Crate 规则 ----------
    if top.package_type == 'Crate':
        suppliers = {c.supplier for c in stack}
        dims = {(round(c.length, 2), round(c.width, 2), round(c.height, 2)) for c in stack}

        # i. 供应商不同 或 尺寸不同 => 不允许叠
        if len(suppliers) > 1 or len(dims) > 1:
            return False

        # ii. 层数 ≤3
        if len(stack) > 3:
            return False

        # iii. 总高度 ≤200 cm
        total_height = sum(c.height for c in stack)
        if total_height > 200:
            return False

    # ---------- Box 规则 ----------
    elif top.package_type == 'Box':
        total_height = sum(c.height for c in stack)
        if total_height > 120:
            return False

    # ---------- 其他包装类型 ----------
    else:
        # 如果没有特别规则，则只遵守 stackable=Y
        return all(c.stackable == 'Y' for c in stack)

    return True


def build_stack_chain(cargo: Cargo, placed: List[Tuple], z0: float, x0: float, y0: float, l: float, w: float):
    """
    从当前位置向下递归收集完整堆叠链，返回从底到顶的货物列表
    """
    chain = [cargo]
    current_z = z0
    while True:
        found_below = None
        for px, py, pz, (pl, pw, ph), pc, pm in placed:
            if abs(pz + ph - current_z) < 1e-6:
                # 判断覆盖关系（必须在水平上有支撑）
                ox = max(0.0, min(x0 + l, px + pl) - max(x0, px))
                oy = max(0.0, min(y0 + w, py + pw) - max(y0, py))
                if ox * oy > 0:
                    found_below = (pc, px, py, pz, pl, pw, ph)
                    break
        if not found_below:
            break
        pc, px, py, pz, pl, pw, ph = found_below
        chain.insert(0, pc)  # 插到前面，保证顺序从下到上
        current_z = pz
        x0, y0, l, w = px, py, pl, pw
    return chain


def can_place_with_constraints(dims, pos, placed, container: Container, cargo: Cargo):
    l, w, h = dims
    x, y, z = pos
    # 1. 边界
    if x + l > container.length + 1e-9 or y + w > container.width + 1e-9 or z + h > container.height + 1e-9:
        return False

    # 2. 碰撞检测 + 不可叠货物检查
    for px, py, pz, (pl, pw, ph), pc, pm in placed:
        overlap_x = not (x + l <= px + 1e-9 or x >= px + pl - 1e-9)
        overlap_y = not (y + w <= py + 1e-9 or y >= py + pw - 1e-9)
        overlap_z = not (z + h <= pz + 1e-9 or z >= pz + ph - 1e-9)
        if overlap_x and overlap_y and overlap_z:
            return False
        if abs(pz + ph - z) < 1e-6 and overlap_x and overlap_y:
            if pc.stackable == 'N':
                return False

    # 3. 支撑 + 堆叠规则
    if z > 0:
        cargo_area = l * w
        support_area = 0.0
        supported = False
        for px, py, pz, (pl, pw, ph), pc, pm in placed:
            if abs(pz + ph - z) < 1e-6:
                ox = max(0.0, min(x + l, px + pl) - max(x, px))
                oy = max(0.0, min(y + w, py + pw) - max(y, py))
                support_area += ox * oy
                if ox * oy > 0:
                    supported = True
        # 必须有 >=90% 支撑
        if cargo_area > 0 and support_area < 0.90 * cargo_area - 1e-9:
            return False

        if supported:
            if cargo.stackable == 'N':
                return False
            # 收集完整堆叠链并检查
            full_stack = build_stack_chain(cargo, placed, z, x, y, l, w)
            if not check_stack_rules_on_stack(full_stack):
                return False

    return True


def update_extreme_points(extreme_points: List[Tuple[float, float, float]], used_point: Tuple[float, float, float],
                          dims: Tuple[float, float, float], placed: List[Tuple], container: Container):
    extreme_points = [p for p in extreme_points if p != used_point]
    x, y, z = used_point
    l, w, h = dims
    candidates = [(x + l, y, z), (x, y + w, z), (x, y, z + h)]
    # 加上一些以已放置箱体边缘对齐的点，增加放置候选
    for px, py, pz, (pl, pw, ph), pc, pm in placed:
        candidates.append((px + pl, y, z))
        candidates.append((x, py + pw, z))
        candidates.append((px, py, pz + ph))
    for p in candidates:
        px, py, pz = p
        if px >= container.length - 1e-9 or py >= container.width - 1e-9 or pz >= container.height - 1e-9:
            continue
        # 如果该点落在任何已放置箱体内部，则跳过
        inside = False
        for qx, qy, qz, (ql, qw, qh), qc, qm in placed:
            if (qx + 1e-9 < px < qx + ql - 1e-9 and
                    qy + 1e-9 < py < qy + qw - 1e-9 and
                    qz + 1e-9 < pz < qz + qh - 1e-9):
                inside = True
                break
        if not inside and p not in extreme_points:
            extreme_points.append(p)
    return extreme_points


def fork_face_of_mode(mode: str) -> List[str]:
    """
    根据放置 mode 返回该放置的'进叉面'集合
    规则（你给定）：
      - 放置 mode 'L' -> 载载长度为 L，进叉面为 W
      - 放置 mode 'W' -> 载载长度为 W，进叉面为 L
      - 放置 mode 'L&W' -> 进叉面可以是 L 或 W
    返回值为 ['L'] 或 ['W'] 或 ['L','W']
    """
    if mode == 'L':
        return ['W']
    elif mode == 'W':
        return ['L']
    else:
        return ['L', 'W']


def cluster_rows_by_y(placements, eps=1e-6):
    rows = []
    for x, y, z, (l, w, h), c, mode in placements:
        y0, y1 = y, y + w
        placed_flag = False
        for r in rows:
            if not (y1 <= r['y_min'] + eps or y0 >= r['y_max'] - eps):
                r['y_min'] = min(r['y_min'], y0)
                r['y_max'] = max(r['y_max'], y1)
                r['items'].append((x, y, z, (l, w, h), c, mode))
                placed_flag = True
                break
        if not placed_flag:
            rows.append({'y_min': y0, 'y_max': y1, 'items': [(x, y, z, (l, w, h), c, mode)]})
    return rows


def every_row_has_entry_face(placements) -> bool:
    """
    检查每行是否至少有一个货物的实际进叉面朝门
    门在 x=0，叉车沿 x 方向叉入
    """
    if not placements:
        return True

    rows = cluster_rows_by_y(placements)

    for r in rows:
        has_entry_face = False
        for x, y, z, (l, w, h), c, mode in r['items']:
            faces = fork_face_of_mode(mode)

            # 判断实际进叉面是否朝门
            for f in faces:
                if f == 'L' and l <= w:  # 货物长度沿 x 方向时进叉面为 L
                    has_entry_face = True
                    break
                elif f == 'W' and w <= l:  # 货物宽度沿 x 方向时进叉面为 W
                    has_entry_face = True
                    break
            if has_entry_face:
                break

        if not has_entry_face:
            return False

    return True


def compute_left_right_weight(placements, container: Container) -> Tuple[float, float]:
    mid = container.length / 2.0
    left = 0.0
    right = 0.0
    for x, y, z, (l, w, h), c, mode in placements:
        cx = x + l / 2.0
        if cx <= mid + 1e-9:
            left += c.gross_weight
        else:
            right += c.gross_weight
    return left, right


# -----------------------
# 极点放置器（EPP）：按组放置（绑定组）
# 返回 dict: placements, utilization, weight, used_volume, packed_ids, ok, note
# -----------------------
def pack_container_epp(groups: List[List[Cargo]], container: Container, allow_partial: bool = True):
    """
    groups: list of binding groups. 每个 group 内货物需要一同放入（如果可能）
    本函数尽量按 groups 顺序放置，若某个 group 无法放入则跳过（allow_partial True 时）。
    """
    placements = []  # (x,y,z,(l,w,h), cargo, mode)
    extreme_points = [(0.0, 0.0, 0.0)]
    used_volume = 0.0
    total_weight = 0.0
    packed_group_indices = set()

    row_entry_face = {}  # key=y区间 tuple(y_min,y_max), value=bool

    for gid, group in enumerate(groups):
        if gid in packed_group_indices:
            continue
        # 尝试将整个 group 放入容器（使用临时结构）
        temp_placements = [p for p in placements]
        temp_extreme = [p for p in extreme_points]
        group_weight = 0.0
        group_volume = 0.0
        success_group = True
        # group 内顺序按照前面排序决定
        for cargo in group:
            placed_flag = False
            # 载重检查（group 内累加）
            if total_weight + group_weight + cargo.gross_weight > container.max_weight + 1e-9:
                success_group = False
                break

            # 遍历极点与朝向
            temp_extreme.sort(key=lambda p: (p[0], p[1], p[2]))
            for ep in list(temp_extreme):
                x0, y0, z0 = ep

                for l, w, h, mode in cargo.orientations(container):
                    # ---- 检查是否进叉面朝门 ----
                    faces = fork_face_of_mode(mode)
                    row_key = None
                    for r in cluster_rows_by_y(temp_placements):
                        if r['y_min'] <= y0 <= r['y_max']:
                            row_key = (r['y_min'], r['y_max'])
                            break
                    if row_key and row_entry_face.get(row_key, False):
                        # 当前行已有朝向门货物，无需强制
                        pass
                    else:
                        # 尝试让该货物进叉面朝门
                        if 'L' in faces and l <= w:
                            # ok
                            pass
                        elif 'W' in faces and w <= l:
                            pass
                        else:
                            continue  # 无法朝门，尝试下一个朝向

                    if not can_place_with_constraints(
                            (l, w, h), (x0, y0, z0), temp_placements, container, cargo):
                        continue

                    # 通过校验 -> 放置在 temp 结构
                    temp_placements.append((x0, y0, z0, (l, w, h), cargo, mode))
                    temp_extreme = update_extreme_points(temp_extreme, (x0, y0, z0), (l, w, h), temp_placements,
                                                         container)
                    group_weight += cargo.gross_weight
                    group_volume += l * w * h
                    placed_flag = True
                    if row_key:
                        row_entry_face[row_key] = True
                    break
                if placed_flag:
                    break
            if not placed_flag:
                success_group = False
                break
        if success_group:
            placements = temp_placements
            extreme_points = temp_extreme
            used_volume += group_volume
            total_weight += group_weight
            packed_group_indices.add(gid)
        else:
            # 如果不允许部分放置，则直接返回失败
            if not allow_partial:
                packed_ids = [p[4].uid for p in placements]
                return {'placements': placements, 'utilization': used_volume / (container.volume() + 1e-12),
                        'weight': total_weight, 'used_volume': used_volume, 'packed_ids': packed_ids, 'ok': False}
            # allow_partial True 时跳过该 group（planner 会在下一个容器尝试）
            continue

    packed_ids = [p[4].uid for p in placements]

    # 进叉面 & 时效 & 左右平衡等全局校验
    # 1) entry-face rule
    # if not every_row_has_entry_face(placements):
    #     return {'placements': placements, 'utilization': used_volume / (container.volume() + 1e-12),
    #             'weight': total_weight, 'used_volume': used_volume, 'packed_ids': packed_ids, 'ok': False}
    # 2) time rules
    cargos_in_container = [p[4] for p in placements]
    if not validate_time_rules_for_container(cargos_in_container, transport_mode_hint=container.transport_mode):
        return {'placements': placements, 'utilization': used_volume / (container.volume() + 1e-12),
                'weight': total_weight, 'used_volume': used_volume, 'packed_ids': packed_ids, 'ok': False}
    # 3) left-right balance
    left, right = compute_left_right_weight(placements, container)
    if abs(left - right) > 0.05 * container.max_weight + 1e-9:
        return {'placements': placements, 'utilization': used_volume / (container.volume() + 1e-12),
                'weight': total_weight, 'used_volume': used_volume, 'packed_ids': packed_ids, 'ok': False}

    return {'placements': placements, 'utilization': used_volume / (container.volume() + 1e-12),
            'weight': total_weight, 'used_volume': used_volume, 'packed_ids': packed_ids, 'ok': True}


# -----------------------
# Planner：对四个桶分别求解，容器可无限开启
# -----------------------
def expand_cargos_bulk(cargos: List[Cargo]) -> List[Cargo]:
    out = []
    uid = 0
    for c in cargos:
        qty = max(1, c.quantity)
        for _ in range(qty):
            nc = Cargo(
                c.transport_mode, c.route, c.supplier, c.customer_order,
                c.length, c.width, c.height, 1,
                c.gross_weight / qty,
                c.order_time, c.package_type, c.stackable, c.load_dir,
                c.attachment_id, c.pallet_id, uid
            )
            out.append(nc)
            uid += 1
    return out


def plan_multi_containers(cargos: List[Cargo], container_types: List[Container],
                          max_container_instances_per_bucket: int = 100):
    expanded = expand_cargos_bulk(cargos)
    if not expanded:
        return []

    buckets = group_by_transport_and_route(expanded)
    all_solutions = []

    for bucket_key, items in buckets.items():
        if not items:
            continue

        # 选择容器类型
        if bucket_key == 'Ground':
            applicable = [c for c in container_types if c.transport_mode == 'Ground']
            transport_mode, route = 'Ground', None
        else:
            transport_mode = 'Ocean'
            route = bucket_key.split('-', 1)[1] if '-' in bucket_key else None
            applicable = [c for c in container_types
                          if c.transport_mode == 'Ocean' and (c.route is None or route is None or c.route == route)]

        if not applicable:
            all_solutions.append({'bucket': bucket_key, 'instances': [], 'note': 'no applicable container types'})
            continue

        groups = create_binding_groups(items)
        groups = sort_groups_for_packing(groups, transport_mode)
        unplaced_groups = groups[:]
        instances = []
        instances_count = 0
        safety = 0

        while unplaced_groups and instances_count < max_container_instances_per_bucket and safety < 2000:
            safety += 1
            chosen = max(applicable, key=lambda c: c.volume())

            # 尝试放置所有剩余组
            pack_result = pack_container_epp(unplaced_groups, chosen, allow_partial=True)
            if pack_result.get('ok', False) and pack_result.get('packed_ids'):
                packed_ids = set(pack_result.get('packed_ids'))
                remaining = [g for g in unplaced_groups if not all(c.uid in packed_ids for c in g)]
                instances.append({'container': chosen, 'result': pack_result})
                unplaced_groups = remaining
                instances_count += 1
                continue  # 成功放置直接下一轮

            # 尝试单独放第一个组
            first = unplaced_groups[0]
            placed_any = False
            for opt in applicable:
                r = pack_container_epp([first], opt, allow_partial=True)
                if r.get('ok', False) and r.get('packed_ids'):
                    instances.append({'container': opt, 'result': r})
                    unplaced_groups = unplaced_groups[1:]
                    placed_any = True
                    instances_count += 1
                    break

            if not placed_any:
                # 无法放置：标记为无法放置且不违反规则
                instances.append({'container': None, 'result': {
                    'placements': [],
                    'utilization': 0.0,
                    'weight': 0.0,
                    'used_volume': 0.0,
                    'packed_ids': [],
                    'ok': False,
                    'note': 'group cannot fit any applicable container (dim/weight/other rules)'
                }})
                unplaced_groups = unplaced_groups[1:]
                instances_count += 1

        all_solutions.append({'bucket': bucket_key, 'instances': instances, 'remaining_groups': unplaced_groups})

    return all_solutions


# -----------------------
# 可视化 (Plotly) 与 UI
# -----------------------
def create_cuboid_plot(x, y, z, dx, dy, dz, name, color):
    X = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    Y = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    Z = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    I = [0, 0, 0, 1, 1, 2, 4, 5, 6, 4, 6, 7]
    J = [1, 2, 4, 2, 5, 3, 5, 6, 7, 0, 7, 3]
    K = [2, 4, 5, 3, 6, 0, 6, 7, 4, 7, 3, 0]
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, opacity=0.8, color=color, name=name, hovertext=name,
                     hoverinfo='text')


def visualize_container_placements(res: Dict[str, Any], container: Container):
    placements = res.get('placements', [])
    fig = go.Figure()
    fig.add_trace(
        create_cuboid_plot(0, 0, 0, container.length, container.width, container.height, 'Container', 'rgba(0,0,0,0)'))

    # 根据包装类型和供应商分配颜色
    type_colors = {'Crate': (200, 50, 50), 'Pallet': (50, 200, 50), 'Box': (50, 50, 200)}
    colors = {}
    for idx, (x, y, z, (l, w, h), c, mode) in enumerate(placements):
        base_rgb = type_colors.get(c.package_type, (150, 150, 150))
        # 调整颜色亮度或偏移以区分供应商
        offset = (hash(c.supplier) % 50) - 25
        r = min(max(base_rgb[0] + offset, 0), 255)
        g = min(max(base_rgb[1] + offset, 0), 255)
        b = min(max(base_rgb[2] + offset, 0), 255)
        colors[c.uid] = f'rgb({r},{g},{b})'

        name = f"ID:{c.uid} Supp:{c.supplier} Pkg:{c.package_type} W:{c.gross_weight:.1f}kg Mode:{mode}"
        fig.add_trace(create_cuboid_plot(x, y, z, l, w, h, name, colors[c.uid]))

    fig.update_layout(scene=dict(xaxis=dict(title='Length'), yaxis=dict(title='Width'), zaxis=dict(title='Height'),
                                 aspectmode='data'))

    fig.update_layout(margin=dict(r=0, l=0, b=0, t=40))
    return fig


def default_demo_df():
    datetime.now()
    rows = [{'transport_mode': 'Ground', 'route': 'US', 'supplier': 'Rockwell', 'customer_order': '1', 'length': 0.5,
             'width': 0.5, 'height': 0.5, 'quantity': 1, 'gross_weight': 12, 'order_time': '20250907',
             'package_type': 'Crate', 'stackable': 'Y', 'load_dir': 'L&W', 'attachment_id': None, 'pallet_id': None},
            {'transport_mode': 'Ground', 'route': 'US', 'supplier': '1', 'customer_order': '1', 'length': 2.3,
             'width': 1, 'height': 0.5, 'quantity': 2, 'gross_weight': 12, 'order_time': '20250907',
             'package_type': 'Pallet', 'stackable': 'N', 'load_dir': 'L&W', 'attachment_id': None, 'pallet_id': None},
            {'transport_mode': 'Ground', 'route': 'US', 'supplier': '1', 'customer_order': '1', 'length': 2.3,
             'width': 1, 'height': 0.5, 'quantity': 2, 'gross_weight': 12, 'order_time': '20250907',
             'package_type': 'Box', 'stackable': 'Y', 'load_dir': 'L&W', 'attachment_id': None, 'pallet_id': None}
            ]
    # {'transport_mode': 'Ground', 'route': 'US', 'supplier': '3', 'customer_order': '3', 'length': 1.2,
    #  'width': 1.0, 'height': 0.5, 'quantity': 10, 'gross_weight': 20, 'order_time': '20250907',
    #  'package_type': 'Box', 'stackable': 'Y', 'load_dir': 'L', 'attachment_id': None, 'pallet_id': None},
    # {'transport_mode': 'Ground', 'route': 'US', 'supplier': '4', 'customer_order': '4', 'length': 1.2,
    #  'width': 1.0, 'height': 0.5, 'quantity': 10, 'gross_weight': 20, 'order_time': '20250907',
    #  'package_type': 'Box', 'stackable': 'Y', 'load_dir': 'L', 'attachment_id': None, 'pallet_id': None},
    # {'transport_mode': 'Ground', 'route': 'US', 'supplier': '5', 'customer_order': '5', 'length': 1.2,
    #  'width': 1.0, 'height': 0.5, 'quantity': 10, 'gross_weight': 20, 'order_time': '20250907',
    #  'package_type': 'Box', 'stackable': 'Y', 'load_dir': 'L', 'attachment_id': None, 'pallet_id': None},
    # {'transport_mode': 'Ground', 'route': 'US', 'supplier': '6', 'customer_order': '6', 'length': 1.2,
    #  'width': 1.0, 'height': 0.5, 'quantity': 10, 'gross_weight': 20, 'order_time': '20250907',
    #  'package_type': 'Box', 'stackable': 'Y', 'load_dir': 'L', 'attachment_id': None, 'pallet_id': None}]
    # {'transport_mode': 'Ground', 'route': 'US', 'supplier': 'FU', 'customer_order': 'CO125', 'length': 1.0,
    #  'width': 1.0, 'height': 0.6, 'quantity': 1, 'gross_weight': 120, 'order_time': '20250908',
    #  'package_type': 'Box', 'stackable': 'Y', 'load_dir': 'L&W', 'attachment_id': None, 'pallet_id': None}]
    # rows.append({'transport_mode':'Ground','route':'SG','supplier':'FUYAO','customer_order':'CO124','length':1.0,'width':1.0,'height':0.6,'quantity':1,'gross_weight':120,'order_time':'20250917','package_type':'Box','stackable':'Y','load_dir':'L&W','attachment_id':None,'pallet_id':None})
    # rows.append({'transport_mode':'Ground','route':'US','supplier':'FU','customer_order':'CO126','length':1.0,'width':1.0,'height':0.6,'quantity':1,'gross_weight':120,'order_time':'20250915','package_type':'Box','stackable':'Y','load_dir':'L&W','attachment_id':None,'pallet_id':None})
    # rows.append({'transport_mode':'Ground','route':'US','supplier':'FU','customer_order':'CO127','length':1.0,'width':1.0,'height':0.6,'quantity':1,'gross_weight':120,'order_time':'20250915','package_type':'Box','stackable':'Y','load_dir':'L&W','attachment_id':None,'pallet_id':None})

    return pd.DataFrame(rows)


def parse_df_to_cargos(df: pd.DataFrame) -> List[Cargo]:
    cargos = []
    uid = 0
    for idx, row in df.iterrows():
        try:
            ot = row.get('order_time')
            if isinstance(ot, str):
                order_time = datetime.fromisoformat(ot)
            elif isinstance(ot, pd.Timestamp):
                order_time = ot.to_pydatetime()
            elif ot is None or (isinstance(ot, float) and math.isnan(ot)):
                order_time = datetime.now()
            else:
                order_time = ot
            c = Cargo(
                transport_mode=str(row.get('transport_mode') or row.get('运输方式') or 'Ocean'),
                route=str(row.get('route') or row.get('路线') or 'MY'),
                supplier=str(row.get('supplier') or row.get('供应商') or ''),
                customer_order=str(row.get('customer_order') or row.get('客户单号') or ''),
                length=float(row.get('length') or row.get('长') or 1.0),
                width=float(row.get('width') or row.get('宽') or 1.0),
                height=float(row.get('height') or row.get('高') or 1.0),
                quantity=int(row.get('quantity') or row.get('件数') or 1),
                gross_weight=float(row.get('gross_weight') or row.get('毛重') or 10.0),
                order_time=order_time,
                package_type=str(row.get('package_type') or row.get('包装类型') or 'Box'),
                stackable=str(row.get('stackable') or row.get('是否堆叠') or 'N'),
                load_dir=str(row.get('load_dir') or row.get('装载方向') or 'L&W'),
                attachment_id=row.get('attachment_id') if 'attachment_id' in row else (
                    row.get('附件箱号/栈板号') if '附件箱号/栈板号' in row else None),
                pallet_id=row.get('pallet_id') if 'pallet_id' in row else None,
                uid=uid
            )
            cargos.append(c)
            uid += 1
        except Exception as e:
            st.error(f"解析第{idx}行货物时错误: {e}")
    return cargos


def app():
    st.set_page_config(page_title='智能装载 ', layout='wide')
    st.title('智能装载系统')

    # st.sidebar.header('容器参数')
    default_containers = [
        Container('45HC', 'Ground', None, 13.5, 2.34, 2.68, 25000, (0.3, 0.04, 0.18)),
        Container('20GP', 'Ocean', None, 5.85, 2.34, 2.38, 10000, (0.15, 0.04, 0.18)),
        Container('40GP', 'Ocean', None, 11.85, 2.34, 2.38, 20000, (0.15, 0.04, 0.18)),
        Container('40HC', 'Ocean', None, 11.85, 2.34, 2.68, 20000, (0.15, 0.04, 0.18)),
    ]
    names = [c.name for c in default_containers]
    # chosen = st.sidebar.multiselect('选择容器类型', names, default=names)
    container_types = [c for c in default_containers if c.name in names]

    # st.sidebar.header('算法参数')
    # max_instances = st.sidebar.slider('每桶最大容器实例数', 1, 200, 50)

    max_instances = 50

    st.subheader('货物输入（编辑或上传 CSV/XLSX）')
    df = st.data_editor(default_demo_df(), use_container_width=True, num_rows='dynamic')
    uploaded = st.file_uploader('上传 CSV/XLSX', type=['csv', 'xlsx'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            df = df_up
        except Exception as e:
            st.error(f'读取文件出错: {e}')

    cargos = parse_df_to_cargos(df)
    st.write(f"输入行数: {len(df)}。")

    if st.button('开始计算'):
        if not container_types:
            st.error('请至少选择一个容器类型')
            return
        t0 = time.time()
        sols = plan_multi_containers(cargos, container_types, max_container_instances_per_bucket=max_instances)
        t1 = time.time()
        if not sols:
            st.error('未生成任何方案')
            return

        for sol in sols:
            st.header(f"Bucket: {sol['bucket']}")
            for idx, inst in enumerate(sol['instances']):
                cont = inst.get('container')
                res = inst.get('result', {})

                # 如果该实例违反规则，不显示货物，只显示警告
                if not res.get('ok', False):
                    st.subheader(f"实例 {idx + 1}: {cont.name if cont else 'N/A'} (无法放置)")
                    st.warning(res.get('note', '该组无法放入任何容器'))
                    continue

                st.subheader(f"实例 {idx + 1}: {cont.name if cont else 'N/A'}")
                st.write(
                    f"利用率: {res.get('utilization', 0.0) * 100:.2f}%"
                    f" 载重: {res.get('weight', 0.0):.2f}/{cont.max_weight if cont else 'N/A'}")
                st.write(f"装载件数: {len(res.get('placements', []))}")

                if res.get('placements'):
                    fig = visualize_container_placements(res, cont)
                    st.plotly_chart(fig, use_container_width=True)
            if sol.get('remaining_groups'):
                st.warning(f"Bucket {sol['bucket']} 剩余未放置组数: {len(sol['remaining_groups'])}")
        st.success(f'完成，总耗时 {t1 - t0:.1f} 秒')


if __name__ == '__main__':
    app()
