# smart_packing_streamlit.py
# streamlit run smart_packing_streamlit.py
import copy
import logging
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

logging.basicConfig(
    level=logging.DEBUG,  # 可根据需要调整为 INFO 或 WARNING
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
    - Ocean CN - US -> 每周二 ~ 下周一 为一个时间段
    - Ocean CN - SG -> (SG_A: 周三~周五, SG_B: 周六~下周二)
    - Ocean CN - MY -> 没有额外限制，用 iso_week
    """
    dow = dt.weekday()  # Monday=0, Sunday=6

    if transport_mode == 'Ocean':
        if route.upper() == 'CN - US':
            # 以周二为 anchor
            base_monday = dt - timedelta(days=dow)  # 本周一
            base_tuesday = base_monday + timedelta(days=1)  # 本周二
            if dt >= base_tuesday:
                anchor = base_tuesday
            else:
                anchor = base_tuesday - timedelta(days=7)
            return 'US', anchor.strftime('%Y-%m-%d')

        elif route.upper() == 'CN - SG':
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

        elif route.upper() == 'CN - MY':
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
            if r == 'CN - US':
                buckets['Ocean-US'].append(c)
            elif r == 'CN - SG':
                buckets['Ocean-SG'].append(c)
            elif r == 'CN - MY':
                buckets['Ocean-MY'].append(c)
            else:
                buckets['Ocean-MY'].append(c)
        else:
            buckets['Ground'].append(c)
    return buckets


def create_binding_groups(cargos: List[Cargo]) -> List[List[Cargo]]:
    """正确的分组策略：只有相同附件单号或客户单号的货物才绑定在一起"""
    binding_groups = {}
    single_items = []

    for c in cargos:
        # 优先按附件单号分组
        if c.attachment_id and c.attachment_id.strip() and str(c.attachment_id).lower() != 'nan':
            key = f"ATT_{c.attachment_id}"
            if key not in binding_groups:
                binding_groups[key] = []
            binding_groups[key].append(c)
        # 其次按客户单号分组
        elif c.customer_order and c.customer_order.strip() and str(c.customer_order).lower() != 'nan':
            key = f"CO_{c.customer_order}"
            if key not in binding_groups:
                binding_groups[key] = []
            binding_groups[key].append(c)
        # 没有绑定关系的单独成组
        else:
            single_items.append([c])

    # 合并所有组
    groups = [copy.deepcopy(group) for group in list(binding_groups.values())] + [copy.deepcopy(group) for group in
                                                                                  single_items]

    # 验证分组结果
    logging.info(f"分组结果: 共 {len(groups)} 个组")
    for i, group in enumerate(groups):
        logging.info(f"组 {i}: {len(group)} 个货物")
        if len(group) > 1:
            # 检查绑定关系是否一致
            if group[0].attachment_id and str(group[0].attachment_id).lower() != 'nan':
                logging.info(f"  绑定方式: 附件单号 {group[0].attachment_id}")
            elif group[0].customer_order and str(group[0].customer_order).lower() != 'nan':
                logging.info(f"  绑定方式: 客户单号 {group[0].customer_order}")

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
    - Ocean-CN - SG:
        i. SG_A 段：周三~周五
        ii. SG_B 段：周六~下周二
        同一容器不能跨段
    - Ocean-MY: 无时效限制
    """

    logging.info(f"时间规则验证: 货物数量={len(cargos)}, 运输方式={transport_mode_hint}")
    if not cargos:
        return True

    # 确认运输方式
    modes = set(c.transport_mode for c in cargos)
    if transport_mode_hint:
        if any(m != transport_mode_hint for m in modes):
            logging.warning("运输方式不匹配")
            return False
    transport_mode = transport_mode_hint if transport_mode_hint else cargos[0].transport_mode

    # ============ Ground ============ #
    if transport_mode == 'Ground':
        times = [c.order_time for c in cargos]
        if not times:
            return True

        # i. 时间跨度 ≤ 2 天
        if (max(times) - min(times)).days > 2:
            logging.warning("Ground 运输方式时间跨度超过2天")
            return False

        # ii. FUYAO/FOXSEMICON 特殊供应商规则
        suppliers = set(c.supplier for c in cargos)
        if 'FUYAO' in suppliers or 'FOXSEMICON' in suppliers:
            special_times = [c.order_time for c in cargos if c.supplier in ('FUYAO', 'FOXSEMICON')]
            if special_times:
                min_sp = min(special_times)
                for c in cargos:
                    if c.supplier not in ():
                        if (c.order_time - min_sp).days > 1:
                            logging.warning("非特殊供应商订单时间晚于限制")
                            return False

        # iii. 不允许跨 ISO 周
        weeks = set((t.isocalendar()[0], t.isocalendar()[1]) for t in times)
        if len(weeks) > 1:
            logging.warning("Ground 运输方式跨 ISO 周")
            return False

        return True

    # ============ Ocean ============ #
    elif transport_mode == 'Ocean':
        routes = set((c.route or '').upper() for c in cargos)
        if len(routes) > 1:
            logging.warning("Ocean 运输方式存在多个路线")
            return False
        route = next(iter(routes))

        if route == 'MY':
            return True  # 无规则，直接合法

        # US/SG 依赖 week_unit_for_route
        units = set(week_unit_for_route(route, 'Ocean', c.order_time) for c in cargos)
        if len(units) > 1:
            logging.warning("Ocean 运输方式跨时间段")
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

    logging.debug(f"堆叠检查: 堆叠链长度={len(stack)}")
    for i, c in enumerate(stack):
        logging.debug(f"  层 {i}: {c.package_type}, 供应商={c.supplier}, 可堆叠={c.stackable}")
    if not stack:
        return True

    # 如果任意货物不可叠，直接失败
    if any(c.stackable == 'N' for c in stack):
        logging.warning("存在不可堆叠货物")
        return False

        # ---------- Box 不可叠在 Crate 上 ----------
    for i in range(1, len(stack)):
        lower = stack[i - 1]
        upper = stack[i]
        if lower.package_type == 'Crate' and upper.package_type == 'Box':
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
        if total_height > 2.0:
            return False

    # ---------- Box 规则 ----------
    elif top.package_type == 'Box':
        total_height = sum(c.height for c in stack)
        if total_height > 1.2:
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


def can_place_with_constraints(dims, pos, placed, container: Container, cargo: Cargo, mode: str = None):
    l, w, h = dims
    x, y, z = pos

    logging.info(f"检查放置: 位置({x}, {y}, {z}), 尺寸({l}, {w}, {h}), 模式({mode})")

    # 根据模式确定实际占用空间（使用不同的变量名）
    if mode == 'L':
        actual_l, actual_w, actual_h = l, w, h
    elif mode == 'W':
        actual_l, actual_w, actual_h = w, l, h
    else:
        actual_l, actual_w, actual_h = l, w, h

    # 1. 边界检查（使用实际尺寸）
    # 添加负值检查，增大容差
    if (x < -1e-6 or y < -1e-6 or z < -1e-6 or
            x + actual_l > container.length + 1e-6 or
            y + actual_w > container.width + 1e-6 or
            z + actual_h > container.height + 1e-6):
        logging.info(
            f"边界检查失败: 位置({x:.2f},{y:.2f},{z:.2f}) 实际尺寸({actual_l:.2f},{actual_w:.2f},{actual_h:.2f})")
        return False

    # 2. 碰撞检测 + 不可叠货物检查
    for placed_x, placed_y, placed_z, (placed_l, placed_w, placed_h), placed_cargo, placed_mode in placed:
        # 根据放置模式确定已放置货物的实际尺寸
        if placed_mode == 'L':
            placed_actual_l, placed_actual_w, placed_actual_h = placed_l, placed_w, placed_h
        elif placed_mode == 'W':
            placed_actual_l, placed_actual_w, placed_actual_h = placed_w, placed_l, placed_h
        else:
            placed_actual_l, placed_actual_w, placed_actual_h = placed_l, placed_w, placed_h

        # 碰撞检测
        overlap_x = min(x + actual_l, placed_x + placed_actual_l) - max(x, placed_x)
        overlap_y = min(y + actual_w, placed_y + placed_actual_w) - max(y, placed_y)
        overlap_z = min(z + actual_h, placed_z + placed_actual_h) - max(z, placed_z)

        if overlap_x > 1e-6 and overlap_y > 1e-6 and overlap_z > 1e-6:
            logging.info(
                f"碰撞检测失败: 与货物{placed_cargo.uid}在位置({placed_x:.2f},{placed_y:.2f},{placed_z:.2f})碰撞")
            return False

    # 3. 支撑 + 堆叠规则（使用实际尺寸）
    if z > 0:
        cargo_area = actual_l * actual_w  # 使用实际底面积
        support_area = 0.0
        supported = False

        for placed_x, placed_y, placed_z, (placed_l, placed_w, placed_h), placed_cargo, placed_mode in placed:
            # 根据放置模式确定已放置货物的实际尺寸
            if placed_mode == 'L':
                placed_actual_l, placed_actual_w, placed_actual_h = placed_l, placed_w, placed_h
            elif placed_mode == 'W':
                placed_actual_l, placed_actual_w, placed_actual_h = placed_w, placed_l, placed_h
            else:
                placed_actual_l, placed_actual_w, placed_actual_h = placed_l, placed_w, placed_h

            # 检查支撑
            if abs(placed_z + placed_actual_h - z) < 1e-6:
                ox = max(0.0, min(x + actual_l, placed_x + placed_actual_l) - max(x, placed_x))
                oy = max(0.0, min(y + actual_w, placed_y + placed_actual_w) - max(y, placed_y))
                support_area += ox * oy
                if ox * oy > 0:
                    supported = True

        # 必须有足够的支撑面积
        if cargo_area > 0 and support_area < 0.80 * cargo_area - 1e-9:
            logging.info(f"支撑不足: 需要{cargo_area * 0.8:.2f}, 实际{support_area:.2f}")
            return False

        if supported:
            if cargo.stackable == 'N':
                logging.info("不可堆叠货物尝试堆叠")
                return False
            # 收集完整堆叠链并检查
            full_stack = build_stack_chain(cargo, placed, z, x, y, actual_l, actual_w)
            if not check_stack_rules_on_stack(full_stack):
                logging.info("堆叠规则检查失败")
                return False

    return True


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
                c.attachment_id, uid
            )
            out.append(nc)
            uid += 1
    return out


def extract_extreme_points(placements: List, container: 'Container'):
    """
    根据已有 placements 生成极点列表，支持模式系统
    """
    if not placements:
        logging.info("空容器，返回原点极点")
        return [(0.0, 0.0, 0.0)]

    points = []
    logging.info(f"处理 {len(placements)} 个已有放置")

    for placement in placements:
        x, y, z, (l, w, h), cargo, mode = placement

        # 根据模式确定实际尺寸方向
        if mode == 'L':  # 长度对应X轴，宽度对应Y轴
            dx, dy, dz = l, w, h
        elif mode == 'W':  # 宽度对应X轴，长度对应Y轴
            dx, dy, dz = w, l, h
        elif mode == 'L&W':  # 默认使用L模式
            dx, dy, dz = l, w, h
        else:  # 未知模式，使用默认
            dx, dy, dz = l, w, h

        # 生成边界点
        boundary_points = [
            # 基础边界点
            (x + dx, y, z),  # X方向末端
            (x, y + dy, z),  # Y方向末端
            (x, y, z + dz),  # Z方向末端
            (x + dx, y + dy, z),  # XY平面角
            (x + dx, y, z + dz),  # XZ平面角
            (x, y + dy, z + dz),  # YZ平面角
            (x + dx, y + dy, z + dz),  # 对角点

            # 贴着容器边缘的点
            (x + dx, 0, z),  # X方向末端，贴着左边缘
            (x + dx, container.width, z),  # X方向末端，贴着右边缘
            (0, y + dy, z),  # Y方向末端，贴着重边缘
            (container.length, y + dy, z),  # Y方向末端，贴着后边缘
            (x + dx, 0, z + dz),  # XZ平面角，贴着左边缘
            (x + dx, container.width, z + dz),  # XZ平面角，贴着右边缘
            (0, y + dy, z + dz),  # YZ平面角，贴着重边缘
            (container.length, y + dy, z + dz),  # YZ平面角，贴着后边缘

            # 贴着底部的点
            (x + dx, y, 0),  # X方向末端，贴着底部
            (x, y + dy, 0),  # Y方向末端，贴着底部
            (x + dx, y + dy, 0),  # XY平面角，贴着底部
        ]

        # 添加有效边界点
        for px, py, pz in boundary_points:
            if (0 <= px <= container.length + 1e-9 and
                    0 <= py <= container.width + 1e-9 and
                    0 <= pz <= container.height + 1e-9):
                points.append((round(px, 6), round(py, 6), round(pz, 6)))

    # 确保包含原点
    if not any(p[0] == 0 and p[1] == 0 and p[2] == 0 for p in points):
        points.append((0.0, 0.0, 0.0))

    # 去重并排序
    unique_points = list(set(points))  # 使用集合去重
    unique_points.sort(key=lambda p: (-p[2], p[0], p[1]))
    logging.info(f'当前极点集{unique_points}')
    return unique_points


def update_extreme_points(extreme_points: List[Tuple[float, float, float]],
                          used_point: Tuple[float, float, float],
                          dims: Tuple[float, float, float],
                          placed: List[Tuple],
                          container: 'Container',
                          mode: str) -> List[Tuple[float, float, float]]:
    """
    更新极点列表 - 带详细调试信息
    """
    logging.info(f"\n=== 开始更新极点 ===")
    logging.info(f"输入极点: {extreme_points}")
    logging.info(f"使用点: {used_point}")
    logging.info(f"货物尺寸: {dims}")
    logging.info(f"模式: {mode}")
    logging.info(f"已有放置数量: {len(placed)}")

    # 创建新的极点列表副本（避免修改原列表）
    new_extreme = list(extreme_points)

    # 移除已使用的极点
    if used_point in new_extreme:
        new_extreme.remove(used_point)
        logging.info(f"✓ 移除已使用极点: {used_point}")
    else:
        logging.info(f"⚠ 极点 {used_point} 不在列表中")

    x0, y0, z0 = used_point
    l, w, h = dims

    # 根据模式确定实际尺寸方向
    if mode == 'L':
        dx, dy, dz = l, w, h
        logging.info(f"模式L: 尺寸({dx}, {dy}, {dz})")
    elif mode == 'W':
        dx, dy, dz = w, l, h
        logging.info(f"模式W: 尺寸({dx}, {dy}, {dz})")
    elif mode == 'L&W':
        dx, dy, dz = l, w, h
        logging.info(f"模式L&W: 尺寸({dx}, {dy}, {dz})")
    else:
        dx, dy, dz = l, w, h
        logging.info(f"未知模式: 尺寸({dx}, {dy}, {dz})")

    # 添加新边界点
    new_candidates = [
        # 基础边界点
        (x0 + dx, y0, z0),  # X方向末端
        (x0, y0 + dy, z0),  # Y方向末端
        (x0, y0, z0 + dz),  # Z方向末端
        (x0 + dx, y0 + dy, z0),  # XY平面角
        (x0 + dx, y0, z0 + dz),  # XZ平面角
        (x0, y0 + dy, z0 + dz),  # YZ平面角
        (x0 + dx, y0 + dy, z0 + dz),  # 对角点

        # 贴着容器边缘的点
        (x0 + dx, 0, z0),  # X方向末端，贴着左边缘
        (x0 + dx, container.width, z0),  # X方向末端，贴着右边缘
        (0, y0 + dy, z0),  # Y方向末端，贴着重边缘
        (container.length, y0 + dy, z0),  # Y方向末端，贴着后边缘
        (x0 + dx, 0, z0 + dz),  # XZ平面角，贴着左边缘
        (x0 + dx, container.width, z0 + dz),  # XZ平面角，贴着右边缘
        (0, y0 + dy, z0 + dz),  # YZ平面角，贴着重边缘
        (container.length, y0 + dy, z0 + dz),  # YZ平面角，贴着后边缘

        # 贴着底部的点
        (x0 + dx, y0, 0),  # X方向末端，贴着底部
        (x0, y0 + dy, 0),  # Y方向末端，贴着底部
        (x0 + dx, y0 + dy, 0),  # XY平面角，贴着底部
    ]

    logging.info(f"候选边界点: {new_candidates}")

    # 验证并添加新极点
    added_count = 0
    for candidate in new_candidates:
        cand_x, cand_y, cand_z = candidate

        # 检查边界
        if not (0 <= cand_x <= container.length + 1e-9 and
                0 <= cand_y <= container.width + 1e-9 and
                0 <= cand_z <= container.height + 1e-9):
            logging.info(f"✗ 跳过边界外点: {candidate} (容器: {container.length}x{container.width}x{container.height})")
            continue

        # 检查是否被阻挡
        blocked = False
        blocking_cargo = None

        for placement in placed:
            placed_x, placed_y, placed_z, (placed_l, placed_w, placed_h), placed_cargo, placed_mode = placement

            # 跳过当前正在放置的货物（自己阻挡自己）
            if (abs(placed_x - x0) < 1e-9 and
                    abs(placed_y - y0) < 1e-9 and
                    abs(placed_z - z0) < 1e-9):
                continue

            # 根据模式确定实际尺寸
            if placed_mode == 'L':
                placed_dx, placed_dy, placed_dz = placed_l, placed_w, placed_h
            elif placed_mode == 'W':
                placed_dx, placed_dy, placed_dz = placed_w, placed_l, placed_h
            else:
                placed_dx, placed_dy, placed_dz = placed_l, placed_w, placed_h

            # 检查候选点是否在已有货物内部
            if (placed_x <= cand_x <= placed_x + placed_dx and
                    placed_y <= cand_y <= placed_y + placed_dy and
                    placed_z <= cand_z <= placed_z + placed_dz):
                blocked = True
                blocking_cargo = placed_cargo
                break

        if not blocked:
            if candidate not in new_extreme:
                new_extreme.append(candidate)
                added_count += 1
                logging.info(f"✓ 添加新极点: {candidate}")
            else:
                logging.info(f"○ 极点已存在: {candidate}")
        else:
            logging.info(f"✗ 点被阻挡: {candidate} (被货物 {blocking_cargo.uid if blocking_cargo else '未知'} 阻挡)")

    logging.info(f"添加了 {added_count} 个新极点")

    # 确保极点列表不为空
    if not new_extreme:
        logging.info("⚠ 警告: 极点列表为空，添加原点")
        new_extreme.append((0.0, 0.0, 0.0))

    # 重新排序（Z降序，X升序，Y升序）
    new_extreme.sort(key=lambda p: (-p[2], p[0], p[1]))

    logging.info(f"最终极点: {new_extreme}")
    logging.info("=== 极点更新完成 ===\n")

    return new_extreme


def pack_container_epp(groups: List[List['Cargo']], container: 'Container', allow_partial: bool = False,
                       existing_placements: List = None):
    """
    EPP 算法 - 完整修复版
    """
    logging.info(f"开始装载到容器 {container.name}, 已有放置: {len(existing_placements or [])}")
    placements = list(existing_placements) if existing_placements else []
    extreme_points = extract_extreme_points(placements, container) if placements else [(0.0, 0.0, 0.0)]
    used_volume = sum(p[3][0] * p[3][1] * p[3][2] for p in placements)
    total_weight = sum(p[4].gross_weight for p in placements)

    # 记录原始组信息用于验证
    original_cargo_ids = set()
    for group in groups:
        for cargo in group:
            original_cargo_ids.add(cargo.uid)

    all_groups_success = True

    for gid, group in enumerate(groups):
        logging.info(f"处理组 {gid}: {len(group)} 个货物")

        # 使用局部变量避免引用问题
        temp_placements = list(placements)
        temp_extreme = list(extreme_points)
        group_weight = 0.0
        group_volume = 0.0
        group_success = True

        for cargo in group:
            logging.debug(
                f"  货物 {cargo.uid}: {cargo.package_type}, 尺寸: {cargo.length}x{cargo.width}x{cargo.height}")
            logging.debug(f"  当前极点: {temp_extreme}")
            logging.debug(f"  当前放置数量: {len(temp_placements)}")

            placed_flag = False

            # 超重检查
            current_total_weight = sum(p[4].gross_weight for p in temp_placements) + cargo.gross_weight
            if current_total_weight > container.max_weight + 1e-9:
                logging.warning(f"超重警告: 货物 {cargo.uid} 超出容器最大重量限制")
                group_success = False
                break

            # 极点按 Z,X,Y 排序（原点优先）
            temp_extreme.sort(key=lambda p: (p != (0.0, 0.0, 0.0), -p[2], p[0], p[1]))

            # 1) 如果是空容器，优先尝试原点放置
            if not temp_placements:
                logging.info(f"空容器，尝试原点放置...")
                for l, w, h, mode in cargo.orientations(container):
                    # 根据模式确定实际尺寸
                    if mode == 'L':
                        actual_l, actual_w, actual_h = l, w, h
                    elif mode == 'W':
                        actual_l, actual_w, actual_h = w, l, h
                    else:
                        actual_l, actual_w, actual_h = l, w, h

                    # 检查边界
                    if (actual_l <= container.length + 1e-9 and
                            actual_w <= container.width + 1e-9 and
                            actual_h <= container.height + 1e-9):

                        if can_place_with_constraints((l, w, h), (0.0, 0.0, 0.0), temp_placements, container, cargo,
                                                      mode):
                            temp_placements.append((0.0, 0.0, 0.0, (l, w, h), cargo, mode))
                            group_weight += cargo.gross_weight
                            group_volume += l * w * h
                            placed_flag = True

                            # 更新极点
                            updated_extreme = update_extreme_points(temp_extreme, (0.0, 0.0, 0.0), (l, w, h),
                                                                    temp_placements, container, mode)
                            temp_extreme = updated_extreme  # 重要：接收返回值
                            logging.info(f"空容器放置成功，新极点: {temp_extreme}")

                            break

                if placed_flag:
                    continue  # 跳过后续极点检查

            # 2) 检查所有极点
            for ep in list(temp_extreme):
                x0, y0, z0 = ep
                logging.info(f"检查极点: {ep}")

                # 禁止在非空容器回退放到原点
                if (x0, y0, z0) == (0.0, 0.0, 0.0) and temp_placements:
                    logging.info("    跳过原点（非空容器）")
                    continue

                stacked = False
                is_crate = str(cargo.package_type).strip().lower() == 'crate'
                is_stackable = cargo.stackable == 'Y'

                # 2.1) crate 优先堆叠
                if is_crate and is_stackable and temp_placements:
                    logging.info("    尝试堆叠放置...")
                    for bx, by, bz, (bl, bw, bh), base_c, base_mode in temp_placements:
                        if abs(x0 - bx) < 1e-3 and abs(y0 - by) < 1e-3 and abs(z0 - (bz + bh)) < 1e-3:
                            if cargo.supplier != base_c.supplier:
                                continue
                            for l, w, h, mode in cargo.orientations(container):
                                if abs(l - bl) < 1e-2 and abs(w - bw) < 1e-2:
                                    if z0 + h < container.height and can_place_with_constraints((l, w, h),
                                                                                                (x0, y0, z0),
                                                                                                temp_placements,
                                                                                                container,
                                                                                                cargo,
                                                                                                mode):
                                        temp_placements.append((x0, y0, z0, (l, w, h), cargo, mode))
                                        group_weight += cargo.gross_weight
                                        group_volume += l * w * h
                                        placed_flag = True
                                        stacked = True

                                        # 更新极点
                                        updated_extreme = update_extreme_points(temp_extreme, (x0, y0, z0), (l, w, h),
                                                                                temp_placements, container, mode)
                                        temp_extreme = updated_extreme
                                        logging.info(f"堆叠放置成功，新极点: {temp_extreme}")
                                        break
                            if stacked:
                                break
                    if stacked:
                        break

                # 2.2) 普通平铺放置
                if not placed_flag:
                    logging.info("    尝试普通平铺放置...")
                    for l, w, h, mode in cargo.orientations(container):
                        if can_place_with_constraints((l, w, h), (x0, y0, z0), temp_placements, container, cargo, mode):
                            temp_placements.append((x0, y0, z0, (l, w, h), cargo, mode))
                            group_weight += cargo.gross_weight
                            group_volume += l * w * h
                            placed_flag = True

                            # 更新极点
                            updated_extreme = update_extreme_points(temp_extreme, (x0, y0, z0), (l, w, h),
                                                                    temp_placements, container, mode)
                            temp_extreme = updated_extreme
                            logging.info(f"平铺放置成功，新极点: {temp_extreme}")
                            break

                if placed_flag:
                    break

            if not placed_flag:
                logging.info(f"货物 {cargo.uid} 无法放置")
                group_success = False
                break

        if group_success:
            placements = temp_placements
            extreme_points = temp_extreme
            total_weight += group_weight
            used_volume += group_volume
            logging.info(f"组 {gid} 成功放置，最终极点: {extreme_points}")
        else:
            if allow_partial:
                logging.info(f"警告: 组 {gid} 无法完全放置，允许部分放置")
                placements = temp_placements
                extreme_points = temp_extreme
                total_weight += group_weight
                used_volume += group_volume
            else:
                logging.info(f"错误: {gid} 无法完全放置，返回失败")
                all_groups_success = False
                break

    # 数据一致性验证
    placed_cargo_ids = {p[4].uid for p in placements}
    missing_ids = original_cargo_ids - placed_cargo_ids

    if not allow_partial and missing_ids:
        logging.info(f"严重错误: 数据不一致! 缺失的货物: {missing_ids}")
        return {
            'placements': placements,
            'utilization': used_volume / (container.volume() + 1e-12),
            'weight': total_weight,
            'used_volume': used_volume,
            'pack_ids': list(placed_cargo_ids),
            'ok': False,
            'note': f'data inconsistency: expected {len(original_cargo_ids)}, placed {len(placed_cargo_ids)}'
        }

    # 最终校验
    packed_ids = [p[4].uid for p in placements]
    cargos_in_container = [p[4] for p in placements]

    if placements and not validate_time_rules_for_container(cargos_in_container,
                                                            transport_mode_hint=container.transport_mode):
        return {'placements': placements,
                'utilization': used_volume / container.volume(),
                'weight': total_weight, 'used_volume': used_volume,
                'packed_ids': packed_ids, 'ok': False, 'note': 'time rules violated'}

    left, right = compute_left_right_weight(placements, container)
    balance_ok = abs(left - right) <= container.max_weight * 0.1  # 10% 容差

    return {
        'placements': placements,
        'utilization': used_volume / container.volume(),
        'weight': total_weight,
        'used_volume': used_volume,
        'packed_ids': packed_ids,
        'ok': all_groups_success,
        'left_right': (left, right),
        'balance_ok': balance_ok
    }


def plan_multi_containers(cargos: List[Cargo], container_types: List[Container]):
    """
    修复版多容器装载调度 - 确保新创建的容器能被后续组使用
    1. 组不允许拆分
    2. 组合并不违反规则
    3. 最大化容器利用率
    4. 确保所有货物都有容器
    """
    expanded = expand_cargos_bulk(cargos)
    if not expanded:
        return []

    # 正确的分组
    groups = create_binding_groups(expanded)
    logging.info(f"总组数: {len(groups)}")

    # 按运输方式分桶
    buckets = {}
    for group in groups:
        if group:
            transport_mode = group[0].transport_mode
            route = group[0].route
            bucket_key = 'Ground' if transport_mode == 'Ground' else f'Ocean-{route}' if route else 'Ocean-MY'

            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(copy.deepcopy(group))

    all_solutions = []

    for bucket_key, bucket_groups in buckets.items():
        logging.info(f"处理Bucket: {bucket_key}, 组数量: {len(bucket_groups)}")

        # 选择适用的容器类型
        applicable = sorted([c for c in container_types if (
                (bucket_key == 'Ground' and c.transport_mode == 'Ground') or
                (bucket_key.startswith('Ocean') and c.transport_mode == 'Ocean')
        )], key=lambda x: x.volume(), reverse=True)

        if not applicable:
            continue

        instances = []
        remaining_groups = copy.deepcopy(bucket_groups)

        # 对组进行排序（数量多的优先，体积大的优先）
        remaining_groups.sort(key=lambda g: (-len(g), -sum(c.volume() for c in g)))

        # 使用单个循环按顺序处理每个组
        i = 0
        while i < len(remaining_groups):
            group = remaining_groups[i]
            placed = False
            logging.info(f"\n处理组 {i}: {len(group)} 个货物")

            # 首先尝试放入现有容器（按利用率从低到高排序）
            if instances:
                instances.sort(key=lambda inst: inst['result']['utilization'])

                for inst_idx, inst in enumerate(instances):
                    logging.info(f"  尝试放入容器 {inst_idx}: {inst['container'].name}, "
                                 f"利用率: {inst['result']['utilization'] * 100:.1f}%")

                    # 检查时间规则兼容性
                    existing_cargos = [p[4] for p in inst['result']['placements']]
                    test_cargos = existing_cargos + group

                    if not validate_time_rules_for_container(test_cargos, inst['container'].transport_mode):
                        logging.info(f"时间规则不兼容，跳过容器 {inst['container'].name}")
                        continue

                    # 检查重量限制
                    total_weight = inst['result']['weight'] + sum(c.gross_weight for c in group)
                    if total_weight > inst['container'].max_weight:
                        logging.info(f"重量超限 {total_weight:.1f}/{inst['container'].max_weight:.1f}，跳过容器")
                        continue

                    # 尝试放置
                    logging.info(f"    尝试放置到容器 {inst['container'].name}...")
                    result = pack_container_epp(
                        [copy.deepcopy(group)],
                        inst['container'],
                        allow_partial=False,
                        existing_placements=copy.deepcopy(inst['result']['placements'])
                    )

                    if result['ok']:
                        # 验证数据一致性
                        placed_ids = {p[4].uid for p in result['placements']}
                        group_ids = {c.uid for c in group}

                        if group_ids.issubset(placed_ids):
                            inst['result'] = result
                            inst['groups'].append(copy.deepcopy(group))
                            logging.info(
                                f"组成功放入现有容器 {inst['container'].name}, 利用率: {result['utilization'] * 100:.1f}%")
                            placed = True
                            break
                        else:
                            logging.info(f"数据验证失败（新组未能全部放入），继续尝试其他容器")
                    else:
                        logging.info(f"放置失败，继续尝试其他容器")

            # 如果无法放入任何现有容器，创建新容器
            if not placed:
                logging.info(f"  无法放入现有容器，创建新容器...")
                largest_container = applicable[0]

                # 检查重量限制
                total_weight = sum(c.gross_weight for c in group)
                if total_weight > largest_container.max_weight:
                    logging.info(f"警告: 组总重量 {total_weight} 超过容器最大载重 {largest_container.max_weight}")
                    # 仍然创建容器但标记失败
                    instances.append({
                        'container': largest_container,
                        'groups': [copy.deepcopy(group)],
                        'result': {
                            'placements': [],
                            'utilization': 0,
                            'weight': 0,
                            'used_volume': 0,
                            'packed_ids': [],
                            'ok': False,
                            'note': f'weight exceeded: {total_weight}/{largest_container.max_weight}'
                        }
                    })
                    placed = True
                else:
                    # 检查体积限制
                    total_volume = sum(c.volume() for c in group)
                    if total_volume > largest_container.volume():
                        logging.info(f"警告: 组总体积 {total_volume} 超过容器容积 {largest_container.volume()}")
                        # 仍然创建容器但标记失败
                        instances.append({
                            'container': largest_container,
                            'groups': [copy.deepcopy(group)],
                            'result': {
                                'placements': [],
                                'utilization': 0,
                                'weight': 0,
                                'used_volume': 0,
                                'packed_ids': [],
                                'ok': False,
                                'note': f'volume exceeded: {total_volume}/{largest_container.volume()}'
                            }
                        })
                        placed = True
                    else:
                        # 尝试放置到新容器
                        result = pack_container_epp(
                            [copy.deepcopy(group)],
                            largest_container,
                            allow_partial=False
                        )

                        if result['ok']:
                            instances.append({
                                'container': largest_container,
                                'groups': [copy.deepcopy(group)],
                                'result': result
                            })
                            logging.info(f"创建新容器放置组, 利用率: {result['utilization'] * 100:.1f}%")
                            placed = True
                        else:
                            # 如果算法无法放置，强制创建容器但标记失败
                            instances.append({
                                'container': largest_container,
                                'groups': [copy.deepcopy(group)],
                                'result': {
                                    'placements': [],
                                    'utilization': 0,
                                    'weight': 0,
                                    'used_volume': 0,
                                    'packed_ids': [],
                                    'ok': False,
                                    'note': 'placement algorithm failed'
                                }
                            })
                            logging.info(f"警告: 算法无法放置组，强制创建容器")
                            placed = True

            # 如果成功放置（无论是现有容器还是新容器），移除该组
            if placed:
                remaining_groups.pop(i)
            else:
                i += 1  # 移动到下一个组

        # 最终统计
        logging.info(f"\n最终结果: 使用 {len(instances)} 个容器")
        for idx, inst in enumerate(instances):
            status = "成功" if inst['result']['ok'] else "失败"
            logging.info(f"  容器 {idx}: {inst['container'].name}, "
                         f"组数: {len(inst['groups'])}, "
                         f"利用率: {inst['result']['utilization'] * 100:.1f}%, "
                         f"状态: {status}")

        all_solutions.append({
            'bucket': bucket_key,
            'instances': instances,
            'remaining_groups': remaining_groups  # 真正无法放置的组
        })

    return all_solutions


# -----------------------
# 可视化 (Plotly) 与 UI
# -----------------------
def visualize_container_placements(res: Dict[str, Any], container: Container, groups: List[List[Cargo]] = None):
    """
    优化的可视化函数 - 深色背景下的清晰3D展示
    """
    placements = res.get('placements', [])

    # 数据验证（保持原有逻辑）
    if groups is not None:
        placed_ids = {p[4].uid for p in placements}
        expected_ids = set()
        for group in groups:
            for cargo in group:
                expected_ids.add(cargo.uid)

        if placed_ids != expected_ids:
            missing = expected_ids - placed_ids
            extra = placed_ids - expected_ids
            logging.info(f"可视化警告: 数据不一致! 期望{len(expected_ids)}个，实际{len(placed_ids)}个")

    fig = go.Figure()

    # 深色背景配色方案
    dark_bg_color = 'rgba(30, 30, 40, 1)'  # 深蓝黑色背景
    axis_grid_color = 'rgba(80, 100, 120, 0.3)'  # 坐标轴网格颜色
    grid_color = 'rgba(139, 134, 130, 0.6)'  # 明显的蓝色网格
    floor_color = 'rgba(60, 80, 100, 0.6)'  # 地板

    # 1. 设置整体背景
    fig.update_layout(
        paper_bgcolor=dark_bg_color,
        plot_bgcolor=dark_bg_color,
        scene=dict(
            xaxis=dict(backgroundcolor=dark_bg_color),
            yaxis=dict(backgroundcolor=dark_bg_color),
            zaxis=dict(backgroundcolor=dark_bg_color)
        )
    )

    # 2. 优化容器地板
    fig.add_trace(go.Mesh3d(
        x=[0, container.length, container.length, 0],
        y=[0, 0, container.width, container.width],
        z=[0, 0, 0, 0],
        color=floor_color,
        opacity=0.9,
        name='Container Floor',
        showlegend=False
    ))

    # 添加地板网格线 - 使用明显的颜色
    for i in range(0, int(container.length) + 1, 1):
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[0, container.width], z=[0, 0],
            mode='lines', line=dict(color=grid_color, width=1.5),
            showlegend=False
        ))
    for i in range(0, int(container.width) + 1, 1):
        fig.add_trace(go.Scatter3d(
            x=[0, container.length], y=[i, i], z=[0, 0],
            mode='lines', line=dict(color=grid_color, width=1.5),
            showlegend=False
        ))

    wall_opacity = 0.9  # 增加不透明度
    wall_color = 'rgba(100, 130, 160, 0.8)'  # 更亮的蓝色墙壁
    wall_border_color = 'rgba(180, 210, 240, 0.9)'  # 明亮的边框
    # 后墙 (Y=container.width)
    fig.add_trace(go.Mesh3d(
        x=[0, container.length, container.length, 0],
        y=[container.width, container.width, container.width, container.width],
        z=[0, 0, container.height, container.height],
        color=wall_color,
        opacity=wall_opacity,
        name='Back Wall',
        showlegend=False
    ))
    # 后墙边框 - 使用明亮的线条
    fig.add_trace(go.Scatter3d(
        x=[0, container.length, container.length, 0, 0],
        y=[container.width, container.width, container.width, container.width, container.width],
        z=[0, 0, container.height, container.height, 0],
        mode='lines',
        line=dict(color=wall_border_color, width=3),  # 加粗边框
        name='Back Wall Border',
        showlegend=False
    ))

    # 右侧墙 (X=container.length)
    fig.add_trace(go.Mesh3d(
        x=[container.length, container.length, container.length, container.length],
        y=[0, container.width, container.width, 0],
        z=[0, 0, container.height, container.height],
        color=wall_color,
        opacity=wall_opacity,
        name='Right Wall',
        showlegend=False
    ))
    # 右侧墙边框
    fig.add_trace(go.Scatter3d(
        x=[container.length, container.length, container.length, container.length, container.length],
        y=[0, container.width, container.width, 0, 0],
        z=[0, 0, container.height, container.height, 0],
        mode='lines',
        line=dict(color=wall_border_color, width=3),
        name='Right Wall Border',
        showlegend=False
    ))

    # 左侧墙 (X=0)
    fig.add_trace(go.Mesh3d(
        x=[0, 0, 0, 0],
        y=[0, container.width, container.width, 0],
        z=[0, 0, container.height, container.height],
        color=wall_color,
        opacity=wall_opacity,
        name='Left Wall',
        showlegend=False
    ))
    # 左侧墙边框
    fig.add_trace(go.Scatter3d(
        x=[0, 0, 0, 0, 0],
        y=[0, container.width, container.width, 0, 0],
        z=[0, 0, container.height, container.height, 0],
        mode='lines',
        line=dict(color=wall_border_color, width=3),
        name='Left Wall Border',
        showlegend=False
    ))
    # 添加顶部边框，让容器更完整
    fig.add_trace(go.Scatter3d(
        x=[0, container.length, container.length, 0, 0],
        y=[0, 0, container.width, container.width, 0],
        z=[container.height, container.height, container.height, container.height, container.height],
        mode='lines',
        line=dict(color=wall_border_color, width=2),
        name='Top Border',
        showlegend=False
    ))

    # 4. 优化门的位置标记 - 使用明亮的颜色
    door_width = min(container.width * 0.9, 2.4)  # 门的宽度（Y方向）
    door_height = min(container.height * 0.9, 2.8)  # 门的高度（Z方向）

    # 计算门的位置 - 在X=container.length的面，居中放置
    door_y_start = (container.width - door_width) / 2  # 门在Y方向的起始位置
    door_y_end = door_y_start + door_width  # 门在Y方向的结束位置
    door_z_end = door_height  # 门的高度

    # 门的面（X=container.length，宽高面）
    fig.add_trace(go.Mesh3d(
        x=[container.length, container.length, container.length, container.length],  # X=container.length
        y=[door_y_start, door_y_end, door_y_end, door_y_start],  # Y方向从door_y_start到door_y_end
        z=[0, 0, door_z_end, door_z_end],  # Z方向从地面到门高
        color='rgba(255, 220, 80, 0.9)',
        name='Door',
        showlegend=False
    ))

    # 门框 - 在X=container.length的面上的门框
    fig.add_trace(go.Scatter3d(
        x=[container.length, container.length, container.length, container.length, container.length],
        y=[door_y_start, door_y_end, door_y_end, door_y_start, door_y_start],
        z=[0, 0, door_z_end, door_z_end, 0],
        mode='lines',
        line=dict(color='rgba(255, 240, 150, 0.9)', width=3),
        name='Door Frame',
        showlegend=False
    ))

    # 添加顶部边框
    fig.add_trace(go.Scatter3d(
        x=[0, container.length, container.length, 0, 0],
        y=[0, 0, container.width, container.width, 0],
        z=[container.height, container.height, container.height, container.height, container.height],
        mode='lines',
        line=dict(color=wall_border_color, width=2),
        showlegend=False
    ))

    # 5. 优化货物颜色方案 - 使用鲜艳的颜色在深色背景下突出
    type_colors = {
        'Crate': (255, 140, 100),  # 温暖的橙色
        'Pallet': (100, 200, 140),  # 清新的绿色
        'Box': (100, 160, 255)  # 明亮的蓝色
    }

    # 预计算所有货物以优化性能
    cargo_traces = []
    arrow_traces = []
    cone_traces = []
    label_traces = []

    for idx, (x, y, z, (l, w, h), cargo, mode) in enumerate(placements):
        # 颜色处理 - 使用鲜艳的颜色
        base_rgb = type_colors.get(cargo.package_type, (200, 200, 200))
        supplier_hash = hash(cargo.supplier) % 20
        r = min(max(base_rgb[0] + supplier_hash - 10, 50), 255)
        g = min(max(base_rgb[1] + supplier_hash - 10, 50), 255)
        b = min(max(base_rgb[2] + supplier_hash - 10, 50), 255)
        color = f'rgb({r},{g},{b})'
        border_color = 'rgba(0, 0, 0, 1)'  # 黑色边框

        # 尺寸和方向处理
        if mode == 'L':
            actual_l, actual_w, actual_h = l, w, h
            fork_direction = 'W'
        elif mode == 'W':
            actual_l, actual_w, actual_h = w, l, h
            fork_direction = 'L'
        else:
            actual_l, actual_w, actual_h = l, w, h
            fork_direction = 'W'

        # 货物立方体 - 使用正确的Mesh3d配置
        # 实心立方体的8个顶点
        vertices_x = [x, x + actual_l, x + actual_l, x, x, x + actual_l, x + actual_l, x]
        vertices_y = [y, y, y + actual_w, y + actual_w, y, y, y + actual_w, y + actual_w]
        vertices_z = [z, z, z, z, z + actual_h, z + actual_h, z + actual_h, z + actual_h]

        # 修复：使用正确的面索引
        # 每个面由2个三角形组成，共12个三角形
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 1]
        k = [0, 7, 2, 3, 6, 7, 1, 5, 5, 5, 7, 6]

        # 创建实心货物 - 修复颜色配置
        cargo_trace = go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=i,
            j=j,
            k=k,
            facecolor=[color] * 12,  # 每个面使用相同的颜色
            opacity=1.0,  # 完全不透明
            flatshading=True,  # 平面着色
            name=f"ID:{cargo.uid}",
            showlegend=False,
            # 直接启用hover，不需要额外的透明层
            hovertemplate='<b>ID:%{customdata[0]}</b><br>'
                          '位置: (X:%{x:.1f}, Y:%{y:.1f}, Z:%{z:.1f})<br>'
                          '尺寸: %{customdata[1]} × %{customdata[2]} × %{customdata[3]}<br>'
                          '类型: %{customdata[4]}<br>'
                          '供应商: %极customdata[5]}<br>'
                          '<extra></extra>',
            customdata=[[
                cargo.uid,
                f"{actual_l:.1f}",
                f"{actual_w:.1f}",
                f"{actual_h:.1f}",
                cargo.package_type,
                cargo.supplier
            ]] * 8
        )
        cargo_traces.append(cargo_trace)
        edges = [
            # 底部边框
            ([x, x + actual_l], [y, y], [z, z]),
            ([x + actual_l, x + actual_l], [y, y + actual_w], [z, z]),
            ([x + actual_l, x], [y + actual_w, y + actual_w], [z, z]),
            ([x, x], [y + actual_w, y], [z, z]),
            # 顶部边框
            ([x, x + actual_l], [y, y], [z + actual_h, z + actual_h]),
            ([x + actual_l, x + actual_l], [y, y + actual_w], [z + actual_h, z + actual_h]),
            ([x + actual_l, x], [y + actual_w, y + actual_w], [z + actual_h, z + actual_h]),
            ([x, x], [y + actual_w, y], [z + actual_h, z + actual_h]),
            # 垂直边框
            ([x, x], [y, y], [z, z + actual_h]),
            ([x + actual_l, x + actual_l], [y, y], [z, z + actual_h]),
            ([x + actual_l, x + actual_l], [y + actual_w, y + actual_w], [z, z + actual_h]),
            ([x, x], [y + actual_w, y + actual_w], [z, z + actual_h])
        ]

        for edge_x, edge_y, edge_z in edges:
            border_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(color=border_color, width=3),  # 加粗边框
                showlegend=False,
                hoverinfo='skip'
            )
            cargo_traces.append(border_trace)

        # 进叉方向指示器
        center_x = x + actual_l / 2
        center_y = y + actual_w / 2

        if fork_direction == 'W':
            arrow_x = [center_x, center_x]
            arrow_y = [center_y, center_y + min(actual_w * 0.3, 0.4)]
        else:
            arrow_x = [center_x, center_x + min(actual_l * 0.3, 0.4)]
            arrow_y = [center_y, center_y]

        arrow_z = [z + 0.05, z + 0.05]

        arrow_traces.append(go.Scatter3d(
            x=arrow_x, y=arrow_y, z=arrow_z,
            mode='lines',
            line=dict(color='rgba(255, 255, 0, 0.9)', width=4),  # 黄色更醒目
            showlegend=False,
            hoverinfo='skip'
        ))

        # 箭头头部 - 加大尺寸
        cone_traces.append(go.Cone(
            x=[arrow_x[1]], y=[arrow_y[1]], z=[arrow_z[1]],
            u=[arrow_x[1] - arrow_x[0]], v=[arrow_y[1] - arrow_y[0]], w=[0],
            colorscale=[[0, 'rgba(255, 255, 0, 0.9)'], [1, 'rgba(255, 255, 0, 0.9)']],
            showscale=False,
            sizemode='absolute',
            sizeref=0.2,  # 加大箭头头部
            showlegend=False,
            hoverinfo='skip'
        ))

        # 货物标签 - 使用白色文字在深色背景下清晰
        label_traces.append(go.Scatter3d(
            x=[center_x], y=[center_y], z=[z + actual_h + 0.1],
            mode='text', text=[f"{cargo.uid}"],
            textfont=dict(size=11, color='rgba(255,255,255,0.95)'),
            showlegend=False, hoverinfo='skip'
        ))

    # 批量添加所有货物相关的trace
    for trace in cargo_traces + arrow_traces + cone_traces + label_traces:
        fig.add_trace(trace)

    # 6. 优化布局设置 - 深色主题
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Length',
                range=[-0, container.length],
                backgroundcolor=dark_bg_color,
                gridcolor=axis_grid_color,
                gridwidth=1,
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title='Width',
                range=[-0, container.width],
                backgroundcolor=dark_bg_color,
                gridcolor=axis_grid_color,
                gridwidth=1,
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                showgrid=True,
                zeroline=False
            ),
            zaxis=dict(
                title='Height',
                range=[-0, container.height],
                backgroundcolor=dark_bg_color,
                gridcolor=axis_grid_color,
                gridwidth=1,
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                showgrid=True,
                zeroline=False
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)

            )

        ),
        title=dict(
            text=f"📦 装载可视化 - {container.name}<br>"
                 f"货物数量: {len(placements)} | 空间利用率: {res.get('utilization', 0) * 100:.1f}%",
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='white')
        ),
        margin=dict(r=20, l=20, b=20, t=80),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(40,40,50,0.9)',
            bordercolor='rgba(100,100,120,0.5)',
            borderwidth=1,
            font=dict(color='white')
        )
    )

    # 添加交互提示
    fig.update_traces(
        selector=dict(type='mesh3d'),  # 只对mesh3d类型的trace应用
        hovertemplate='<b>%{customdata[0]}</b><br>'
                      '位置: (X:%{x:.1f}, Y:%{y:.1f}, Z:%{z:.1f})<br>'
                      '尺寸: %{customdata[1]} × %{customdata[2]} × %{customdata[3]}<br>'
                      '类型: %{customdata[4]}<br>'
                      '供应商: %{customdata[5]}<br>'
                      '<extra></extra>'
    )


    return fig


def create_cuboid_plot(x, y, z, length, width, height, name, color):
    """创建优化的立方体绘图"""
    # 立方体的8个顶点
    vertices = [
        [x, y, z],
        [x + length, y, z],
        [x + length, y + width, z],
        [x, y + width, z],
        [x, y, z + height],
        [x + length, y, z + height],
        [x + length, y + width, z + height],
        [x, y + width, z + height]
    ]

    # 立方体的6个面
    faces = [
        [0, 1, 2, 3],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [2, 3, 7, 6],  # 后面
        [0, 3, 7, 4],  # 左面
        [1, 2, 6, 5]  # 右面
    ]

    # 提取坐标
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]

    # 创建mesh trace
    return go.Mesh3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        i=[face[0] for face in faces],
        j=[face[1] for face in faces],
        k=[face[2] for face in faces],
        facecolor=[color] * 6,
        opacity=0.8,
        name=name,
        showlegend=True,
        hoverinfo='skip',
        customdata=[[name, length, width, height]]
    )


def default_demo_df():
    df = pd.read_excel(r'C:\Users\HMG-BA110\Desktop\forecastorderdetail_1757643601667.xlsx', dtype=str,
                       sheet_name='Sheet1')
    # df = pd.DataFrame()

    return df


def parse_df_to_cargos(df: pd.DataFrame) -> List[Cargo]:
    cargos = []
    uid = 0
    for idx, row in df.iterrows():
        logging.info(f"解析第 {idx} 行: {dict(row)}")
        try:
            attachment_id = None
            attachment_val = row.get('附件箱号/栈板号')
            if (pd.notna(attachment_val) and
                    str(attachment_val).strip() and
                    str(attachment_val).lower() != 'nan' and
                    str(attachment_val).lower() != 'null'):
                attachment_id = str(attachment_val).strip()

            ot = row.get('预计发货时间')
            # 时间格式：2025-09-12 00:00:00
            order_time = datetime.strptime(ot, '%Y-%m-%d %H:%M:%S')
            c = Cargo(
                transport_mode=str(row.get('transport_mode') or row.get('转运方式') or 'Ocean'),
                route=str(row.get('route') or row.get('路线') or 'MY'),
                supplier=str(row.get('supplier') or row.get('供应商名称') or ''),
                customer_order=str(row.get('customer_order') or row.get('客户单号') or ''),
                length=float(row.get('长')) / 100,
                width=float(row.get('宽')) / 100,
                height=float(row.get('高')) / 100,
                # 输入的是厘米 这里修改成米
                quantity=int(row.get('quantity') or row.get('件数') or 1),
                gross_weight=float(row.get('gross_weight') or row.get('毛重') or 10.0),
                order_time=order_time,
                package_type=str(row.get('package_type') or row.get('包装类型') or 'Box'),
                # 数据输入为‘是’ 或 ‘否’ 映射为 Y N,
                stackable='Y' if row.get('是否可堆叠') == '是' else 'N',
                load_dir=str(row.get('load_dir') or row.get('装载长度类型') or 'L&W'),
                # attachment_id=str(row.get('附件箱号/栈板号')),
                attachment_id=attachment_id,

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
    st.subheader('货物输入（编辑或上传 CSV/XLSX）')
    df = st.data_editor(default_demo_df(), use_container_width=True, num_rows='dynamic', hide_index=True)
    uploaded = st.file_uploader('上传 CSV/XLSX', type=['csv', 'xlsx'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded, dtype=str)
            df = df_up
        except Exception as e:
            st.error(f'读取文件出错: {e}')

    cargos = parse_df_to_cargos(df)
    st.write(f"输入行数: {len(df)}。")

    if st.button('开始计算'):
        logging.info('------------------------------------------')
        if not container_types:
            st.error('请至少选择一个容器类型')
            return
        t0 = time.time()

        try:
            sols = plan_multi_containers(cargos, container_types)
            t1 = time.time()

            if not sols:
                st.error('未生成任何方案')
                return

            total_containers = 0
            total_unplaced = 0

            for sol in sols:
                total_containers += len(sol['instances'])
                total_unplaced += len(sol.get('remaining_groups', []))

                st.header(f"Bucket: {sol['bucket']}")

                # 显示未放置的组
                if sol.get('remaining_groups') and len(sol['remaining_groups']) > 0:
                    st.warning(f"有 {len(sol['remaining_groups'])} 个组无法放置:")
                    for i, group in enumerate(sol['remaining_groups']):
                        st.write(f"未放置组 {i + 1}:")
                        for cargo in group:
                            st.write(
                                f"- ID:{cargo.uid}, 订单: {cargo.customer_order}, "
                                f"尺寸: {cargo.length}x{cargo.width}x{cargo.height}")

                for idx, inst in enumerate(sol['instances']):
                    cont = inst['container']
                    res = inst['result']
                    groups_in_container = inst['groups']

                    st.subheader(f"容器 {idx + 1}: {cont.name}")

                    # 验证数据一致性
                    expected_cargos = sum(len(group) for group in groups_in_container)
                    actual_cargos = len(res.get('placements', []))

                    if res.get('ok', False):
                        if expected_cargos == actual_cargos:
                            st.success("✓ 数据一致性验证通过")
                        else:
                            st.error(f"✗ 数据不一致! 期望{expected_cargos}个货物，实际{actual_cargos}个")

                        st.write(f"利用率: {res.get('utilization', 0.0) * 100:.2f}%")
                        st.write(f"载重: {res.get('weight', 0.0):.2f}/{cont.max_weight}kg")
                        st.write(f"装载件数: {actual_cargos}")
                    else:
                        st.warning("⚠️ 算法无法放置货物")
                        st.write(f"原因: {res.get('note', '未知错误')}")
                        st.write(f"应装载件数: {expected_cargos}")

                    st.write(f"装载组数: {len(groups_in_container)}")

                    # 显示容器中的组信息
                    with st.expander("查看容器内组详情及约束验证"):
                        # 时间规则验证
                        all_cargos = []
                        for group in groups_in_container:
                            for cargo in group:
                                all_cargos.append(cargo)

                        time_valid = validate_time_rules_for_container(all_cargos, cont.transport_mode)
                        st.write(f"时间规则验证: {'✓ 通过' if time_valid else '✗ 失败'}")

                        for group_idx, group in enumerate(groups_in_container):
                            st.write(f"组 {group_idx + 1}: {len(group)} 个货物")
                            for cargo in group:
                                st.write(
                                    f"订单: {cargo.customer_order},供应商{cargo.supplier},预计发货时间{cargo.order_time}")

                    if res.get('placements'):
                        fig = visualize_container_placements(res, cont)
                        st.plotly_chart(fig, use_container_width=True)

            st.success(f'完成！共使用 {total_containers} 个容器，耗时 {t1 - t0:.1f} 秒')
            if total_unplaced > 0:
                st.warning(f'有 {total_unplaced} 个组无法放置，请检查货物尺寸或约束条件')

        except Exception as e:
            st.error(f"计算过程中出现错误: {e}")
            import traceback
            st.text(traceback.format_exc())


if __name__ == '__main__':
    app()
