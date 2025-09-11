import itertools
import random
import plotly.graph_objects as go
import streamlit as st

# 第一代演示版本
# ---------------- 核心类和方法 ----------------

# ---------------- 数据结构 ----------------
class Cargo:
    def __init__(self, length, width, height, weight, quantity=1,
                 stackable=True, vertical_only=False,
                 allow_stack_on_top=True, allow_stack_below=True):
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.quantity = quantity
        self.stackable = stackable
        self.vertical_only = vertical_only
        self.allow_stack_on_top = allow_stack_on_top
        self.allow_stack_below = allow_stack_below

    def orientations(self):
        """返回6个朝向 (长宽高互换)"""
        dims = (self.length, self.width, self.height)
        if self.vertical_only:  # 只能直立
            return [dims]
        return set(itertools.permutations(dims, 3))


class Vehicle:
    def __init__(self, length, width, height, cost, max_weight, name):
        self.length = length
        self.width = width
        self.height = height
        self.cost = cost
        self.max_weight = max_weight
        self.name = name


# ---------------- 极点装载器 ----------------
def pack_cargos(cargos, vehicle, order=None):
    if order:
        cargos_ordered = [cargos[i] for i in order]
    else:
        cargos_ordered = sorted(cargos, key=lambda c: c.length * c.width * c.height, reverse=True)

    placements = []
    extreme_points = [(0, 0, 0)]
    used_volume = 0
    total_weight = 0

    for cargo in cargos_ordered:
        for _ in range(cargo.quantity):
            placed = False
            extreme_points.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
            for ep in extreme_points:
                # 尝试所有旋转方向
                for dims in cargo.orientations():
                    if can_place(dims, cargo, ep, placements, vehicle):
                        placements.append((ep[0], ep[1], ep[2], dims, cargo))
                        used_volume += dims[0] * dims[1] * dims[2]
                        total_weight += cargo.weight
                        extreme_points = update_extreme_points(extreme_points, ep, dims, placements, vehicle)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                return None
    if total_weight > vehicle.max_weight:
        return None
    utilization = used_volume / (vehicle.length * vehicle.width * vehicle.height)
    return {"placements": placements, "utilization": utilization, "weight": total_weight}


def can_place(dims, cargo, position, placed, vehicle):
    l, w, h = dims
    x, y, z = position

    # 边界检查
    if x + l > vehicle.length or y + w > vehicle.width or z + h > vehicle.height:
        return False

    # 碰撞检测
    for px, py, pz, (pl, pw, ph), pcargo in placed:
        if not (x + l <= px or x >= px + pl or
                y + w <= py or y >= py + pw or
                z + h <= pz or z >= pz + ph):
            return False
        if pz + ph == z and not pcargo.allow_stack_on_top:
            return False
        if z > 0 and not cargo.allow_stack_below:
            return False

    # 悬空检测：如果不是放在地板上(z>0)，必须有支撑
    if z > 0:
        supported_area = 0
        cargo_area = l * w
        for px, py, pz, (pl, pw, ph), pcargo in placed:
            if abs(pz + ph - z) < 1e-6:  # 在同一高度
                # 计算重叠面积
                overlap_x = max(0, min(x + l, px + pl) - max(x, px))
                overlap_y = max(0, min(y + w, py + pw) - max(y, py))
                overlap = overlap_x * overlap_y
                supported_area += overlap
        # 至少要覆盖 90% 底面积（阈值可调）
        if supported_area < 0.95 * cargo_area:
            return False

    return True


def update_extreme_points(extreme_points, used_point, dims, placed, vehicle):
    extreme_points = [p for p in extreme_points if p != used_point]
    new_points = generate_new_extreme_points(used_point, dims)
    valid_points = []
    for p in new_points:
        if p in extreme_points:
            continue
        if p[0] >= vehicle.length or p[1] >= vehicle.width or p[2] >= vehicle.height:
            continue
        # 避免插入已有货物中
        if any(not (p[0] >= x + l or p[0] <= x - l or
                    p[1] >= y + w or p[1] <= y - w or
                    p[2] >= z + h or p[2] <= z - h)
               for x, y, z, (l, w, h), c in placed):
            continue
        valid_points.append(p)
    extreme_points.extend(valid_points)
    return extreme_points


def generate_new_extreme_points(position, dims):
    x, y, z = position
    l, w, h = dims
    return [
        (x + l, y, z),
        (x, y + w, z),
        (x, y, z + h)
    ]


# ---------------- 遗传算法 ----------------
def genetic_algorithm(cargos, vehicle, pop_size=50, generations=100, mutation_rate=0.1):
    genome_length = len(cargos)
    # 初始化：一部分随机 + 一部分贪心
    population = [random.sample(range(genome_length), genome_length) for _ in range(pop_size - 1)]
    population.append(
        sorted(range(genome_length), key=lambda i: cargos[i].length * cargos[i].width * cargos[i].height, reverse=True))

    def fitness(order):
        res = pack_cargos(cargos, vehicle, order)
        if not res:
            return 0
        return res["utilization"]

    for _ in range(generations):
        scored = [(chrom, fitness(chrom)) for chrom in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        survivors = [chrom for chrom, _ in scored[:pop_size // 2]]
        offspring = []
        while len(offspring) < pop_size // 2:
            p1, p2 = random.sample(survivors, 2)
            if genome_length > 2:
                cut = random.randint(1, genome_length - 2)
                child = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
            else:
                child = p1[:]
            if random.random() < mutation_rate and genome_length > 1:
                i, j = random.sample(range(genome_length), 2)
                child[i], child[j] = child[j], child[i]
            offspring.append(child)
        population = survivors + offspring

    best_order = max(population, key=fitness)
    return pack_cargos(cargos, vehicle, best_order)


# ---------------- 多车型推荐 ----------------
def recommend_vehicle(cargos, vehicles):
    vehicles_sorted = sorted(vehicles, key=lambda v: (v.length * v.width * v.height, v.max_weight))
    for v in vehicles_sorted:
        res = genetic_algorithm(cargos, v)
        if res:  # 找到能装下的第一个车型直接返回
            score = res["utilization"] - v.cost * 0.01
            return {
                "vehicle": v,
                "utilization": res["utilization"],
                "placements": res["placements"],
                "cost": v.cost,
                "weight": res["weight"],
                "score": score
            }
    return None


# ---------------- 可视化 ----------------
def create_cuboid(x, y, z, dx, dy, dz, color, name):
    X = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    Y = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    Z = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    I = [0, 0, 0, 1, 1, 2, 4, 5, 6, 4, 6, 7]
    J = [1, 2, 4, 2, 5, 3, 5, 6, 7, 0, 7, 3]
    K = [2, 4, 5, 3, 6, 0, 6, 7, 4, 7, 3, 0]
    return go.Mesh3d(
        x=X, y=Y, z=Z,
        i=I, j=J, k=K,
        opacity=0.9,
        color=color,
        name=name,
        hovertext=name,
        hoverinfo="text"
    )


def visualize_packing_plotly(placements, vehicle):
    fig = go.Figure()
    fig.add_trace(
        create_cuboid(0, 0, 0, vehicle.length, vehicle.width, vehicle.height, "rgba(0,0,0,0)", "Vehicle")
    )
    colors = {}
    for idx, (x, y, z, dims, cargo) in enumerate(placements):
        if cargo not in colors:
            colors[cargo] = f"rgb({random.randint(50, 200)}, {random.randint(50, 200)}, {random.randint(50, 200)})"
        name = f"货物{idx + 1}<br>尺寸:{dims[0]: .2f}x{dims[1]: .2f}x{dims[2]: .2f}m<br>重量:{cargo.weight}kg"
        fig.add_trace(
            create_cuboid(x, y, z, dims[0], dims[1], dims[2], colors[cargo], name)
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Length (m)'),
            yaxis=dict(title='Width (m)'),
            zaxis=dict(title='Height (m)'),
            aspectmode='data'
        ),
        title="3D 货物装载可视化",
        margin=dict(r=0, l=0, b=0, t=40)
    )
    fig.show()

# ---------------- 前端部分 ----------------

def main():
    st.set_page_config(page_title="智能装载推荐系统", layout="wide")
    st.title("🚚 智能货物装载与车型推荐")

    # ========== 全局参数（侧边栏） ==========
    st.sidebar.header("全局参数")

    # 车型
    vehicles = [
        Vehicle(4.2, 1.9, 1.9, cost=200, max_weight=2000, name="厢式货车(载重2T)"),
        Vehicle(4.2, 2.1, 2.1, cost=300, max_weight=3500, name="厢式货车(载重3T)"),
        Vehicle(5.2, 2.1, 2.1, cost=500, max_weight=5000, name="厢式货车(载重5T)"),
        Vehicle(6.8, 2.4, 2.7, cost=1000, max_weight=10000, name="栏板货车(载重10T)"),
        Vehicle(13.55, 2.34,2.69, cost=1200, max_weight=25000, name="45HQ")
    ]
    vehicle_names = [v.name for v in vehicles]
    selected_vehicles = st.sidebar.multiselect("选择可用车", vehicle_names, default=vehicle_names)
    chosen_vehicles = [v for v in vehicles if v.name in selected_vehicles]

    # 算法参数
    st.sidebar.subheader("算法参数")
    pop_size = st.sidebar.slider("种群大小", 10, 100, 50)
    generations = st.sidebar.slider("迭代次数", 10, 200, 100)
    mutation_rate = st.sidebar.slider("变异概率", 0.01, 0.5, 0.1)

    # ========== 货物清单（表格输入） ==========
    st.subheader("📦 货物清单（支持 Excel 粘贴）")
    default_data = [
        {"编号": 1, "长(m)": 1.3, "宽(m)": 1.1, "高(m)": 0.75, "重量(KG)": 194, "数量": 1, "可堆叠": True, "仅竖放": True,
         "可在上放": True, "可在下放": False},
        {"编号": 2, "长(m)": 0.76, "宽(m)": 0.62, "高(m)": 0.46, "重量(KG)": 52, "数量": 1, "可堆叠": True,
         "仅竖放": True,
         "可在上放": True, "可在下放": False},
        {"编号": 3, "长(m)": 1.27, "宽(m)": 1.1, "高(m)": 0.8, "重量(KG)": 174, "数量": 1, "可堆叠": True,
         "仅竖放": True,
         "可在上放": True, "可在下放": False},
        {"编号": 4, "长(m)": 1.5, "宽(m)": 1, "高(m)": 0.7, "重量(KG)": 266, "数量": 1, "可堆叠": True,
         "仅竖放": True,
         "可在上放": True, "可在下放": False},
        {"编号": 5, "长(m)": 1.2, "宽(m)": 0.64, "高(m)": 0.62, "重量(KG)": 94, "数量": 1, "可堆叠": True,
         "仅竖放": True,
         "可在上放": True, "可在下放": False},
        {"编号": 6, "长(m)": 1.23, "宽(m)": 1.03, "高(m)": 0.93, "重量(KG)": 44, "数量": 1, "可堆叠": True,
         "仅竖放": True,
         "可在上放": True, "可在下放": False}
    ]

    cargo_records = st.data_editor(default_data, use_container_width=False, num_rows="dynamic", hide_index=True)

    # 构造 Cargo 对象
    cargos = []
    for c in cargo_records:
        cargos.append(Cargo(
            float(c["长(m)"]), float(c["宽(m)"]), float(c["高(m)"]),
            weight=float(c["重量(KG)"]),
            quantity=int(c["数量"]),  # 🔑 强制转 int
            stackable=bool(c["可堆叠"]),
            vertical_only=bool(c["仅竖放"]),
            allow_stack_on_top=bool(c["可在上放"]),
            allow_stack_below=bool(c["可在下放"])
        ))

    # ========== 开始计算 ==========
    if st.button("开始推荐"):
        with st.spinner("正在计算最优装载方案..."):
            best = recommend_vehicle(cargos, chosen_vehicles)

        if best:
            st.success(f"推荐车型: {best['vehicle'].name}")
            st.write(f"利用率: {best['utilization'] * 100: .2f}%")
            st.write(f"装载重量: {best['weight']} / {best['vehicle'].max_weight} kg")
            st.write(f"费用: {best['cost']}")

            # 可视化
            fig = go.Figure()
            fig.add_trace(create_cuboid(0, 0, 0, best['vehicle'].length, best['vehicle'].width, best['vehicle'].height,
                                        "rgba(0,0,0,0)", "车辆"))
            colors = {}
            for idx, (x, y, z, dims, cargo) in enumerate(best['placements']):
                if cargo not in colors:
                    colors[
                        cargo] = f"rgb({random.randint(50, 200)}, {random.randint(50, 200)}, {random.randint(50, 200)})"
                name = f"货物{idx + 1}<br>尺寸:{dims[0]}x{dims[1]}x{dims[2]} m<br>重量:{cargo.weight}kg"
                fig.add_trace(create_cuboid(x, y, z, dims[0], dims[1], dims[2], colors[cargo], name))

            fig.update_layout(
                scene=dict(xaxis=dict(title='长 (m)'), yaxis=dict(title='宽 (m)'), zaxis=dict(title='高 (m)'),
                           aspectmode='data'),
                title="3D 装载方案",
                margin=dict(l=0, r=0, b=0, t=30)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("所有车型都装不下或超重！")


if __name__ == "__main__":
    main()

