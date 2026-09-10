from ortools.sat.python import cp_model
import math
from collections import defaultdict

# ============================================================
# 参数
# ============================================================

EDOS = [72, 108]
TOLERANCES = [4.0, 5.5, 7.0]   # cents
NOTE_COUNTS = range(5, 13)

MIN_SEPARATION_CENTS = 50.0

# 如果想尽可能证明全局最优，可以调大
MAX_TIME_SECONDS = 300.0

NUM_WORKERS = 16


# ============================================================
# 9 类和谐音程
#
# 名称保持你定义的 harmonic ratio，而实际会自动
# octave-reduce 到 [1,2)，并同时加入八度反转。
# ============================================================

HARMONIC_INTERVALS = [
    ("3/1", 3.0 / 1.0),
    ("5/1", 5.0 / 1.0),
    ("7/1", 7.0 / 1.0),

    ("5/3", 5.0 / 3.0),
    ("7/3", 7.0 / 3.0),
    ("7/5", 7.0 / 5.0),

    ("9/1", 9.0 / 1.0),
    ("9/5", 9.0 / 5.0),
    ("9/7", 9.0 / 7.0),
]


# ============================================================
# 工具函数
# ============================================================

def octave_reduce(r):
    """把比例折叠到 [1,2)."""
    while r >= 2.0:
        r /= 2.0
    while r < 1.0:
        r *= 2.0
    return r


def ratio_to_cents(r):
    return 1200.0 * math.log2(r)


def circular_cents_distance(a, b):
    """两个 octave-class cents 的最短圆周距离."""
    d = abs(a - b) % 1200.0
    return min(d, 1200.0 - d)


# ============================================================
# 构造 9 类 target
#
# 对每一类保存：
#   direct center
#   inversion center
#
# 比如 3/1:
#   3/2 = 701.955c
#   4/3 = 498.045c
# ============================================================

def build_interval_targets():
    targets = []

    for name, ratio in HARMONIC_INTERVALS:
        r = octave_reduce(ratio)
        c = ratio_to_cents(r)

        inverse_c = 1200.0 - c

        targets.append({
            "name": name,
            "ratio": r,
            "center": c,
            "inverse": inverse_c,
        })

    return targets


TARGETS = build_interval_targets()


def print_targets():
    print("============================================================")
    print("Harmonic interval targets")
    print("============================================================")

    for t in TARGETS:
        print(
            f"{t['name']:>4} : "
            f"{t['center']:10.6f} c"
            f"   inverse={t['inverse']:10.6f} c"
        )

    print()


# ============================================================
# 判断一个 EDO interval 是否合格
#
# 返回：
#   合格与否
#   命中的类型
#   对应误差
#
# 正常情况下各 target 不重叠。
# 如果未来容差放很大导致重叠，也只把一个音对计一次。
# ============================================================

def classify_interval(steps, edo, tolerance):
    cents = 1200.0 * steps / edo

    matches = []

    for t in TARGETS:
        e1 = abs(cents - t["center"])
        e2 = abs(cents - t["inverse"])

        err = min(e1, e2)

        if err <= tolerance + 1e-9:
            # 保留带符号误差，方便输出
            if e1 <= e2:
                signed_error = cents - t["center"]
                orientation = "direct"
            else:
                signed_error = cents - t["inverse"]
                orientation = "inverse"

            matches.append(
                (
                    err,
                    t["name"],
                    signed_error,
                    orientation
                )
            )

    if not matches:
        return False, None, None, None

    matches.sort()

    _, name, signed_error, orientation = matches[0]

    return True, name, signed_error, orientation


# ============================================================
# 预计算 EDO 中哪些步数属于和谐音程
# ============================================================

def build_harmonic_step_table(edo, tolerance):
    table = {}

    for d in range(1, edo):
        ok, name, error, orientation = classify_interval(
            d, edo, tolerance
        )

        if ok:
            table[d] = {
                "type": name,
                "error": error,
                "orientation": orientation,
            }

    return table


# ============================================================
# 最小音距硬约束
#
# pitch-class 是圆周结构：
#
# min(d, edo-d) * 1200/edo >= 50c
# ============================================================

def too_close(i, j, edo):
    d = abs(j - i)
    d = min(d, edo - d)

    cents = d * 1200.0 / edo

    return cents < MIN_SEPARATION_CENTS - 1e-9


# ============================================================
# 精确求解一个：
#
#   edo
#   tolerance
#   note_count
#
# 目标：
#   最大化和谐无向音对数量
#
# 因为 EDO 具有旋转对称性，
# 可以 WLOG 固定 pitch class 0 被选中。
# ============================================================

def solve_scale(edo, tolerance, note_count):

    harmonic_steps = build_harmonic_step_table(
        edo, tolerance
    )

    model = cp_model.CpModel()

    # x[i] = 是否选择第 i 个 EDO pitch class
    x = [
        model.NewBoolVar(f"x_{i}")
        for i in range(edo)
    ]

    # --------------------------------------------------------
    # 固定音数
    # --------------------------------------------------------

    model.Add(sum(x) == note_count)

    # --------------------------------------------------------
    # 消除旋转对称性
    #
    # 任意解都可以整体平移，让其中一个音位于 0。
    # --------------------------------------------------------

    model.Add(x[0] == 1)

    # --------------------------------------------------------
    # 50 cents 最小圆周音距
    # --------------------------------------------------------

    for i in range(edo):
        for j in range(i + 1, edo):

            if too_close(i, j, edo):
                model.Add(x[i] + x[j] <= 1)

    # --------------------------------------------------------
    # harmonic edge variables
    #
    # 对每一个本身属于和谐音程的 EDO 点对建立 y_ij。
    #
    # y_ij = x_i AND x_j
    # --------------------------------------------------------

    y_vars = []
    edge_info = []

    for i in range(edo):
        for j in range(i + 1, edo):

            d = j - i

            # 无向音对，因此 d 与 edo-d 等价；
            # classify_interval 已经显式考虑 inverse，
            # 直接使用 d 即可。
            if d not in harmonic_steps:
                continue

            y = model.NewBoolVar(f"y_{i}_{j}")

            model.Add(y <= x[i])
            model.Add(y <= x[j])
            model.Add(y >= x[i] + x[j] - 1)

            y_vars.append(y)

            edge_info.append(
                (
                    y,
                    i,
                    j,
                    d,
                    harmonic_steps[d]
                )
            )

    # --------------------------------------------------------
    # 最大化和谐音对数
    # --------------------------------------------------------

    model.Maximize(sum(y_vars))

    # ========================================================
    # 求解
    # ========================================================

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = MAX_TIME_SECONDS
    solver.parameters.num_search_workers = NUM_WORKERS

    # 尽可能增强证明能力
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 2

    status = solver.Solve(model)

    # ========================================================
    # 输出
    # ========================================================

    status_name = solver.StatusName(status)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        return {
            "status": status_name,
            "edo": edo,
            "tolerance": tolerance,
            "note_count": note_count,
        }

    selected = [
        i
        for i in range(edo)
        if solver.Value(x[i])
    ]

    harmonic_edges = []

    type_counts = defaultdict(int)

    for y, i, j, d, info in edge_info:

        if solver.Value(y):

            harmonic_edges.append(
                (i, j, d, info)
            )

            type_counts[info["type"]] += 1

    score = len(harmonic_edges)

    total_pairs = (
        note_count * (note_count - 1) // 2
    )

    # 循环步长
    gaps = []

    for k in range(note_count):
        a = selected[k]
        b = selected[(k + 1) % note_count]

        if k == note_count - 1:
            gap = edo - a + selected[0]
        else:
            gap = b - a

        gaps.append(gap)

    # pitch cents
    cents = [
        1200.0 * s / edo
        for s in selected
    ]

    return {
        "status": status_name,
        "optimal": status == cp_model.OPTIMAL,

        "edo": edo,
        "tolerance": tolerance,
        "note_count": note_count,

        "score": score,
        "total_pairs": total_pairs,
        "ratio": score / total_pairs,

        "steps": selected,
        "gaps": gaps,
        "cents": cents,

        "type_counts": dict(type_counts),
        "edges": harmonic_edges,

        # CP-SAT 的上界
        "best_bound": solver.BestObjectiveBound(),
    }


# ============================================================
# 打印单个结果
# ============================================================

def print_result(r, print_edges=False):

    print(
        f"{r['edo']:3d}EDO   "
        f"tol={r['tolerance']:4.1f}c   "
        f"n={r['note_count']:2d}"
    )

    if "score" not in r:
        print("  STATUS:", r["status"])
        return

    proof = (
        "OPTIMAL"
        if r["optimal"]
        else f"FEASIBLE, bound={r['best_bound']:.0f}"
    )

    print(
        f"  harmony = "
        f"{r['score']:2d}/{r['total_pairs']:2d}"
        f" = {100*r['ratio']:6.2f}%"
        f"   [{proof}]"
    )

    print(
        "  steps =",
        " ".join(map(str, r["steps"]))
    )

    print(
        "  gaps  =",
        " ".join(map(str, r["gaps"]))
    )

    print(
        "  cents =",
        " ".join(
            f"{x:.3f}"
            for x in r["cents"]
        )
    )

    # 按固定顺序输出 9 类边数
    print("  edge types:")

    for name, _ in HARMONIC_INTERVALS:
        count = r["type_counts"].get(name, 0)

        print(
            f"    {name:>4}: {count:2d}"
        )

    if print_edges:

        print("  edges:")

        for i, j, d, info in r["edges"]:

            cents = 1200.0 * d / r["edo"]

            print(
                f"    "
                f"{i:3d}-{j:3d}"
                f"  d={d:3d}"
                f"  {cents:9.4f} c"
                f"  {info['type']:>4}"
                f"  error={info['error']:+7.3f} c"
            )

    print()


# ============================================================
# 主搜索
# ============================================================

def main():

    print_targets()

    all_results = []

    for edo in EDOS:

        print()
        print("#" * 72)
        print(f"# {edo} EDO")
        print("#" * 72)
        print()

        for tolerance in TOLERANCES:

            print("=" * 72)
            print(
                f"{edo} EDO, tolerance = ±{tolerance} cents"
            )
            print("=" * 72)
            print()

            for n in NOTE_COUNTS:

                r = solve_scale(
                    edo,
                    tolerance,
                    n
                )

                all_results.append(r)

                print_result(r)

    # ========================================================
    # 最后给一个紧凑 summary
    # ========================================================

    print()
    print("#" * 72)
    print("# SUMMARY")
    print("#" * 72)

    for edo in EDOS:

        print()
        print(f"{edo} EDO")

        for tol in TOLERANCES:

            row = []

            for n in NOTE_COUNTS:

                r = next(
                    x for x in all_results
                    if x["edo"] == edo
                    and x["tolerance"] == tol
                    and x["note_count"] == n
                )

                if "score" in r:
                    marker = (
                        ""
                        if r["optimal"]
                        else "?"
                    )

                    row.append(
                        f"{n}:{r['score']}{marker}"
                    )
                else:
                    row.append(
                        f"{n}:X"
                    )

            print(
                f"  ±{tol:3.1f}c : "
                + "  ".join(row)
            )


if __name__ == "__main__":
    main()