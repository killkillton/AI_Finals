"""Compares four Kakuro solver configurations on the sample boards."""

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Task3.kakuro_csp import KakuroCSP

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BOARDS = ["board_5x5", "board_10x10"]
REPEATS = 5

CONFIGS = [
    (
        "Backtracking + Forward Checking (Baseline - Static Order)",
        dict(use_mrv=False, use_degree=False, use_lcv=False, use_ac3=False, use_fc=True),
    ),
    (
        "Backtracking + Forward Checking + MRV + Degree + LCV",
        dict(use_mrv=True, use_degree=True, use_lcv=True, use_ac3=False, use_fc=True),
    ),
    (
        "Backtracking + FC + AC-3 (No Ordering Heuristics)",
        dict(use_mrv=False, use_degree=False, use_lcv=False, use_ac3=True, use_fc=True),
    ),
    (
        "Backtracking + FC + AC-3 + MRV + Degree + LCV (Full Solver)",
        dict(use_mrv=True, use_degree=True, use_lcv=True, use_ac3=True, use_fc=True),
    ),
]

FIELDS = [
    "board_id", "configuration", "solved", "valid", "steps",
    "assignments_tried", "backtracks", "prunings", "elapsed_ms",
]


def run_all():
    rows = []
    for board_id in BOARDS:
        for name, cfg in CONFIGS:
            times = []
            result = None
            for _ in range(REPEATS):
                csp = KakuroCSP(board_id)
                result = csp.solve(**cfg)
                times.append(result["elapsed_ms"])
            rows.append({
                "board_id": board_id,
                "configuration": name,
                "solved": result["solved"],
                "valid": csp.is_valid_solution(result["assignments"]),
                "steps": result["steps"],
                "assignments_tried": result["assignments_tried"],
                "backtracks": result["backtracks"],
                "prunings": result["prunings"],
                "elapsed_ms": round(statistics.median(times), 3),
            })
    return rows


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = ["FC (static)", "FC + heuristics", "FC + AC-3", "Full solver"]
    names = [name for name, _ in CONFIGS]
    metrics = [("backtracks", "Backtracks"), ("assignments_tried", "Assignments tried"),
               ("elapsed_ms", "Median time (ms)")]
    fig, axes = plt.subplots(len(metrics), len(BOARDS), figsize=(11, 10))
    colors = ["#c0504d", "#f79646", "#4f81bd", "#9bbb59"]
    for j, board_id in enumerate(BOARDS):
        for i, (key, label) in enumerate(metrics):
            ax = axes[i][j]
            values = [next(r[key] for r in rows if r["board_id"] == board_id and r["configuration"] == n)
                      for n in names]
            bars = ax.bar(short, values, color=colors)
            ax.set_title(f"{board_id}: {label}")
            ax.tick_params(axis="x", labelrotation=15, labelsize=8)
            for bar, v in zip(bars, values):
                ax.annotate(f"{v:g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    rows = run_all()
    save_csv(rows, RESULTS_DIR / "kakuro_solver_comparison.csv")
    save_chart(rows, RESULTS_DIR / "kakuro_performance_comparison.png")
    for r in rows:
        print(f"{r['board_id']:<12} bt={r['backtracks']:<5} tried={r['assignments_tried']:<5} "
              f"ms={r['elapsed_ms']:<9} valid={r['valid']}  {r['configuration']}")
    print("Saved results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
