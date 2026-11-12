import csv
import os
import random
from typing import List
try:
    from .n_queens_solver import NQueenSolver
except ImportError:
    from n_queens_solver import NQueenSolver


def run_pysat_benchmarks(
    n_values: List[int], output_csv: str = "task1_pysat_benchmark.csv"
) -> None:
    """
    Runs PySAT Glucose3 experiments across increasing values of N (N >= 4)
    and exports N, num_variables, num_clauses, runtime_ms to CSV.
    """
    rows = []
    print("=== Running PySAT (Glucose3) Benchmarks ===")
    print(f"{'N':>4} | {'Variables':>10} | {'Clauses':>10} | {'Runtime (ms)':>14} | {'Solved':>6}")
    print("-" * 56)

    for n in n_values:
        solver = NQueenSolver(n)
        res = solver.solve_pysat()
        assert solver.board_helper.is_valid_solution(res["board"]), f"Invalid PySAT board for N={n}"

        rows.append(
            {
                "N": res["n"],
                "num_variables": res["num_variables"],
                "num_clauses": res["num_clauses"],
                "runtime_ms": round(res["runtime_ms"], 4),
            }
        )
        print(
            f"{res['n']:>4} | {res['num_variables']:>10} | {res['num_clauses']:>10} | "
            f"{res['runtime_ms']:>14.4f} | {str(res['solved']):>6}"
        )

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["N", "num_variables", "num_clauses", "runtime_ms"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved PySAT benchmark to {output_csv}\n")


def run_min_conflicts_benchmarks(
    n_values: List[int],
    trials: int = 20,
    max_steps: int = 10000,
    output_csv: str = "task1_min_conflicts_benchmark.csv",
) -> None:
    """
    Evaluates Standard Min-Conflicts vs. Custom Heuristic Min-Conflicts across N values.
    Exports N, variant, avg_steps, avg_runtime_ms, success_rate to CSV.
    """
    rows = []
    print("=== Running Min-Conflicts Benchmarks (Standard vs. Custom Heuristic) ===")
    print(
        f"{'N':>4} | {'Variant':<22} | {'Avg Steps':>10} | {'Avg Time (ms)':>14} | {'Success Rate':>12}"
    )
    print("-" * 72)

    for n in n_values:
        for use_custom in (False, True):
            variant_name = (
                "MinConflicts_Custom" if use_custom else "MinConflicts_Standard"
            )
            total_steps = 0
            total_runtime = 0.0
            success_count = 0

            for trial in range(trials):
                # Seed deterministically per (N, trial) so both variants face reproducible runs
                random.seed(1000 * n + trial)
                solver = NQueenSolver(n)
                res = solver.solve_min_conflicts(
                    use_custom_heuristic=use_custom, max_steps=max_steps
                )

                if res["solved"]:
                    assert solver.board_helper.is_valid_solution(
                        res["board"]
                    ), f"Invalid Min-Conflicts solution for N={n}"
                    success_count += 1

                total_steps += res["steps"]
                total_runtime += res["runtime_ms"]

            avg_steps = round(total_steps / trials, 2)
            avg_runtime_ms = round(total_runtime / trials, 4)
            success_rate = round(success_count / trials, 2)

            rows.append(
                {
                    "N": n,
                    "variant": variant_name,
                    "avg_steps": avg_steps,
                    "avg_runtime_ms": avg_runtime_ms,
                    "success_rate": success_rate,
                }
            )
            print(
                f"{n:>4} | {variant_name:<22} | {avg_steps:>10.2f} | "
                f"{avg_runtime_ms:>14.4f} | {success_rate * 100:>11.1f}%"
            )

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "variant",
                "avg_steps",
                "avg_runtime_ms",
                "success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Min-Conflicts benchmark to {output_csv}\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pysat_csv_path = os.path.join(base_dir, "task1_pysat_benchmark.csv")
    mc_csv_path = os.path.join(base_dir, "task1_min_conflicts_benchmark.csv")

    # PySAT: N = 4..30
    pysat_n_range = list(range(4, 31))
    run_pysat_benchmarks(pysat_n_range, output_csv=pysat_csv_path)

    # Min-Conflicts: N = [4, 8, 16, 32, 50, 100, 200] with 20 trials per N
    mc_n_list = [4, 8, 16, 32, 50, 100, 200]
    run_min_conflicts_benchmarks(
        mc_n_list, trials=20, max_steps=10000, output_csv=mc_csv_path
    )
