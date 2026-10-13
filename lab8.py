# AI Lab 8: Uninformed Search Algorithms
# Algorithms: BFS, DFS, DLS, Iterative Deepening

import csv
import random
import time
import multiprocessing as mp
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt

TIME_LIMIT = 60
DLS_LIMIT = 20
TESTS = 10
BFS_STATE_LIMIT = 200000
ALGORITHMS = ["BFS", "DFS", "DLS", "ID"]


class StateLimitReached(Exception):
    pass


# Create goal state
def goal_state(n):
    return tuple(list(range(1, n * n)) + [0])


# Generate valid neighboring states
def neighbors(state, n):
    zero = state.index(0)
    row, col = divmod(zero, n)
    positions = []

    if row > 0:
        positions.append(zero - n)
    if row < n - 1:
        positions.append(zero + n)
    if col > 0:
        positions.append(zero - 1)
    if col < n - 1:
        positions.append(zero + 1)

    for pos in positions:
        board = list(state)
        board[zero], board[pos] = board[pos], board[zero]
        yield tuple(board)


# Generate a solvable state by scrambling the goal
def random_state(n):
    state = goal_state(n)
    previous = None

    for _ in range(random.randint(15, 30)):
        choices = [
            nxt for nxt in neighbors(state, n)
            if nxt != previous
        ]
        nxt = random.choice(choices)
        previous, state = state, nxt

    return state


# Reconstruct the solution path
def reconstruct_path(parent, state):
    path = [state]

    while parent[state] is not None:
        state = parent[state]
        path.append(state)

    return path[::-1]


# 1. Breadth-First Search
def bfs(start, goal, n):
    queue = deque([start])
    parent = {start: None}

    while queue:
        state = queue.popleft()

        if state == goal:
            return reconstruct_path(parent, state)

        for nxt in neighbors(state, n):
            if nxt not in parent:
                if len(parent) >= BFS_STATE_LIMIT:
                    raise StateLimitReached()

                parent[nxt] = state
                queue.append(nxt)

    return None


# 2. Depth-First Search
def dfs(start, goal, n):
    stack = [start]
    parent = {start: None}

    while stack:
        state = stack.pop()

        if state == goal:
            return reconstruct_path(parent, state)

        for nxt in neighbors(state, n):
            if nxt not in parent:
                parent[nxt] = state
                stack.append(nxt)

    return None


# 3. Depth-Limited Search
def dls(start, goal, n, limit=DLS_LIMIT):
    path = [start]
    on_path = {start}

    def visit(state, depth):
        if state == goal:
            return list(path)

        if depth >= limit:
            return None

        for nxt in neighbors(state, n):
            if nxt not in on_path:
                on_path.add(nxt)
                path.append(nxt)

                result = visit(nxt, depth + 1)

                if result is not None:
                    return result

                path.pop()
                on_path.remove(nxt)

        return None

    return visit(start, 0)


# 4. Iterative Deepening Search
def iterative_deepening(start, goal, n):
    max_depth = n * n * n * n

    for limit in range(max_depth + 1):
        result = dls(start, goal, n, limit)

        if result is not None:
            return result

    return None


# Run an algorithm in a separate process
def worker(algorithm, start, goal, n, result_queue):
    functions = {
        "BFS": bfs,
        "DFS": dfs,
        "DLS": lambda s, g, size: dls(
            s, g, size, DLS_LIMIT
        ),
        "ID": iterative_deepening
    }

    began = time.perf_counter()

    try:
        path = functions[algorithm](start, goal, n)

        result_queue.put({
            "outcome": (
                "Success" if path is not None
                else "Failure (Exhausted)"
            ),
            "elapsed": time.perf_counter() - began,
            "path_length": (
                len(path) - 1 if path is not None else ""
            )
        })

    except StateLimitReached:
        result_queue.put({
            "outcome": "State Limit Reached",
            "elapsed": time.perf_counter() - began,
            "path_length": ""
        })

    except Exception as exc:
        result_queue.put({
            "outcome": "Error: " + str(exc),
            "elapsed": time.perf_counter() - began,
            "path_length": ""
        })


# Enforce the timeout
def run_algorithm(algorithm, start, goal, n):
    context = mp.get_context("spawn")
    result_queue = context.Queue()

    process = context.Process(
        target=worker,
        args=(algorithm, start, goal, n, result_queue)
    )

    process.start()
    process.join(TIME_LIMIT)

    if process.is_alive():
        process.terminate()
        process.join()
        result_queue.close()

        return {
            "outcome": "Timeout",
            "elapsed": TIME_LIMIT,
            "path_length": ""
        }

    try:
        return result_queue.get(timeout=2)

    except Exception:
        return {
            "outcome": "Error: no result returned",
            "elapsed": TIME_LIMIT,
            "path_length": ""
        }


# Generate the required graphs
def create_graphs(rows, folder):
    metrics = [
        (
            "time",
            "execution_time.png",
            "Average Execution Time",
            "Seconds"
        ),
        (
            "success",
            "success_rate.png",
            "Success Rate",
            "Percentage"
        ),
        (
            "length",
            "path_length.png",
            "Average Solution Path Length",
            "Moves"
        )
    ]

    for metric, filename, title, ylabel in metrics:
        plt.figure()

        for n in (3, 4):
            values = []

            for algorithm in ALGORITHMS:
                tests = [
                    row for row in rows
                    if row["n"] == n
                    and row["algorithm"] == algorithm
                ]

                if metric == "time":
                    # Timeouts count as 60 seconds.
                    value = (
                        sum(row["elapsed"] for row in tests)
                        / len(tests) if tests else 0
                    )

                elif metric == "success":
                    value = (
                        100 * sum(
                            row["outcome"] == "Success"
                            for row in tests
                        ) / len(tests) if tests else 0
                    )

                else:
                    lengths = [
                        row["path_length"] for row in tests
                        if row["outcome"] == "Success"
                        and row["path_length"] != ""
                    ]

                    value = (
                        sum(lengths) / len(lengths)
                        if lengths else 0
                    )

                values.append(value)

            plt.plot(
                ALGORITHMS,
                values,
                marker="o",
                label=f"n={n}"
            )

        plt.title(title)
        plt.xlabel("Algorithm")
        plt.ylabel(ylabel)

        if metric == "success":
            plt.ylim(0, 100)

        plt.legend()
        plt.tight_layout()
        plt.savefig(folder / filename, dpi=150)
        plt.close()

    print("Created all three graph images.")


# Main experimental program
def main():
    random.seed()

    folder = Path(__file__).resolve().parent
    rows = []

    print("AI Lab 8: Search Algorithm Experiments")
    print("Timeout:", TIME_LIMIT, "seconds per run")
    print("BFS state limit:", BFS_STATE_LIMIT)

    for n in (3, 4):
        goal = goal_state(n)

        states = [
            random_state(n) for _ in range(TESTS)
        ]

        for test_number, start in enumerate(states, 1):
            print(
                f"\nPuzzle {n}x{n}, "
                f"test {test_number}/{TESTS}"
            )

            for algorithm in ALGORITHMS:
                print(
                    "Running", algorithm, "...",
                    flush=True
                )

                result = run_algorithm(
                    algorithm, start, goal, n
                )

                row = {
                    "n": n,
                    "test": test_number,
                    "algorithm": algorithm,
                    "outcome": result["outcome"],
                    "elapsed": round(
                        min(
                            float(result["elapsed"]),
                            TIME_LIMIT
                        ),
                        4
                    ),
                    "path_length": result["path_length"]
                }

                rows.append(row)

                print(
                    row["outcome"],
                    "| Time:", row["elapsed"],
                    "| Moves:",
                    row["path_length"]
                    if row["path_length"] != ""
                    else "N/A"
                )

    # Save all experimental results
    csv_path = folder / "results.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "n",
                "test",
                "algorithm",
                "outcome",
                "elapsed",
                "path_length"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved to results.csv")

    create_graphs(rows, folder)

    print("\nAll experiments completed.")
    print("Check your LAB-8 folder for the CSV and graphs.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
