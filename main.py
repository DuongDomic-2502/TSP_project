import os
import sys
import time
from util import read_cities


def run_algorithm(algorithm, name, output_dir):

    print(f"\n========== {name} ==========")

    start_time = time.time()

    best_route, best_cost = algorithm.run(
        output_dir=output_dir
    )

    end_time = time.time()

    print(f"Best Cost      : {best_cost:.2f}")
    print(f"Execution Time : {end_time - start_time:.4f} seconds")

    return {
        "name": name,
        "route": best_route,
        "cost": best_cost,
        "time": end_time - start_time
    }


if __name__ == "__main__":

    print("\n===== TSP MENU =====")
    print("1. Branch and Bound")
    print("2. Genetic Algorithm")
    print("3. Ant Colony Optimization")
    print("4. Simulated Annealing")
    print("5. Branch and Bound + 3-opt")

    try:
        choice = int(input("Chọn thuật toán (1-5): "))
        size = int(input("Nhập số thành phố: "))
    except ValueError:
        print("Input không hợp lệ!")
        sys.exit()

    cities = read_cities(size)

    OUTPUT_ROOT = "results"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # =========================
    # SELECT ALGORITHM
    # =========================
    if choice == 1:
        from Algorithms.BnB import BranchAndBound
        algo = BranchAndBound(cities)
        name = "Branch and Bound"
        output_dir = f"{OUTPUT_ROOT}/BnB"

    elif choice == 2:
        from Algorithms.GA import GA

        algo = GA(
            cities=cities,
            pop_size=100,
            n_generations=200,
            mutation_rate=0.01
        )

        name = "Genetic Algorithm"
        output_dir = f"{OUTPUT_ROOT}/GA"

    elif choice == 3:
        from Algorithms.ACO import ACO

        algo = ACO(
            cities=cities,
            n_ants=20,
            n_iterations=100,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.5,
            Q=100
        )

        name = "Ant Colony Optimization"
        output_dir = f"{OUTPUT_ROOT}/ACO"

    elif choice == 4:
        from Algorithms.SA import SA

        algo = SA(
            cities=cities,
            initial_temp=1000.0,
            cooling_rate=0.995,
            t_min=1e-8,
            steps_per_temp=100   # L bước tại mỗi nhiệt độ
        )

        name = "Simulated Annealing"
        output_dir = f"{OUTPUT_ROOT}/SA"

    elif choice == 5:
        from AlgorithmsPlus.BnBPlus import BnBPlus
        algo = BnBPlus(cities)
        name = "Branch and Bound + 3-opt"
        output_dir = f"{OUTPUT_ROOT}/BnBPlus"

    else:
        print("Lựa chọn không hợp lệ!")
        sys.exit()

    # =========================
    # RUN
    # =========================
    result = run_algorithm(
        algo,
        name,
        output_dir
    )

    print("\n==============================")
    print("       FINAL RESULT")
    print("==============================")

    print(f"Algorithm : {result['name']}")
    print(f"Cost      : {result['cost']:.2f}")
    print(f"Time      : {result['time']:.4f} s")

    print("\nBest Route:")
    for city in result['route']:
        print(city)