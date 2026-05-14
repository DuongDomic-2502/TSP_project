import os
import sys
import time
import matplotlib.pyplot as plt

from Algorithms.BnB import BranchAndBound

from util import read_cities, path_cost

# =========================================================
#  HÀM CHẠY THUẬT TOÁN
# =========================================================

def run_algorithm(algorithm, name, output_dir):

    print(f"\n========== {name} ==========")

    start_time = time.time()

    best_route, best_cost = algorithm.run(
        output_dir=output_dir
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Best Cost      : {best_cost:.2f}")
    print(f"Execution Time : {execution_time:.4f} seconds")

    return {
        "name": name,
        "route": best_route,
        "cost": best_cost,
        "time": execution_time
    }

# =========================================================
#  MAIN
# =========================================================

if __name__ == "__main__":

    print("\n===== TSP MENU =====")
    print("1. Branch and Bound")
    print("2. Genetic Algorithm")
    print("3. Ant Colony Optimization")

    choice = int(input("Chọn thuật toán (1-3): "))

    size = int(input("Nhập số thành phố (ví dụ 10, 16, 20): "))

    cities = read_cities(size)

    OUTPUT_ROOT = "results"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    results = []

    # =========================
    # BRANCH AND BOUND
    # =========================
    if choice == 1:
        algo = BranchAndBound(cities)
        name = "Branch and Bound"
        folder = f"{OUTPUT_ROOT}/BnB"

    # =========================
    # GA
    # =========================
    elif choice == 2:
        from Algorithms.GA import GeneticAlgorithm

        algo = GeneticAlgorithm(
            cities=cities,
            population_size=100,
            generations=200,
            mutation_rate=0.01
        )
        name = "Genetic Algorithm"
        folder = f"{OUTPUT_ROOT}/GA"

    # =========================
    # ACO
    # =========================
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
        folder = f"{OUTPUT_ROOT}/ACO"

    else:
        print("Lựa chọn không hợp lệ!")
        sys.exit()

    # =========================
    # RUN
    # =========================
    result = run_algorithm(algo, name, folder)

    print("\n==============================")
    print("       FINAL RESULT")
    print("==============================")

    print(f"Algorithm : {result['name']}")
    print(f"Cost      : {result['cost']:.2f}")
    print(f"Time      : {result['time']:.4f} s")
    
    print("\nBest Route (coordinates):")
    for city in result['route']:
        print(city)
