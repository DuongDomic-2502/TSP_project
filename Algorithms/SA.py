import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math


class SA:
    def __init__(self, cities,
                 initial_temp=1000,
                 cooling_rate=0.995,
                 n_iterations=5000):

        self.cities = cities
        self.n = len(cities)

        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_iterations = n_iterations

        self.best_route = None
        self.best_cost = float('inf')

    # ----------------------------
    def _dist(self, i, j):
        return self.cities[i].distance(self.cities[j])

    def _cost(self, path):
        cost = 0
        for i in range(self.n - 1):
            cost += self._dist(path[i], path[i + 1])
        cost += self._dist(path[-1], path[0])
        return cost

    # ----------------------------
    def _neighbor(self, path):
        a, b = random.sample(range(self.n), 2)
        new_path = path[:]
        new_path[a], new_path[b] = new_path[b], new_path[a]
        return new_path

    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # reset temperature mỗi lần chạy
        T = self.initial_temp

        # init solution
        current = list(range(self.n))
        random.shuffle(current)

        current_cost = self._cost(current)

        self.best_route = current[:]
        self.best_cost = current_cost

        # SA loop
        for _ in range(self.n_iterations):

            if T < 1e-8:   # tránh chia 0
                break

            candidate = self._neighbor(current)
            candidate_cost = self._cost(candidate)

            delta = candidate_cost - current_cost

            accept_prob = math.exp(-delta / T) if delta > 0 else 1.0

            if random.random() < accept_prob:
                current = candidate
                current_cost = candidate_cost

                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_route = current[:]

            # cooling
            T *= self.cooling_rate

        # save final result
        if output_dir and self.best_route:
            self._save_final(output_dir)

        return (
            [self.cities[i] for i in self.best_route],
            self.best_cost
        )

    # ----------------------------
    def _save_final(self, output_dir):
        plt.figure(figsize=(7, 5))

        route = [self.cities[i] for i in self.best_route]

        x = [c.x for c in route] + [route[0].x]
        y = [c.y for c in route] + [route[0].y]

        plt.plot(x, y, 'o-')
        plt.title(f"SA TSP | best cost = {self.best_cost:.4f}")

        plt.savefig(f"{output_dir}/sa_final.png")
        plt.close()