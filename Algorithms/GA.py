import os
import numpy as np
import matplotlib.pyplot as plt
import random


class GA:
    def __init__(self, cities,
                 pop_size=50,
                 n_generations=200,
                 crossover_rate=0.8,
                 mutation_rate=0.2):

        self.cities = cities
        self.n = len(cities)

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cr = crossover_rate
        self.mr = mutation_rate

        self.best_route = None
        self.best_cost = float('inf')

    # ----------------------------
    def _dist(self, i, j):
        return self.cities[i].distance(self.cities[j])

    def _path_cost(self, path):
        cost = 0
        for i in range(self.n - 1):
            cost += self._dist(path[i], path[i + 1])
        cost += self._dist(path[-1], path[0])
        return cost

    # ----------------------------
    def _init_population(self):
        base = list(range(self.n))
        return [random.sample(base, self.n) for _ in range(self.pop_size)]

    # ----------------------------
    def _selection(self, pop, costs):
        idx = np.argsort(costs)
        return [pop[i] for i in idx[:self.pop_size // 2]]

    # ----------------------------
    def _crossover(self, p1, p2):
        if random.random() > self.cr:
            return p1[:]

        a, b = sorted(random.sample(range(self.n), 2))

        child = [-1] * self.n
        child[a:b] = p1[a:b]

        fill = [x for x in p2 if x not in child]

        j = 0
        for i in range(self.n):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1

        return child

    # ----------------------------
    def _mutate(self, path):
        if random.random() < self.mr:
            i, j = random.sample(range(self.n), 2)
            path[i], path[j] = path[j], path[i]
        return path

    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        population = self._init_population()

        for _ in range(self.n_generations):

            costs = [self._path_cost(p) for p in population]

            # update best
            best_idx = np.argmin(costs)
            if costs[best_idx] < self.best_cost:
                self.best_cost = costs[best_idx]
                self.best_route = population[best_idx][:]

            # selection
            selected = self._selection(population, costs)

            # next generation
            new_pop = selected[:]

            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(selected, 2)

                child = self._crossover(p1, p2)
                child = self._mutate(child)

                new_pop.append(child)

            population = new_pop

        # ----------------------------
        # SAVE FINAL ONLY
        # ----------------------------
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
        plt.title(f"GA TSP | best cost = {self.best_cost:.4f}")

        plt.savefig(f"{output_dir}/ga_final.png")
        plt.close()