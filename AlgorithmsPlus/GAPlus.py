import os
import numpy as np
import matplotlib.pyplot as plt
import random


class GAPlus:

    def __init__(self, cities,
                 pop_size=0,
                 n_generations=0,
                 crossover_rate=0,
                 mutation_rate=0):

        self.cities = cities
        self.n = len(cities)

        # =========================================
        # Optimized parameters for n < 300
        # =========================================
        self.pop_size = pop_size
        self.n_generations = n_generations

        self.cr = crossover_rate
        self.mr = mutation_rate

        self.best_route = None
        self.best_cost = float('inf')

        # cache distance matrix
        self.distance_matrix = (
            self._compute_distance_matrix()
        )

    # =================================================
    # Distance Matrix
    # =================================================
    def _compute_distance_matrix(self):

        matrix = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):

                if i != j:

                    matrix[i][j] = (
                        self.cities[i].distance(
                            self.cities[j]
                        )
                    )

        return matrix

    # =================================================
    # Distance
    # =================================================
    def _dist(self, i, j):

        return self.distance_matrix[i][j]

    # =================================================
    # Path Cost
    # =================================================
    def _path_cost(self, path):

        cost = 0

        for i in range(self.n - 1):

            cost += self._dist(
                path[i],
                path[i + 1]
            )

        cost += self._dist(
            path[-1],
            path[0]
        )

        return cost

    # =================================================
    # 2-OPT
    # =================================================
    def _two_opt(self, path):

        best = path[:]

        improved = True

        while improved:

            improved = False

            for i in range(1, self.n - 2):

                for j in range(i + 1, self.n):

                    if j - i == 1:
                        continue

                    new_path = (
                        best[:i]
                        + best[i:j][::-1]
                        + best[j:]
                    )

                    old_cost = (
                        self._path_cost(best)
                    )

                    new_cost = (
                        self._path_cost(new_path)
                    )

                    if new_cost < old_cost:

                        best = new_path
                        improved = True

        return best

    # =================================================
    # Init Population
    # =================================================
    def _init_population(self):

        base = list(range(self.n))

        return [
            random.sample(base, self.n)
            for _ in range(self.pop_size)
        ]

    # =================================================
    # Tournament Selection
    # =================================================
    def _selection(self, pop, costs, k=5):

        candidates = random.sample(
            range(len(pop)),
            k
        )

        best = min(
            candidates,
            key=lambda i: costs[i]
        )

        return pop[best][:]

    # =================================================
    # Order Crossover (OX)
    # =================================================
    def _crossover(self, p1, p2):

        if random.random() > self.cr:

            return p1[:], p2[:]

        a, b = sorted(
            random.sample(range(self.n), 2)
        )

        def ox(parent_a, parent_b):

            child = [-1] * self.n

            child[a:b] = parent_a[a:b]

            fill = [
                x for x in parent_b
                if x not in child
            ]

            j = 0

            for i in range(self.n):

                if child[i] == -1:

                    child[i] = fill[j]

                    j += 1

            return child

        return ox(p1, p2), ox(p2, p1)

    # =================================================
    # Inversion Mutation
    # =================================================
    def _mutate(self, path):

        if random.random() < self.mr:

            i, j = sorted(
                random.sample(range(self.n), 2)
            )

            path[i:j] = path[i:j][::-1]

        return path

    # =================================================
    # RUN
    # =================================================
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # ---------------------------------------------
        # Init population
        # ---------------------------------------------
        population = self._init_population()

        costs = [
            self._path_cost(p)
            for p in population
        ]

        # =================================================
        # GENERATIONS
        # =================================================
        for generation in range(
            self.n_generations
        ):

            new_pop = []

            # ---------------------------------------------
            # ELITISM
            # giữ lại best cá thể
            # ---------------------------------------------
            elite_idx = int(np.argmin(costs))

            elite = population[elite_idx][:]

            new_pop.append(elite)

            # ---------------------------------------------
            # Create offspring
            # ---------------------------------------------
            while len(new_pop) < self.pop_size:

                parent1 = self._selection(
                    population,
                    costs
                )

                parent2 = self._selection(
                    population,
                    costs
                )

                child1, child2 = (
                    self._crossover(
                        parent1,
                        parent2
                    )
                )

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                # =========================================
                # APPLY 2-OPT
                # chỉ apply cho child tốt hơn
                # để tăng tốc
                # =========================================
                if random.random() < 0.3:

                    child1 = self._two_opt(
                        child1
                    )

                if random.random() < 0.3:

                    child2 = self._two_opt(
                        child2
                    )

                new_pop.append(child1)

                if len(new_pop) < self.pop_size:

                    new_pop.append(child2)

            # ---------------------------------------------
            # Update population
            # ---------------------------------------------
            population = new_pop

            costs = [
                self._path_cost(p)
                for p in population
            ]

            # ---------------------------------------------
            # Update BEST
            # ---------------------------------------------
            best_idx = int(np.argmin(costs))

            if costs[best_idx] < self.best_cost:

                self.best_cost = (
                    costs[best_idx]
                )

                self.best_route = (
                    population[best_idx][:]
                )

            print(
                f"Generation {generation+1} | "
                f"Best = {self.best_cost:.2f}"
            )

        # =================================================
        # SAVE FINAL
        # =================================================
        if output_dir and self.best_route:

            self._save_final(output_dir)

        return (
            [self.cities[i]
             for i in self.best_route],
            self.best_cost
        )

    # =================================================
    # SAVE FINAL
    # =================================================
    def _save_final(self, output_dir):

        plt.figure(figsize=(8, 6))

        route = [
            self.cities[i]
            for i in self.best_route
        ]

        x = [c.x for c in route]
        y = [c.y for c in route]

        x.append(x[0])
        y.append(y[0])

        plt.plot(x, y, 'o-')

        plt.title(
            f"GA + 2OPT\n"
            f"Cost = {self.best_cost:.2f}"
        )

        plt.savefig(
            f"{output_dir}/GA_2OPT_final.png"
        )

        plt.close()