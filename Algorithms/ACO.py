import os
import numpy as np
import matplotlib.pyplot as plt


class ACO:
    def __init__(self, cities,
                 n_ants=10,
                 n_iterations=100,
                 alpha=1.0,
                 beta=2.0,
                 evaporation_rate=0.5,
                 Q=1.0):

        self.cities = cities
        self.n = len(cities)

        self.n_ants = n_ants
        self.n_iterations = n_iterations

        self.alpha = alpha
        self.beta = beta

        self.rho = evaporation_rate
        self.Q = Q

        self.pheromone = np.ones((self.n, self.n))

        self.best_path = None
        self.best_cost = float('inf')

    # ----------------------------
    def _dist(self, i, j):
        return self.cities[i].distance(self.cities[j])

    # ----------------------------
    def _select_next(self, current, unvisited):
        probs = []

        for j in unvisited:
            dist = self._dist(current, j)
            dist = max(dist, 1e-9)

            tau = self.pheromone[current][j] ** self.alpha
            eta = (1.0 / dist) ** self.beta

            probs.append(tau * eta)

        probs = np.array(probs)

        if probs.sum() == 0:
            return np.random.choice(unvisited)

        probs /= probs.sum()

        return np.random.choice(unvisited, p=probs)

    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for _ in range(self.n_iterations):

            all_paths = []
            all_costs = []

            # --------------------
            # Construct solutions
            # --------------------
            for _ in range(self.n_ants):

                start = np.random.randint(self.n)

                path = [start]
                visited = set([start])

                cost = 0

                while len(visited) < self.n:

                    current = path[-1]
                    unvisited = list(set(range(self.n)) - visited)

                    nxt = self._select_next(current, unvisited)

                    cost += self._dist(current, nxt)

                    path.append(nxt)
                    visited.add(nxt)

                # return to start
                cost += self._dist(path[-1], path[0])
                path.append(path[0])

                all_paths.append(path)
                all_costs.append(cost)

                # update best
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = path[:-1]

            # --------------------
            # Evaporation
            # --------------------
            self.pheromone *= (1 - self.rho)

            # --------------------
            # Deposit pheromone
            # --------------------
            for path, cost in zip(all_paths, all_costs):
                deposit = self.Q / cost

                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    self.pheromone[a][b] += deposit
                    self.pheromone[b][a] += deposit

        # --------------------
        # Save final result
        # --------------------
        if output_dir and self.best_path:
            self._save_final(output_dir)

        return (
            [self.cities[i] for i in self.best_path],
            self.best_cost
        )

    # ----------------------------
    def _save_final(self, output_dir):
        plt.figure(figsize=(7, 5))

        x = [self.cities[i].x for i in self.best_path]
        y = [self.cities[i].y for i in self.best_path]

        x.append(x[0])
        y.append(y[0])

        plt.plot(x, y, 'o-', color='red')
        plt.title(f"Best ACO route | cost = {self.best_cost:.4f}")

        plt.savefig(f"{output_dir}/final_route.png")
        plt.close()