import os
import numpy as np
import matplotlib.pyplot as plt


class ACOPlus:

    def __init__(self, cities,
                 n_ants=0,
                 n_iterations=0,
                 alpha=0.0,
                 beta=0.0,
                 evaporation_rate=0.0,
                 Q=0):

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

    # =================================================
    # Distance
    # =================================================
    def _dist(self, i, j):

        return self.cities[i].distance(
            self.cities[j]
        )

    # =================================================
    # Route Cost
    # =================================================
    def _route_cost(self, path):

        cost = 0

        for i in range(len(path) - 1):

            cost += self._dist(
                path[i],
                path[i + 1]
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

            for i in range(1, len(best) - 2):

                for j in range(i + 1, len(best) - 1):

                    new_path = (
                        best[:i]
                        + best[i:j + 1][::-1]
                        + best[j + 1:]
                    )

                    old_cost = self._route_cost(best)

                    new_cost = self._route_cost(
                        new_path
                    )

                    if new_cost < old_cost:

                        best = new_path
                        improved = True

        return best

    # =================================================
    # Build Route
    # =================================================
    def _build_route(self, start):

        visited = {start}

        path = [start]

        current = start

        while len(visited) < self.n:

            unvisited = [
                j for j in range(self.n)
                if j not in visited
            ]

            probs = []

            for j in unvisited:

                dist = max(
                    self._dist(current, j),
                    1e-9
                )

                tau = (
                    self.pheromone[current][j]
                    ** self.alpha
                )

                eta = (
                    (1.0 / dist)
                    ** self.beta
                )

                probs.append(tau * eta)

            probs = np.array(probs)

            if probs.sum() == 0:

                probs = (
                    np.ones(len(unvisited))
                    / len(unvisited)
                )

            else:

                probs /= probs.sum()

            nxt = np.random.choice(
                unvisited,
                p=probs
            )

            path.append(nxt)

            visited.add(nxt)

            current = nxt

        path.append(start)

        return path

    # =================================================
    # RUN
    # =================================================
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for iteration in range(self.n_iterations):

            all_paths = []
            all_costs = []

            # -----------------------------------------
            # Mỗi ant xây route  (KHÔNG 2-opt ở đây)
            # -----------------------------------------
            for k in range(self.n_ants):

                start = k % self.n

                path = self._build_route(start)

                cost = self._route_cost(path)

                all_paths.append(path)

                all_costs.append(cost)

            # -----------------------------------------
            # Tìm best ant của iteration
            # -----------------------------------------
            best_idx = int(np.argmin(all_costs))

            iter_best_path = all_paths[best_idx]
            iter_best_cost = all_costs[best_idx]

            # -----------------------------------------
            # Chỉ apply 2-opt cho best ant
            # thay vì chạy cho toàn bộ n_ants kiến
            # -----------------------------------------
            opt_path = self._two_opt(iter_best_path)
            opt_cost = self._route_cost(opt_path)

            if opt_cost < iter_best_cost:
                iter_best_path = opt_path
                iter_best_cost = opt_cost

            # -----------------------------------------
            # Update BEST
            # -----------------------------------------
            if iter_best_cost < self.best_cost:

                self.best_cost = iter_best_cost

                self.best_path = iter_best_path[:-1]

            # -----------------------------------------
            # Evaporation
            # -----------------------------------------
            self.pheromone *= (1 - self.rho)

            # -----------------------------------------
            # Deposit Pheromone
            # -----------------------------------------
            for path, cost in zip(
                all_paths,
                all_costs
            ):

                deposit = self.Q / cost

                for i in range(len(path) - 1):

                    a = path[i]
                    b = path[i + 1]

                    self.pheromone[a][b] += deposit
                    self.pheromone[b][a] += deposit

        # =================================================
        # Save Final
        # =================================================
        if output_dir and self.best_path:

            self._save_final(output_dir)

        return (
            [self.cities[i] for i in self.best_path],
            self.best_cost
        )

    # =================================================
    # SAVE FINAL
    # =================================================
    def _save_final(self, output_dir):

        plt.figure(figsize=(7, 5))

        x = [
            self.cities[i].x
            for i in self.best_path
        ]

        y = [
            self.cities[i].y
            for i in self.best_path
        ]

        x.append(x[0])
        y.append(y[0])

        plt.plot(
            x,
            y,
            'o-',
            color='blue'
        )

        plt.title(
            f"ACO + 2OPT\n"
            f"Cost = {self.best_cost:.4f}"
        )

        plt.savefig(
            f"{output_dir}/ACO_2OPT_final.png"
        )

        plt.close()