import os
import numpy as np
import matplotlib.pyplot as plt


class BranchAndBound:
    def __init__(self, cities):
        self.cities = cities
        self.n = len(cities)

        self.best_route = None
        self.best_cost = float('inf')

        self.distance_matrix = self._compute_distance_matrix()

    # ----------------------------
    # Distance matrix
    # ----------------------------
    def _compute_distance_matrix(self):
        matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    matrix[i][j] = float('inf')
                else:
                    matrix[i][j] = self.cities[i].distance(self.cities[j])
        return matrix

    # ----------------------------
    # MST (Prim) lower bound
    # ----------------------------
    def _mst_cost(self, nodes):
        if len(nodes) <= 1:
            return 0

        nodes = list(nodes)
        in_mst = set([nodes[0]])
        total_cost = 0

        while len(in_mst) < len(nodes):
            best_edge = float('inf')
            best_v = None

            for u in in_mst:
                for v in nodes:
                    if v not in in_mst:
                        if self.distance_matrix[u][v] < best_edge:
                            best_edge = self.distance_matrix[u][v]
                            best_v = v

            in_mst.add(best_v)
            total_cost += best_edge

        return total_cost

    # ----------------------------
    # Lower bound
    # ----------------------------
    def _lower_bound(self, current, visited):
        unvisited = [i for i in range(self.n) if i not in visited]

        if not unvisited:
            return self.distance_matrix[current][visited[0]]

        lb = 0

        lb += self._mst_cost(unvisited)

        lb += min(self.distance_matrix[current][j] for j in unvisited)

        start = visited[0]
        lb += min(self.distance_matrix[j][start] for j in unvisited)

        return lb

    # ----------------------------
    # Branch and Bound DFS
    # ----------------------------
    def _bnb(self, current, visited, path, current_cost):

        if len(visited) == self.n:
            total_cost = current_cost + self.distance_matrix[current][path[0]]

            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_route = path[:]

            return

        for nxt in range(self.n):
            if nxt not in visited:

                new_cost = current_cost + self.distance_matrix[current][nxt]

                lb = new_cost + self._lower_bound(nxt, visited + [nxt])

                if lb >= self.best_cost:
                    continue

                path.append(nxt)
                visited.append(nxt)

                self._bnb(nxt, visited, path, new_cost)

                path.pop()
                visited.pop()

    # ----------------------------
    # Run
    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.best_cost = float('inf')
        self.best_route = None

        start = 0
        self._bnb(start, [start], [start], 0)

        # save final route
        if output_dir and self.best_route:
            plt.figure(figsize=(7, 5))

            route = [self.cities[i] for i in self.best_route]

            x = [c.x for c in route] + [route[0].x]
            y = [c.y for c in route] + [route[0].y]

            plt.plot(x, y, 'o-')
            plt.title(f"Best TSP route | cost = {self.best_cost:.4f}")

            plt.savefig(f"{output_dir}/final_route.png")
            plt.close()

        return (
            [self.cities[i] for i in self.best_route],
            self.best_cost
        )