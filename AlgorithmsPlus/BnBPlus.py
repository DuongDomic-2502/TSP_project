import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from util import path_cost


class BnBPlus:

    def __init__(self, cities):

        self.cities = cities
        self.n = len(cities)

        self.best_route = None
        self.best_cost = float('inf')

        self.distance_matrix = self._compute_distance_matrix()

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

                else:
                    matrix[i][j] = float('inf')

        return matrix

    # =================================================
    # GREEDY INITIAL SOLUTION
    # -> tạo upper bound ban đầu
    # =================================================
    def _greedy_initial_solution(self):

        visited = [False] * self.n

        route = [0]
        visited[0] = True

        current = 0
        total_cost = 0

        # ---------------------------------------------
        # Nearest Neighbor
        # ---------------------------------------------
        for _ in range(self.n - 1):

            next_city = None
            best_dist = float('inf')

            for j in range(self.n):

                if (
                    not visited[j]
                    and self.distance_matrix[current][j]
                    < best_dist
                ):

                    best_dist = (
                        self.distance_matrix[current][j]
                    )

                    next_city = j

            route.append(next_city)

            visited[next_city] = True

            total_cost += best_dist

            current = next_city

        # quay về thành phố đầu
        total_cost += self.distance_matrix[current][0]

        self.best_route = route[:]
        self.best_cost = total_cost

    # =================================================
    # Improved Lower Bound
    # =================================================
    def _lower_bound(self, visited, current):

        unvisited = [
            i for i in range(self.n)
            if i not in visited
        ]

        if not unvisited:

            return self.distance_matrix[current][0]

        lb = 0

        # ---------------------------------------------
        # 2 cạnh nhỏ nhất của mỗi node
        # ---------------------------------------------
        for u in unvisited:

            edges = sorted(
                self.distance_matrix[u][v]
                for v in range(self.n)
                if u != v
            )

            lb += (edges[0] + edges[1]) / 2

        # current -> unvisited nhỏ nhất
        min_current = min(
            self.distance_matrix[current][v]
            for v in unvisited
        )

        # unvisited -> start nhỏ nhất
        min_return = min(
            self.distance_matrix[v][0]
            for v in unvisited
        )

        lb += (min_current + min_return) / 2

        return lb

    # =================================================
    # Branch and Bound
    # =================================================
    def _branch_and_bound(
        self,
        current,
        visited,
        path,
        current_cost,
        frames,
        output_dir,
        frame_count
    ):

        # ---------------------------------------------
        # Đã đi hết
        # ---------------------------------------------
        if len(visited) == self.n:

            total_cost = (
                current_cost
                + self.distance_matrix[current][0]
            )

            if total_cost < self.best_cost:

                self.best_cost = total_cost
                self.best_route = path[:]

                # -------------------------------------
                # Save GIF frame
                # -------------------------------------
                if output_dir and frame_count[0] < 50:

                    plt.figure(figsize=(8, 6))

                    route = [
                        self.cities[i]
                        for i in path
                    ]

                    x = [c.x for c in route] + [route[0].x]
                    y = [c.y for c in route] + [route[0].y]

                    plt.plot(x, y, 'ro-')

                    plt.title(
                        f'BnB TSP | Cost = {total_cost:.2f}'
                    )

                    frame_path = (
                        f'{output_dir}/bnb_'
                        f'{frame_count[0]}.png'
                    )

                    plt.savefig(frame_path)
                    plt.close()

                    frames.append(
                        imageio.imread(frame_path)
                    )

                    os.remove(frame_path)

                    frame_count[0] += 1

            return

        # =================================================
        # Loại bỏ hoán vị dư thừa:
        # sort theo khoảng cách gần nhất
        # -> node tốt xét trước
        # =================================================
        candidates = []

        for next_city in range(self.n):

            if next_city not in visited:

                dist = (
                    self.distance_matrix[current][next_city]
                )

                candidates.append(
                    (dist, next_city)
                )

        # ---------------------------------------------
        # xét thành phố gần trước
        # -> tìm best sớm hơn
        # -> pruning mạnh hơn
        # ---------------------------------------------
        candidates.sort()

        for _, next_city in candidates:

            new_cost = (
                current_cost
                + self.distance_matrix[current][next_city]
            )

            lb = self._lower_bound(
                visited + [next_city],
                next_city
            )

            # -----------------------------------------
            # PRUNING
            # -----------------------------------------
            if new_cost + lb >= self.best_cost:
                continue

            path.append(next_city)

            self._branch_and_bound(
                next_city,
                visited + [next_city],
                path,
                new_cost,
                frames,
                output_dir,
                frame_count
            )

            path.pop()

    # =================================================
    # RUN
    # =================================================
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        frames = []
        frame_count = [0]

        # =================================================
        # GREEDY -> upper bound ban đầu
        # =================================================
        self._greedy_initial_solution()

        # =================================================
        # BnB
        # =================================================
        self._branch_and_bound(
            current=0,
            visited=[0],
            path=[0],
            current_cost=0,
            frames=frames,
            output_dir=output_dir,
            frame_count=frame_count
        )

        # =================================================
        # Save GIF
        # =================================================
        if output_dir and frames:

            imageio.mimsave(
                f'{output_dir}/bnb_process.gif',
                frames,
                fps=5
            )

        # =================================================
        # Save final route
        # =================================================
        if output_dir and self.best_route:

            plt.figure(figsize=(8, 6))

            route = [
                self.cities[i]
                for i in self.best_route
            ]

            x = [c.x for c in route] + [route[0].x]
            y = [c.y for c in route] + [route[0].y]

            plt.plot(x, y, 'ro-')

            plt.title(
                f'Improved BnB Final Route\n'
                f'Cost = {self.best_cost:.2f}'
            )

            plt.savefig(
                f'{output_dir}/final_route.png'
            )

            plt.close()

        return (
            [self.cities[i] for i in self.best_route],
            self.best_cost
        )