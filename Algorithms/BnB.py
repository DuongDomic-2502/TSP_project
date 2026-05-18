import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio


class BnB:

    def __init__(self, cities):

        self.cities = cities
        self.n = len(cities)

        self.best_route = None
        self.best_cost = float('inf')

        self.distance_matrix = (
            self._compute_distance_matrix()
        )

        # =============================================
        # Precompute: 2 cạnh nhỏ nhất của mỗi đỉnh
        # Dùng cho lower bound — tính 1 lần duy nhất
        # =============================================
        self._min2 = self._precompute_min2()

    # -------------------------------------------------
    # Distance Matrix — vectorized
    # -------------------------------------------------
    def _compute_distance_matrix(self):

        coords = np.array(
            [[c.x, c.y] for c in self.cities]
        )

        diff = (
            coords[:, np.newaxis, :]
            - coords[np.newaxis, :, :]
        )

        matrix = np.sqrt((diff ** 2).sum(axis=2))

        # Diagonal = inf để tránh self-loop
        np.fill_diagonal(matrix, float('inf'))

        return matrix

    # -------------------------------------------------
    # Precompute 2 cạnh nhỏ nhất mỗi đỉnh
    # Dùng trong lower bound Held-Karp
    # -------------------------------------------------
    def _precompute_min2(self):

        min2 = []

        for i in range(self.n):

            row = sorted(
                self.distance_matrix[i][j]
                for j in range(self.n)
                if j != i
            )

            min2.append((row[0], row[1]))

        return min2

    # -------------------------------------------------
    # Lower Bound — Held-Karp chuẩn
    #
    # Công thức:
    #   LB = current_cost
    #      + Σ (min1[u] + min2[u]) / 2   (u chưa thăm)
    #      + min cạnh từ current → unvisited
    #      + min cạnh từ unvisited → start (0)
    #
    # KHÔNG tính đôi nhờ chia 2 đúng chỗ
    # -------------------------------------------------
    def _lower_bound(
        self,
        current_cost,
        current,
        unvisited_set
    ):

        if not unvisited_set:
            return (
                current_cost
                + self.distance_matrix[current][0]
            )

        lb = current_cost

        for u in unvisited_set:
            lb += (self._min2[u][0] + self._min2[u][1]) / 2

        min_from_current = min(
            self.distance_matrix[current][v]
            for v in unvisited_set
        )
        lb += min_from_current / 2

        min_to_start = min(
            self.distance_matrix[v][0]
            for v in unvisited_set
        )
        lb += min_to_start / 2

        return lb

    # -------------------------------------------------
    # Branch and Bound — đệ quy thuần
    # -------------------------------------------------
    def _branch_and_bound(
        self,
        current,
        visited_set,
        path,
        current_cost,
        unvisited_set,
        frames,
        output_dir,
        frame_count
    ):

        # ---------------------------------------------
        # Base case: đã thăm hết
        # ---------------------------------------------
        if not unvisited_set:

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
        # ---------------------------------------------
        # Pruning — lower bound vượt best hiện tại
        # ---------------------------------------------
        lb = self._lower_bound(
            current_cost,
            current,
            unvisited_set
        )
        if lb >= self.best_cost:
            return
        for next_city in unvisited_set:
            new_cost = (
                current_cost
                + self.distance_matrix[current][next_city]
            )
            new_unvisited = unvisited_set - {next_city}
            branch_lb = self._lower_bound(
                new_cost,
                next_city,
                new_unvisited
            )
            if branch_lb >= self.best_cost:
                continue
            path.append(next_city)
            visited_set.add(next_city)
            self._branch_and_bound(
                next_city,
                visited_set,
                path,
                new_cost,
                new_unvisited,
                frames,
                output_dir,
                frame_count
            )
            path.pop()
            visited_set.discard(next_city)

    # -------------------------------------------------
    # Run
    # -------------------------------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.best_cost = float('inf')
        self.best_route = None

        frames = []
        frame_count = [0]

        self._branch_and_bound(
            current=0,
            visited_set={0},
            path=[0],
            current_cost=0,
            unvisited_set=set(range(1, self.n)),
            frames=frames,
            output_dir=output_dir,
            frame_count=frame_count
        )

        # ---------------------------------------------
        # Save GIF
        # ---------------------------------------------
        if output_dir and frames:

            imageio.mimsave(
                f'{output_dir}/bnb_process.gif',
                frames,
                fps=5
            )

        # ---------------------------------------------
        # Save final route
        # ---------------------------------------------
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
                f'Branch and Bound Final Route\n'
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