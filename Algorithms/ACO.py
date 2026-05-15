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
    # Xây lộ trình cho một kiến
    # Khớp pseudocode:
    #   visited = {điểm_xuất_phát}
    #   while chưa thăm hết:
    #       P[j] = τ^α × η^β / Σ P
    #       Chọn j_tiếp_theo theo xác suất P
    # ----------------------------
    def _build_route(self, start):
        visited = {start}
        path = [start]
        cost = 0

        while len(visited) < self.n:
            current = path[-1]
            unvisited = [j for j in range(self.n) if j not in visited]

            # Tính xác suất cho từng node chưa thăm
            probs = []
            for j in unvisited:
                dist = max(self._dist(current, j), 1e-9)
                tau = self.pheromone[current][j] ** self.alpha
                eta = (1.0 / dist) ** self.beta
                probs.append(tau * eta)

            # Chuẩn hóa xác suất
            probs = np.array(probs)
            if probs.sum() == 0:
                probs = np.ones(len(unvisited)) / len(unvisited)
            else:
                probs /= probs.sum()

            # Chọn node tiếp theo theo xác suất
            nxt = np.random.choice(unvisited, p=probs)

            cost += self._dist(current, nxt)
            path.append(nxt)
            visited.add(nxt)

        # Quay về điểm xuất phát
        cost += self._dist(path[-1], path[0])
        path.append(path[0])

        return path, cost

    # ----------------------------
    # Chạy ACO
    # Khớp pseudocode:
    #   Lặp T lần:
    #       Với mỗi kiến k: Xây_lộ_trình(k)
    #       Evaporate pheromone
    #       Deposit pheromone
    #       Cập nhật best SAU khi tất cả kiến xong
    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for _ in range(self.n_iterations):

            all_paths = []
            all_costs = []

            # --------------------
            # Bước 1: Tất cả kiến xây lộ trình
            # --------------------
            for k in range(self.n_ants):
                # Mỗi kiến xuất phát từ một thành phố theo thứ tự
                start = k % self.n
                path, cost = self._build_route(start)

                all_paths.append(path)
                all_costs.append(cost)

            # --------------------
            # Bước 2: Cập nhật best SAU khi tất cả kiến xong
            # (khớp pseudocode: "Cập nhật lời giải tốt nhất")
            # --------------------
            best_idx = int(np.argmin(all_costs))
            if all_costs[best_idx] < self.best_cost:
                self.best_cost = all_costs[best_idx]
                self.best_path = all_paths[best_idx][:-1]  # bỏ node lặp cuối

            # --------------------
            # Bước 3: Bay hơi pheromone
            # --------------------
            self.pheromone *= (1 - self.rho)

            # --------------------
            # Bước 4: Deposit pheromone
            # --------------------
            for path, cost in zip(all_paths, all_costs):
                deposit = self.Q / cost
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    self.pheromone[a][b] += deposit
                    self.pheromone[b][a] += deposit

        # --------------------
        # Lưu kết quả
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