import os
import numpy as np
import matplotlib.pyplot as plt
import random


class GA:
    def __init__(self, cities,
                 pop_size=0,
                 n_generations=0,
                 crossover_rate=0,
                 mutation_rate=0):

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
    # Khởi tạo quần thể ngẫu nhiên
    # Khớp pseudocode: "Quần_thể = Khởi_tạo_ngẫu_nhiên(N)"
    # ----------------------------
    def _init_population(self):
        base = list(range(self.n))
        return [random.sample(base, self.n) for _ in range(self.pop_size)]

    # ----------------------------
    # Tournament selection
    # Khớp pseudocode: "cha = Chọn_lọc(Quần_thể)"
    # ----------------------------
    def _selection(self, pop, costs, k=3):
        candidates = random.sample(range(len(pop)), k)
        best = min(candidates, key=lambda i: costs[i])
        return pop[best][:]

    # ----------------------------
    # Order Crossover (OX)
    # Khớp pseudocode:
    #   "Nếu random() < p_cross: con1, con2 = Lai_ghép(cha, mẹ)"
    #   "Ngược lại: con1, con2 = cha, mẹ"
    # ----------------------------
    def _crossover(self, p1, p2):
        if random.random() > self.cr:
            return p1[:], p2[:]

        a, b = sorted(random.sample(range(self.n), 2))

        def ox(parent_a, parent_b):
            child = [-1] * self.n
            child[a:b] = parent_a[a:b]
            fill = [x for x in parent_b if x not in child]
            j = 0
            for i in range(self.n):
                if child[i] == -1:
                    child[i] = fill[j]
                    j += 1
            return child

        return ox(p1, p2), ox(p2, p1)

    # ----------------------------
    # Swap mutation
    # Khớp pseudocode:
    #   "Nếu random() < p_mut: con = Đột_biến(con)"
    # ----------------------------
    def _mutate(self, path):
        if random.random() < self.mr:
            i, j = random.sample(range(self.n), 2)
            path[i], path[j] = path[j], path[i]
        return path

    # ----------------------------
    # Chạy GA
    # Khớp pseudocode:
    #   Quần_thể = Khởi_tạo_ngẫu_nhiên(N)
    #   Đánh_giá fitness
    #   Lặp T lần:
    #       Giữ lại elite (cá thể tốt nhất)
    #       Quần_thể_mới = [elite]
    #       while |Quần_thể_mới| < N:
    #           cha = Chọn_lọc(...)
    #           mẹ = Chọn_lọc(...)
    #           con1, con2 = Lai_ghép(cha, mẹ)
    #           con1 = Đột_biến(con1)
    #           con2 = Đột_biến(con2)
    #           Quần_thể_mới.append(con1, con2)
    #       Đánh_giá fitness Quần_thể_mới
    #       Cập nhật best
    #       Quần_thể = Quần_thể_mới
    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Khởi tạo quần thể
        population = self._init_population()

        # Đánh giá fitness ban đầu
        costs = [self._path_cost(p) for p in population]

        for _ in range(self.n_generations):

            # --------------------
            # Elitism: giữ lại cá thể tốt nhất
            # đảm bảo best không bao giờ tệ hơn
            # --------------------
            elite_idx = int(np.argmin(costs))
            elite = population[elite_idx][:]
            new_pop = [elite]

            # --------------------
            # Tạo phần còn lại từ con cái
            # --------------------
            while len(new_pop) < self.pop_size:

                # Chọn lọc: tournament selection
                cha = self._selection(population, costs)
                me  = self._selection(population, costs)

                # Lai ghép → 2 con
                con1, con2 = self._crossover(cha, me)

                # Đột biến từng con
                con1 = self._mutate(con1)
                con2 = self._mutate(con2)

                new_pop.append(con1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(con2)

            # --------------------
            # Đánh giá fitness quần thể mới
            # --------------------
            population = new_pop
            costs = [self._path_cost(p) for p in population]

            # --------------------
            # Cập nhật best SAU khi có quần thể mới
            # --------------------
            best_idx = int(np.argmin(costs))
            if costs[best_idx] < self.best_cost:
                self.best_cost = costs[best_idx]
                self.best_route = population[best_idx][:]

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

        plt.savefig(f"{output_dir}/GA_final.png")
        plt.close()