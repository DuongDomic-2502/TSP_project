import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math


class SA:
    def __init__(self, cities,
                 initial_temp=0,
                 cooling_rate=0,
                 t_min=0,
                 steps_per_temp=0):       # L bước tại mỗi nhiệt độ

        self.cities = cities
        self.n = len(cities)

        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.t_min = t_min
        self.steps_per_temp = steps_per_temp   # L trong pseudocode

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
    # Tạo láng giềng bằng swap
    # Khớp pseudocode: "s_láng_giềng = Tạo_láng_giềng(s_hiện_tại)"
    # ----------------------------
    def _neighbor(self, path):
        a, b = random.sample(range(self.n), 2)
        new_path = path[:]
        new_path[a], new_path[b] = new_path[b], new_path[a]
        return new_path

    # ----------------------------
    # Chạy SA
    # Khớp pseudocode:
    #   s_hiện_tại = s₀
    #   T = T₀
    #   s_tốt_nhất = s₀
    #   Trong khi T > T_min:          ← vòng ngoài
    #       Lặp L lần:                ← vòng trong
    #           s_láng_giềng = Tạo_láng_giềng(s_hiện_tại)
    #           Δf = f(s_láng_giềng) - f(s_hiện_tại)
    #           Nếu Δf < 0: chấp nhận
    #           Ngược lại: chấp nhận với xác suất exp(-Δf / T)
    #           Cập nhật s_tốt_nhất
    #       T = α × T                 ← cooling SAU L bước
    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Khởi tạo lời giải ban đầu
        current = list(range(self.n))
        random.shuffle(current)
        current_cost = self._cost(current)

        self.best_route = current[:]
        self.best_cost = current_cost

        # Nhiệt độ ban đầu
        T = self.initial_temp

        # --------------------
        # Vòng ngoài: theo nhiệt độ
        # Khớp pseudocode: "Trong khi T > T_min"
        # --------------------
        while T > self.t_min:

            # --------------------
            # Vòng trong: L bước tại nhiệt độ T
            # Khớp pseudocode: "Lặp L lần"
            # --------------------
            for _ in range(self.steps_per_temp):

                # Tạo láng giềng
                candidate = self._neighbor(current)
                candidate_cost = self._cost(candidate)

                # Tính Δf
                delta = candidate_cost - current_cost

                # Chấp nhận lời giải
                # Khớp pseudocode:
                #   Nếu Δf < 0        → chấp nhận
                #   Ngược lại         → chấp nhận với exp(-Δf / T)
                if delta < 0:
                    current = candidate
                    current_cost = candidate_cost
                else:
                    if random.random() < math.exp(-delta / T):
                        current = candidate
                        current_cost = candidate_cost

                # Cập nhật best
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_route = current[:]

            # --------------------
            # Cooling SAU khi hoàn thành L bước
            # Khớp pseudocode: "T = α × T"
            # --------------------
            T *= self.cooling_rate

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

        plt.savefig(f"{output_dir}/SA_final.png")
        plt.close()