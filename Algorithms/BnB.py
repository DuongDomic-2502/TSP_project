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
    def _lower_bound(self, current, visited_set, start):
        unvisited = [i for i in range(self.n) if i not in visited_set]

        if not unvisited:
            return self.distance_matrix[current][start]

        lb = 0

        # MST trên các node chưa thăm
        lb += self._mst_cost(unvisited)

        # Cạnh ngắn nhất từ current tới unvisited
        lb += min(self.distance_matrix[current][j] for j in unvisited)

        # Cạnh ngắn nhất từ unvisited về start
        lb += min(self.distance_matrix[j][start] for j in unvisited)

        return lb

    # ----------------------------
    # Branch and Bound (hàng đợi tường minh - DFS stack)
    # Khớp với pseudocode:
    #   Hàng_đợi = [nút_gốc]
    #   while Hàng_đợi không rỗng:
    #       nút = Chọn_nút(...)
    #       lb  = Tính_cận_dưới(nút)
    #       if lb >= z_tốt_nhất: Cắt tỉa
    #       if lời giải khả thi: cập nhật best
    #       else: Phân_nhánh → push con vào hàng đợi
    # ----------------------------
    def _bnb(self, start):
        # Mỗi phần tử trong stack là một "nút":
        # (current, path, visited_set, current_cost)
        stack = [(start, [start], {start}, 0)]

        while stack:
            # Chọn nút (DFS → pop từ cuối)
            current, path, visited_set, current_cost = stack.pop()

            # Tính cận dưới của nút hiện tại
            lb = current_cost + self._lower_bound(current, visited_set, start)

            # Cắt tỉa
            if lb >= self.best_cost:
                continue

            # Lời giải khả thi (đã thăm hết)
            if len(visited_set) == self.n:
                total_cost = current_cost + self.distance_matrix[current][start]

                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_route = path[:]

                continue

            # Phân nhánh → tạo các nút con, push vào stack
            for nxt in range(self.n):
                if nxt not in visited_set:          # O(1) với set
                    new_cost = current_cost + self.distance_matrix[current][nxt]

                    # Chỉ push nếu còn hi vọng (tỉa sớm)
                    child_lb = new_cost + self._lower_bound(nxt, visited_set | {nxt}, start)

                    if child_lb < self.best_cost:
                        stack.append((
                            nxt,
                            path + [nxt],
                            visited_set | {nxt},
                            new_cost
                        ))

    # ----------------------------
    # Run
    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.best_cost = float('inf')
        self.best_route = None

        self._bnb(start=0)

        # Save final route
        if output_dir and self.best_route:
            plt.figure(figsize=(7, 5))

            route = [self.cities[i] for i in self.best_route]

            x = [c.x for c in route] + [route[0].x]
            y = [c.y for c in route] + [route[0].y]

            plt.plot(x, y, 'o-')
            plt.title(f"Best BnB TSP route | cost = {self.best_cost:.4f}")

            plt.savefig(f"{output_dir}/BnBFinal.png")
            plt.close()

        return (
            [self.cities[i] for i in self.best_route],
            self.best_cost
        )