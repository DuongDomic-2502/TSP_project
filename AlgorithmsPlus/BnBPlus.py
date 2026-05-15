import os
import numpy as np
import matplotlib.pyplot as plt


class BnBPlus:
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
    # Greedy Nearest-Neighbor – tạo nghiệm khởi tạo O(n²)
    # Dùng làm warm-start để best_cost chặt trước khi BnB chạy
    # ----------------------------
    def _greedy_initial(self, start=0):
        visited = [False] * self.n
        route = [start]
        visited[start] = True
        cost = 0.0

        for _ in range(self.n - 1):
            current = route[-1]
            best_dist = float('inf')
            best_next = -1

            for j in range(self.n):
                if not visited[j] and self.distance_matrix[current][j] < best_dist:
                    best_dist = self.distance_matrix[current][j]
                    best_next = j

            route.append(best_next)
            visited[best_next] = True
            cost += best_dist

        cost += self.distance_matrix[route[-1]][start]
        return route, cost

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
    # Lower bound = cost_đã_đi + MST(unvisited) + min_in + min_out
    # ----------------------------
    def _lower_bound(self, current, visited_set, start):
        unvisited = [i for i in range(self.n) if i not in visited_set]

        if not unvisited:
            return self.distance_matrix[current][start]

        lb = 0
        lb += self._mst_cost(unvisited)
        lb += min(self.distance_matrix[current][j] for j in unvisited)
        lb += min(self.distance_matrix[j][start] for j in unvisited)

        return lb

    # ----------------------------
    # Branch and Bound (DFS stack) – truyền thống, không thay đổi
    # Hiệu quả nhờ best_cost được khởi tạo chặt từ greedy + 3-opt
    # ----------------------------
    def _bnb(self, start):
        stack = [(start, [start], {start}, 0)]

        while stack:
            current, path, visited_set, current_cost = stack.pop()

            lb = current_cost + self._lower_bound(current, visited_set, start)

            if lb >= self.best_cost:
                continue

            if len(visited_set) == self.n:
                total_cost = current_cost + self.distance_matrix[current][start]

                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_route = path[:]

                continue

            for nxt in range(self.n):
                if nxt not in visited_set:
                    new_cost = current_cost + self.distance_matrix[current][nxt]

                    child_lb = new_cost + self._lower_bound(nxt, visited_set | {nxt}, start)

                    if child_lb < self.best_cost:
                        stack.append((
                            nxt,
                            path + [nxt],
                            visited_set | {nxt},
                            new_cost
                        ))

    # ----------------------------
    # 3-opt helpers
    # ----------------------------
    def _route_cost(self, route):
        n = len(route)
        return sum(
            self.distance_matrix[route[i]][route[(i + 1) % n]]
            for i in range(n)
        )

    def _three_opt_move(self, route, i, j, k):
        n = len(route)
        a, b = route[i], route[(i + 1) % n]
        c, d = route[j], route[(j + 1) % n]
        e, f = route[k], route[(k + 1) % n]

        d0 = (self.distance_matrix[a][b]
              + self.distance_matrix[c][d]
              + self.distance_matrix[e][f])

        seg_A = route[:i + 1]
        seg_B = route[i + 1:j + 1]
        seg_C = route[j + 1:k + 1]
        seg_D = route[k + 1:]

        candidates = [
            # 1. Đảo B
            (seg_A + seg_B[::-1] + seg_C + seg_D,
             self.distance_matrix[a][seg_B[-1]] + self.distance_matrix[seg_B[0]][d] + self.distance_matrix[e][f]),
            # 2. Đảo C
            (seg_A + seg_B + seg_C[::-1] + seg_D,
             self.distance_matrix[a][b] + self.distance_matrix[c][seg_C[-1]] + self.distance_matrix[seg_C[0]][f]),
            # 3. Đảo B và C
            (seg_A + seg_B[::-1] + seg_C[::-1] + seg_D,
             self.distance_matrix[a][seg_B[-1]] + self.distance_matrix[seg_B[0]][seg_C[-1]] + self.distance_matrix[seg_C[0]][f]),
            # 4. Hoán vị B↔C
            (seg_A + seg_C + seg_B + seg_D,
             self.distance_matrix[a][d] + self.distance_matrix[e][b] + self.distance_matrix[c][f]),
            # 5. Hoán vị B↔C, đảo B
            (seg_A + seg_C + seg_B[::-1] + seg_D,
             self.distance_matrix[a][d] + self.distance_matrix[e][seg_B[-1]] + self.distance_matrix[seg_B[0]][f]),
            # 6. Hoán vị B↔C, đảo C
            (seg_A + seg_C[::-1] + seg_B + seg_D,
             self.distance_matrix[a][seg_C[-1]] + self.distance_matrix[seg_C[0]][b] + self.distance_matrix[c][f]),
            # 7. Hoán vị B↔C, đảo cả hai
            (seg_A + seg_C[::-1] + seg_B[::-1] + seg_D,
             self.distance_matrix[a][seg_C[-1]] + self.distance_matrix[seg_C[0]][seg_B[-1]] + self.distance_matrix[seg_B[0]][f]),
        ]

        best_delta = 0.0
        best_route = None

        for new_route, d_new in candidates:
            delta = d_new - d0
            if delta < best_delta:
                best_delta = delta
                best_route = new_route

        return best_delta, best_route

    def _three_opt(self, route):
        improved = True
        best_route = route[:]
        n = len(best_route)

        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        delta, new_route = self._three_opt_move(best_route, i, j, k)
                        if new_route is not None and delta < -1e-10:
                            best_route = new_route
                            n = len(best_route)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return best_route

    # ----------------------------
    # Run
    # ----------------------------
    def run(self, output_dir=None):

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # ---- Bước 1: Greedy → nghiệm thô O(n²) ----
        greedy_route, greedy_cost = self._greedy_initial(start=0)

        # ---- Bước 2: 3-opt cải thiện nghiệm greedy O(n³) ----
        #   → best_cost chặt, gần tối ưu TRƯỚC khi BnB chạy
        #   → BnB cắt tỉa phần lớn nhánh ngay từ node đầu tiên
        warm_route = self._three_opt(greedy_route)
        warm_cost  = self._route_cost(warm_route)

        self.best_route = warm_route
        self.best_cost  = warm_cost

        # ---- Bước 3: BnB tối ưu chính xác ----
        #   nhờ best_cost chặt → không gian tìm kiếm thu hẹp đáng kể
        self._bnb(start=0)

        # ---- Bước 4: Lưu ảnh ----
        if output_dir and self.best_route:
            plt.figure(figsize=(7, 5))

            route = [self.cities[i] for i in self.best_route]

            x = [c.x for c in route] + [route[0].x]
            y = [c.y for c in route] + [route[0].y]

            plt.plot(x, y, 'o-')
            plt.title(f"Best TSP route (BnBPlus) | cost = {self.best_cost:.4f}")

            plt.savefig(f"{output_dir}/BnBPlusFinal.png")
            plt.close()

        return (
            [self.cities[i] for i in self.best_route],
            self.best_cost
        )