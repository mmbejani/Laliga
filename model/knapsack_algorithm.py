import numpy as np


class Knapsack:

    def __init__(self, noises, computation_cost, max_computation_cost):
        self.n = noises
        self.cc = computation_cost
        self.max_cc = max_computation_cost

    def optimize(self):
        n_n = len(self.n)
        n_cc = self.max_cc

        K = [[0 for x in range(n_cc + 1)] for x in range(n_n + 1)]

        for i in range(n_n + 1):
            for j in range(n_cc + 1):
                if i == 0 or j == 0:
                    K[i][j] = 0
                elif self.cc[i - 1] <= j:
                    K[i][j] = max(self.n[i - 1] + K[i - 1][j - self.cc[i - 1]], K[i - 1][j])
                else:
                    K[i][j] = K[i - 1][j]

        res = K[n_n][n_cc]
        w = n_cc
        path = list()
        for i in range(n_n, 0, -1):
            if res <= 0:
                break
            if res == K[i - 1][w]:
                continue
            else:

                path.append(i - 1)

                res = res - self.n[i - 1]
                w = w - self.cc[i - 1]

        return sorted(path)
