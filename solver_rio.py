#!/usr/bin/env python3

import sys
import math
import itertools
import random
from common import print_solution, read_input


def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# START OF MY CODE
INF = 10 ** 100


def held_karp(dist, need_back=True):
    cities_len = len(dist)
    dp = [[-1] * cities_len for _ in range(2 ** cities_len)]
    route = [[-1] * cities_len for _ in range(2 ** cities_len)]

    def rec(s: int, v: int):
        if dp[s][v] >= 0:
            return dp[s][v]
        if s == (1 << cities_len) - 1 and (v == 0 or not need_back):
            dp[s][v] = 0
            return 0
        res = INF
        for u in range(cities_len):
            if not (s >> u & 1):
                old_val = res
                res = min(res, rec(s | 1 << u, u) + dist[v][u])
                if res < old_val:
                    route[s | 1 << u][u] = v
        dp[s][v] = res
        return res

    print(rec(0, 0))


def rio_solver(cities, dist):
    x_max = max(x for x, y in cities) + 1
    y_max = max(y for x, y in cities) + 1

    # 10 x 10 　に分割
    group_size = 2  # 偶数 or 1
    assert group_size % 2 == 0 or group_size == 1
    group = [[[] for _ in range(group_size)] for _ in range(group_size)]
    for i, (x, y) in enumerate(cities):
        x_group = int(x // (x_max / group_size))
        y_group = int(y // (y_max / group_size))
        # print(x_group,y_group,x,y)
        group[x_group][y_group].append((i, (x, y)))
    solution = []

    def greedy_group(i, j):
        g = group[i][j]
        if not g:
            return
        if not solution:
            solution.append(g[0][0])
        current_city = solution[-1]
        unvisited_cities = set(index for index, _ in g) - {current_city}
        while unvisited_cities:
            current_city = min(unvisited_cities, key=lambda c: dist[current_city][c])
            unvisited_cities.remove(current_city)
            solution.append(current_city)

    def optimize():
        MAX_LEN = min(len(cities) + 1, 9)
        start = -random.randint(1, len(solution))  # use minus index to avoid index out of range
        index_list = [v for v in range(start, start + MAX_LEN)]
        city_index = [solution[i] for i in index_list]
        cost = INF
        _optimal = None
        for _tmp_city_index in itertools.permutations(city_index[1:-1]):
            tmp_city_index = (city_index[0],) + _tmp_city_index + (city_index[-1],)
            current_cost = sum(dist[tmp_city_index[i]][tmp_city_index[i + 1]] for i in range(len(tmp_city_index) - 1))
            if current_cost < cost:
                cost = current_cost
                _optimal = tmp_city_index
        for index, i in enumerate(index_list):
            assert index != 0 or solution[i] == _optimal[index]
            solution[i] = _optimal[index]

    iterate_list = [(0, 0)]
    for i in range(group_size):
        for j in range(1, group_size):
            if i % 2:
                j = group_size - j
            iterate_list.append((i, j))
    for i in reversed(range(1, group_size)):
        iterate_list.append((i, 0))

    for i, j in iterate_list:
        greedy_group(i, j)

    for _ in range(1000):
        optimize()
    return solution


# END OF MY CODE
def solve(cities):
    N = len(cities)

    dist = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])

    current_city = 0
    unvisited_cities = set(range(1, N))

    return rio_solver(cities, dist)
    # while unvisited_cities:
    #    next_city = min(unvisited_cities, key=distance_from_current_city)
    #    unvisited_cities.remove(next_city)
    #    solution.append(next_city)
    #    current_city = next_city


if __name__ == '__main__':
    assert len(sys.argv) > 1
    solution = solve(read_input(sys.argv[1]))
    print_solution(solution)
