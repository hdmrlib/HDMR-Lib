from itertools import combinations

dims = [0, 1, 4]

for i in range(1, len(dims)):
                    for g_combination in combinations(dims, i):
                        print(g_combination)