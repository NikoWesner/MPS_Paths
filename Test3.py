import numpy as np

def find_perm_sqrt_with_swaps(p1):
    n = len(p1)
    visited = [False] * n
    p2 = list(range(n))
    required_swaps = []
    
    # 1. Cycle Decomposition
    cycles = []
    for i in range(n):
        if not visited[i]:
            curr, cycle = i, []
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p1[curr]
            cycles.append(cycle)

    from collections import defaultdict
    length_map = defaultdict(list)
    for cycle in cycles:
        length_map[len(cycle)].append(cycle)

    # 2. Processing
    for length, group in length_map.items():
        if length % 2 != 0:
            # Odd cycles: Solvable
            step = (length + 1) // 2
            for cycle in group:
                for i in range(length):
                    p2[cycle[i]] = cycle[(i + step) % length]
        else:
            # Even cycles: Pair them
            while len(group) >= 2:
                c1, c2 = group.pop()
                c3 = group.pop() # c2/c3 doesn't matter, just need a pair
                for j in range(length):
                    p2[c1[j]] = c3[j]
                    p2[c3[j]] = c1[(j + 1) % length]
            
            # 3. The "Lonely" Even Cycle -> Decompose into Swaps
            if len(group) == 1:
                lonely = group.pop()
                # A cycle (a, b, c, d) is corrected by swaps: (a,b), (b,c), (c,d)
                for k in range(len(lonely) - 1):
                    required_swaps.append((lonely[k], lonely[k+1]))

    return p2, required_swaps

# --- Example ---
p1 = [1, 0, 3, 2, 4] # Two 2-cycles: (0,1) and (2,3). If we had only one, we'd need swaps.
# Let's force a failure by using a single swap and fixed points
# p1_fail = [1, 0, 2, 3, 4] 

p2, swaps = find_perm_sqrt_with_swaps(p1)

print(f"P2 Permutation: {p2}")
print(f"Simple Swaps to apply after: {swaps}")