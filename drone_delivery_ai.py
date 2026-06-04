"""
DRONE DELIVERY AI PROJECT
CO1 -> State Representation, PEAS Formulation
CO2 -> Informed & Uninformed Searches (BFS, DFS, UCS, Greedy, A*)
CO3 -> Constraint Satisfaction Problem (Backtracking + MRV + LCV)
CO4 -> Adversarial Game Search (Minimax + Alpha-Beta Pruning)
CO5 -> Probabilistic Inference (Bayesian Risk Assessment + Markov Weather Dynamics)
CO6 -> Integrated Pipeline + Explainable Reasoning Logs

Run:
    python drone_delivery_ai_COwise.py
"""

import heapq
from collections import deque
from dataclasses import dataclass
import time

# ==========================================================
# CO1 : STATE REPRESENTATION & PEAS FORMULATION
# ==========================================================
"""
PEAS Formulation:
- Performance Measure: Minimum path cost, valid battery utilization, optimal scheduling, safety.
- Environment: 2D Grid with static obstacles, dynamic weather transitions, package dispatch windows.
- Actuators: Rotors/Motors (Move North, South, East, West), Schedule Allocator, Risk Abatement system.
- Sensors: GPS/Odometer (Position Coordinates), Battery Gauge (%), Weather Forecast Engine.
"""

@dataclass
class DroneState:
    position: tuple
    battery: int
    delivered: bool = False


class Logger:
    """CO6 Explainable Reasoning Trace Engine"""
    def __init__(self):
        self.logs = []

    def add(self, stage, msg):
        self.logs.append(f"[{stage}] {msg}")

    def show(self):
        print("\n==================== CO6: EXPLAINABLE REASONING TRACE ====================")
        for step in self.logs:
            print(step)
        print("==========================================================================")


logger = Logger()

# ==========================================================
# USER INPUT ENVIRONMENT SETUP
# ==========================================================

def create_environment():
    print("--- Initialize Drone Environment ---")
    rows = int(input("Enter rows: "))
    cols = int(input("Enter cols: "))

    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    print("\nStart Position")
    sx = int(input(f"Row (0-{rows-1}): "))
    sy = int(input(f"Col (0-{cols-1}): "))

    print("\nGoal Position")
    gx = int(input(f"Row (0-{rows-1}): "))
    gy = int(input(f"Col (0-{cols-1}): "))

    obs = int(input("\nNumber of obstacles: "))

    for i in range(obs):
        print(f"Obstacle {i+1}")
        r = int(input("  Row: "))
        c = int(input("  Col: "))
        if 0 <= r < rows and 0 <= c < cols:
            if (r, c) != (sx, sy) and (r, c) != (gx, gy):
                grid[r][c] = 1

    battery = int(input("\nBattery status (%): "))

    return grid, (sx, sy), (gx, gy), battery


# ==========================================================
# COMMON CORE UTILITIES
# ==========================================================

def neighbors(grid, node):
    rows = len(grid)
    cols = len(grid[0])
    x, y = node
    result = []

    # Movement Actions: North, South, West, East
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            result.append((nx, ny))
    return result


# ==========================================================
# CO2 : UNINFORMED & INFORMED SEARCH STRATEGIES
# ==========================================================

def bfs(grid, start, goal):
    if start == goal:
        return [start]
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        for nbr in neighbors(grid, node):
            if nbr == goal:
                return path + [nbr]
            if nbr not in visited:
                visited.add(nbr)
                queue.append(path + [nbr])
    return None


def dfs(grid, start, goal):
    stack = [[start]]
    visited = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for nbr in neighbors(grid, node):
                if nbr not in visited:
                    stack.append(path + [nbr])
    return None


def ucs(grid, start, goal):
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == goal:
            return path, cost

        if node not in visited:
            visited.add(node)
            for nbr in neighbors(grid, node):
                if nbr not in visited:
                    heapq.heappush(pq, (cost + 1, nbr, path + [nbr]))
    return None, float('inf')


def heuristic(node, goal):
    # Manhattan distance calculation
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def greedy(grid, start, goal):
    pq = [(heuristic(start, goal), start, [start])]
    visited = {start}

    while pq:
        _, node, path = heapq.heappop(pq)

        if node == goal:
            return path

        for nbr in neighbors(grid, node):
            if nbr not in visited:
                visited.add(nbr)
                heapq.heappush(pq, (heuristic(nbr, goal), nbr, path + [nbr]))
    return None


def astar(grid, start, goal):
    pq = [(heuristic(start, goal), start, [start], 0)]
    visited = {}

    while pq:
        _, node, path, g = heapq.heappop(pq)

        if node == goal:
            return path, g

        if node in visited and visited[node] <= g:
            continue

        visited[node] = g

        for nbr in neighbors(grid, node):
            new_g = g + 1
            if nbr not in visited or new_g < visited.get(nbr, float('inf')):
                f_cost = new_g + heuristic(nbr, goal)
                heapq.heappush(pq, (f_cost, nbr, path + [nbr], new_g))
                
    return None, float('inf')


# ==========================================================
# CO3 : CONSTRAINT SATISFACTION PROBLEM (DELIVERY SLOTS)
# ==========================================================

domains = {
    "Drone_1": ["9AM", "10AM"],
    "Drone_2": ["10AM", "11AM"],
    "Drone_3": ["9AM", "11AM"]
}

def consistent(assign):
    values = list(assign.values())
    return len(values) == len(set(values))

def mrv(assign):
    unassigned = [v for v in domains if v not in assign]
    return min(unassigned, key=lambda x: len(domains[x]))

def lcv(var, assign):
    # Base configuration: returns domain values directly
    return domains[var]

def backtrack(assign):
    if len(assign) == len(domains):
        return assign

    var = mrv(assign)

    for value in lcv(var, assign):
        assign[var] = value
        if consistent(assign):
            result = backtrack(assign)
            if result:
                return result
        del assign[var]
    return None


# ==========================================================
# CO4 : ADVERSARIAL MULTI-AGENT GAME SEARCH
# ==========================================================

game_tree = {
    "A": ["B", "C"],
    "B": [3, 5],
    "C": [2, 9]
}

def minimax(node, maximizing):
    if isinstance(node, int):
        return node

    if maximizing:
        value = float("-inf")
        for child in game_tree[node]:
            value = max(value, minimax(child, False))
        return value
    else:
        value = float("inf")
        for child in game_tree[node]:
            value = min(value, minimax(child, True))
        return value


def alphabeta(node, alpha, beta, maximizing):
    if isinstance(node, int):
        return node

    if maximizing:
        value = float("-inf")
        for child in game_tree[node]:
            value = max(value, alphabeta(child, alpha, beta, False))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = float("inf")
        for child in game_tree[node]:
            value = min(value, alphabeta(child, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


# ==========================================================
# CO5 : PROBABILISTIC INFERENCE AND UNCERTAINTY
# ==========================================================

def predict_weather(current):
    current = current.capitalize()
    transition = {
        "Sunny": {"Sunny": 0.7, "Cloudy": 0.3, "Rainy": 0.0},
        "Cloudy": {"Sunny": 0.6, "Cloudy": 0.0, "Rainy": 0.4},
        "Rainy": {"Sunny": 0.0, "Cloudy": 0.5, "Rainy": 0.5}
    }
    if current not in transition:
        return "Sunny" # Default fallback safety state
    return max(transition[current], key=transition[current].get)


def bayes_success(weather):
    probs = {
        "Sunny": 0.95,
        "Cloudy": 0.80,
        "Rainy": 0.50
    }
    return probs.get(weather, 0.95)


# ==========================================================
# CO6 : INTEGRATED PIPELINE & EXPLAINABLE VISUALIZATION
# ==========================================================

def draw_path(grid, path):
    board = [["X" if c == 1 else "." for c in row] for row in grid]
    for x, y in path:
        board[x][y] = "*"
    
    # Mark Explicit Start/End markers
    sx, sy = path[0]
    gx, gy = path[-1]
    board[sx][sy] = "S"
    board[gx][gy] = "G"

    print("\nRENDERED NAVIGATION MAP (S=Start, G=Goal, *=Path, X=Obstacle):")
    for row in board:
        print(" ".join(row))


def main():
    # --- CO1 Stage ---
    grid, start, goal, battery = create_environment()
    drone = DroneState(start, battery)
    logger.add("CO1", f"Drone initialized at {start} with {battery}% Battery capacity.")

    # --- CO2 Stage ---
    print("\n========== CO2: SEARCH METRICS & BENCHMARKING ==========")
    
    start_time = time.time()
    bfs_path = bfs(grid, start, goal)
    bfs_time = time.time() - start_time

    start_time = time.time()
    dfs_path = dfs(grid, start, goal)
    dfs_time = time.time() - start_time

    ucs_path, ucs_cost = ucs(grid, start, goal)
    greedy_path = greedy(grid, start, goal)
    astar_path, astar_cost = astar(grid, start, goal)

    print(f"BFS Path Length   : {len(bfs_path) if bfs_path else 'UNREACHABLE'}")
    print(f"DFS Path Length   : {len(dfs_path) if dfs_path else 'UNREACHABLE'}")
    print(f"UCS Step Cost     : {ucs_cost if ucs_path else 'UNREACHABLE'}")
    print(f"Greedy Path Length: {len(greedy_path) if greedy_path else 'UNREACHABLE'}")
    print(f"A* Path Cost      : {astar_cost if astar_path else 'UNREACHABLE'}")
    print(f"BFS Computation   : {bfs_time:.6f} sec")
    print(f"DFS Computation   : {dfs_time:.6f} sec")
    
    logger.add("CO2", "Grid spatial discovery metrics analyzed via informed/uninformed searches.")

    # --- CO3 Stage ---
    print("\n========== CO3: CONSTRAINT SATISFACTION SCHEDULING ==========")
    schedule = backtrack({})
    print(f"De-conflicted Drone Timetable: {schedule}")
    logger.add("CO3", "Resolved timetable resource contentions using Backtracking + MRV.")

    # --- CO4 Stage ---
    print("\n========== CO4: ADVERSARIAL RISK EVALUATION ==========")
    minimax_val = minimax("A", True)
    alphabeta_val = alphabeta("A", float("-inf"), float("inf"), True)
    print(f"Standard Minimax Value       : {minimax_val}")
    print(f"Alpha-Beta Pruned Edge Value: {alphabeta_val}")
    logger.add("CO4", f"Evaluated competitive airspace hazards. Game Value: {alphabeta_val}")

    # --- CO5 Stage ---
    print("\n========== CO5: PROBABILISTIC UNCERTAINTY HANDLING ==========")
    raw_weather = input("Input Current Weather Pattern (Sunny/Cloudy/Rainy): ")
    predicted = predict_weather(raw_weather)
    success_rate = bayes_success(predicted)
    print(f"Markov Step State Prediction : {predicted}")
    print(f"Bayesian Flight Success Prob : {success_rate * 100:.1f}%")
    logger.add("CO5", f"Transitioned weather to {predicted}. Survival probability evaluated at {success_rate*100}%.")

    # --- CO6 Integrated Final Deployment Execution ---
    print("\n========== CO6: MISSION INTEGRATION PIPELINE ==========")
    
    if not astar_path:
        print("CRITICAL DAMAGE: Flight Route planning failed. Target Destination Unreachable!")
        logger.add("CO6", "Mission aborted. No path through grid configuration.")
        logger.show()
        return

    # Energy model assertion constraint: 2% battery drain per step unit
    required_power = len(astar_path) * 2
    if battery < required_power:
        print(f"CRITICAL SAFETY WARNING: Insufficient energy metrics. Requires {required_power}%, Has {battery}%")
        logger.add("CO6", f"Mission rejected dynamically due to battery violation (Requires {required_power}%).")
        logger.show()
        return

    # Execute path visualization and change operational state profile
    draw_path(grid, astar_path)
    drone.position = goal
    drone.delivered = True
    
    logger.add("CO6", f"Optimal A* path committed safely. Target delivery reached at location {goal}.")
    logger.show()


if __name__ == "__main__":
    main()