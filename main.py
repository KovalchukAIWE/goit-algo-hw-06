import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Створення графа
G = nx.Graph()

# Додавання станцій метро (вузлів)
stations = [
    'Central', 'North', 'South', 'East', 'West', 
    'North-East', 'South-East', 'South-West', 'North-West', 'Airport'
]

G.add_nodes_from(stations)

# Додавання ліній метро (ребер) з вагами
connections = [
    ('Central', 'North', 5), ('Central', 'South', 7), ('Central', 'East', 3), ('Central', 'West', 4),
    ('North', 'North-East', 4), ('East', 'North-East', 6),
    ('South', 'South-East', 5), ('East', 'South-East', 8),
    ('South', 'South-West', 6), ('West', 'South-West', 7),
    ('North', 'North-West', 5), ('West', 'North-West', 6),
    ('Airport', 'North-East', 10), ('Airport', 'South-East', 9)
]

G.add_weighted_edges_from(connections)

# Візуалізація графа
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)  # Автоматичне розташування вузлів
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=12, font_weight='bold', edge_color='gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Транспортна мережа міста з вагами')
plt.show()

# Реалізація алгоритмів DFS та BFS

def dfs(graph, start, goal, path=None, visited=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    visited.add(start)
    path.append(start)
    if start == goal:
        return path
    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            result = dfs(graph, neighbor, goal, path.copy(), visited.copy())
            if result:
                return result
    return None


def bfs(graph, start, goal):
    visited = set()
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None

# Виконання алгоритмів
start_station = 'Central'
end_station = 'Airport'

# DFS
dfs_path = dfs(G, start_station, end_station)

# BFS
bfs_path = bfs(G, start_station, end_station)

print(f"\nDFS шлях від {start_station} до {end_station}: {dfs_path}")
print(f"BFS найкоротший шлях від {start_station} до {end_station}: {bfs_path}")

# Порівняння результатів
print("\nПорівняння алгоритмів:")
if dfs_path != bfs_path:
    print("DFS і BFS дали різні результати через різну стратегію обходу. DFS заглиблюється в одну гілку до кінця перед переходом до наступної, тоді як BFS досліджує всі сусідні вузли на одному рівні перед переходом на наступний.")
else:
    print("DFS і BFS знайшли однаковий шлях.")

# Реалізація алгоритму Дейкстри

def dijkstra(graph, start):
    visited = set()
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in graph.nodes}
    
    while visited != set(graph.nodes):
        min_node = None
        for node in graph.nodes:
            if node not in visited:
                if min_node is None or distances[node] < distances[min_node]:
                    min_node = node
        
        if min_node is None:
            break

        visited.add(min_node)
        for neighbor in graph.neighbors(min_node):
            weight = graph[min_node][neighbor]['weight']
            if distances[min_node] + weight < distances[neighbor]:
                distances[neighbor] = distances[min_node] + weight
                previous_nodes[neighbor] = min_node

    return distances, previous_nodes


def reconstruct_path(previous_nodes, start, end):
    path = []
    current = end
    while current != start:
        path.insert(0, current)
        current = previous_nodes[current]
        if current is None:
            return None
    path.insert(0, start)
    return path

# Виконання алгоритму Дейкстри
for station in stations:
    distances, previous_nodes = dijkstra(G, station)
    print(f"\nНайкоротші шляхи від {station}:")
    for target in stations:
        if station != target:
            path = reconstruct_path(previous_nodes, station, target)
            print(f"{station} -> {target}: Шлях: {path}, Довжина: {distances[target]}")
