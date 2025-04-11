neighborhoods = {
    "Valletta": {
        "Sliema": 2,
        "St. Julian's": 3,
        "Qormi": 5,
        "Żabbar": 9,
        "Marsaskala": 9,
        "Marsaxlokk": 15,
        "Mdina": 13,
        "Rabat": 13,
        "Birgu": 1,
        "Senglea": 1.3,
        "Cospicua": 2.5,
        "Birkirkara": 7,
        "Mosta": 10
    },
    "Birkirkara": {
        "Valletta": 7,
        "Sliema": 4,
        "St. Julian's": 4,
        "Mosta": 5,
        "Qormi": 3,
        "Żebbuġ": 6
    },
    "Sliema": {
        "Valletta": 2,
        "Birkirkara": 4,
        "St. Julian's": 3
    },
    "St. Julian's": {
        "Valletta": 3,
        "Sliema": 3,
        "Birkirkara": 4,
        "Mosta": 6
    },
    "Mosta": {
        "Birkirkara": 5,
        "St. Julian's": 6,
        "Qormi": 5,
        "Żebbuġ": 5,
        "Mdina": 4,
        "Rabat": 5,
        "Valletta": 10
    },
    "Qormi": {
        "Valletta": 5,
        "Birkirkara": 3,
        "Mosta": 5,
        "Żebbuġ": 4,
        "Mdina": 6,
        "Rabat": 7
    },
    "Żabbar": {
        "Valletta": 9,
        "Marsaskala": 4,
        "Birgu": 5,
        "Senglea": 5,
        "Cospicua": 5
    },
    "Żebbuġ": {
        "Birkirkara": 6,
        "Mosta": 5,
        "Qormi": 4,
        "Mdina": 3,
        "Rabat": 4
    },
    "Marsaskala": {
        "Valletta": 9,
        "Żabbar": 4,
        "Marsaxlokk": 7
    },
    "Marsaxlokk": {
        "Valletta": 15,
        "Marsaskala": 7
    },
    "Mdina": {
        "Valletta": 13,
        "Mosta": 4,
        "Qormi": 6,
        "Żebbuġ": 3,
        "Rabat": 2
    },
    "Rabat": {
        "Valletta": 13,
        "Mosta": 5,
        "Qormi": 7,
        "Żebbuġ": 4,
        "Mdina": 2
    },
    "Birgu": {
        "Valletta": 1,
        "Żabbar": 5,
        "Senglea": 1,
        "Cospicua": 1
    },
    "Senglea": {
        "Valletta": 1.3,
        "Żabbar": 5,
        "Birgu": 1,
        "Cospicua": 1
    },
    "Cospicua": {
        "Valletta": 2.5,
        "Żabbar": 5,
        "Birgu": 1,
        "Senglea": 1
    }
}

neighborhoods_heu = {
'Valletta': 
    {
        'Birkirkara': 4.83,
        'Sliema': 1.84,
        "St. Julian's": 3.05,
        'Mosta': 8.09,
        'Qormi': 4.63,
        'Żabbar': 3.18,
        'Żebbuġ': 7.23,
        'Marsaskala': 5.95,
        'Marsaxlokk': 7.0,
        'Mdina': 10.16,
        'Rabat': 10.65,
        'Birgu': 1.29,
        'Senglea': 1.39,
        'Cospicua': 2.57
    },
 'Birkirkara': {'Valletta': 4.83,
  'Sliema': 3.98,
  "St. Julian's": 3.53,
  'Mosta': 3.47,
  'Qormi': 2.55,
  'Żabbar': 7.05,
  'Żebbuġ': 3.18,
  'Marsaskala': 10.12,
  'Marsaxlokk': 9.75,
  'Mdina': 5.38,
  'Rabat': 5.9,
  'Birgu': 5.29,
  'Senglea': 5.25,
  'Cospicua': 5.73},
 'Sliema': {'Valletta': 1.84,
  'Birkirkara': 3.98,
  "St. Julian's": 1.23,
  'Mosta': 6.83,
  'Qormi': 4.79,
  'Żabbar': 5.01,
  'Żebbuġ': 6.91,
  'Marsaskala': 7.76,
  'Marsaxlokk': 8.77,
  'Mdina': 9.31,
  'Rabat': 9.86,
  'Birgu': 3.07,
  'Senglea': 3.14,
  'Cospicua': 4.26},
 "St. Julian's": {'Valletta': 3.05,
  'Birkirkara': 3.53,
  'Sliema': 1.23,
  'Mosta': 5.9,
  'Qormi': 4.99,
  'Żabbar': 6.2,
  'Żebbuġ': 6.67,
  'Marsaskala': 8.99,
  'Marsaxlokk': 9.88,
  'Mdina': 8.64,
  'Rabat': 9.21,
  'Birgu': 4.23,
  'Senglea': 4.29,
  'Cospicua': 5.35},
 'Mosta': {'Valletta': 8.09,
  'Birkirkara': 3.47,
  'Sliema': 6.83,
  "St. Julian's": 5.9,
  'Qormi': 5.59,
  'Żabbar': 10.52,
  'Żebbuġ': 4.2,
  'Marsaskala': 13.59,
  'Marsaxlokk': 13.12,
  'Mdina': 3.29,
  'Rabat': 3.93,
  'Birgu': 8.72,
  'Senglea': 8.68,
  'Cospicua': 9.21},
 'Qormi': {'Valletta': 4.63,
  'Birkirkara': 2.55,
  'Sliema': 4.79,
  "St. Julian's": 4.99,
  'Mosta': 5.59,
  'Żabbar': 5.66,
  'Żebbuġ': 2.82,
  'Marsaskala': 8.62,
  'Marsaxlokk': 7.59,
  'Mdina': 6.32,
  'Rabat': 6.66,
  'Birgu': 4.43,
  'Senglea': 4.33,
  'Cospicua': 4.27},
 'Żabbar': {'Valletta': 3.18,
  'Birkirkara': 7.05,
  'Sliema': 5.01,
  "St. Julian's": 6.2,
  'Mosta': 10.52,
  'Qormi': 5.66,
  'Żebbuġ': 8.47,
  'Marsaskala': 3.07,
  'Marsaxlokk': 3.96,
  'Mdina': 11.93,
  'Rabat': 12.3,
  'Birgu': 1.99,
  'Senglea': 1.96,
  'Cospicua': 1.39},
 'Żebbuġ': {'Valletta': 7.23,
  'Birkirkara': 3.18,
  'Sliema': 6.91,
  "St. Julian's": 6.67,
  'Mosta': 4.2,
  'Qormi': 2.82,
  'Żabbar': 8.47,
  'Marsaskala': 11.38,
  'Marsaxlokk': 10.0,
  'Mdina': 3.68,
  'Rabat': 3.93,
  'Birgu': 7.21,
  'Senglea': 7.11,
  'Cospicua': 7.08},
 'Marsaskala': {'Valletta': 5.95,
  'Birkirkara': 10.12,
  'Sliema': 7.76,
  "St. Julian's": 8.99,
  'Mosta': 13.59,
  'Qormi': 8.62,
  'Żabbar': 3.07,
  'Żebbuġ': 11.38,
  'Marsaxlokk': 3.48,
  'Mdina': 14.93,
  'Rabat': 15.27,
  'Birgu': 4.95,
  'Senglea': 4.96,
  'Cospicua': 4.42},
 'Marsaxlokk': {'Valletta': 7.0,
  'Birkirkara': 9.75,
  'Sliema': 8.77,
  "St. Julian's": 9.88,
  'Mosta': 13.12,
  'Qormi': 7.59,
  'Żabbar': 3.96,
  'Żebbuġ': 10.0,
  'Marsaskala': 3.48,
  'Mdina': 13.69,
  'Rabat': 13.9,
  'Birgu': 5.72,
  'Senglea': 5.64,
  'Cospicua': 4.53},
 'Mdina': {'Valletta': 10.16,
  'Birkirkara': 5.38,
  'Sliema': 9.31,
  "St. Julian's": 8.64,
  'Mosta': 3.29,
  'Qormi': 6.32,
  'Żabbar': 11.93,
  'Żebbuġ': 3.68,
  'Marsaskala': 14.93,
  'Marsaxlokk': 13.69,
  'Rabat': 0.64,
  'Birgu': 10.44,
  'Senglea': 10.36,
  'Cospicua': 10.54},
 'Rabat': 
    {
        'Valletta': 10.65,
        'Birkirkara': 5.9,
        'Sliema': 9.86,
        "St. Julian's": 9.21,
        'Mosta': 3.93,
        'Qormi': 6.66,
        'Żabbar': 12.3,
        'Żebbuġ': 3.93,
        'Marsaskala': 15.27,
        'Marsaxlokk': 13.9,
        'Mdina': 0.64,
        'Birgu': 10.87,
        'Senglea': 10.79,
        'Cospicua': 10.91
    },
 'Birgu': 
    {
        'Valletta': 1.29,
        'Birkirkara': 5.29,
        'Sliema': 3.07,
        "St. Julian's": 4.23,
        'Mosta': 8.72,
        'Qormi': 4.43,
        'Żabbar': 1.99,
        'Żebbuġ': 7.21,
        'Marsaskala': 4.95,
        'Marsaxlokk': 5.72,
        'Mdina': 10.44,
        'Rabat': 10.87,
        'Senglea': 0.14,
        'Cospicua': 1.3
    },
 'Senglea': 
    {
        'Valletta': 1.39,
        'Birkirkara': 5.25,
        'Sliema': 3.14,
        "St. Julian's": 4.29,
        'Mosta': 8.68,
        'Qormi': 4.33,
        'Żabbar': 1.96,
        'Żebbuġ': 7.11,
        'Marsaskala': 4.96,
        'Marsaxlokk': 5.64,
        'Mdina': 10.36,
        'Rabat': 10.79,
        'Birgu': 0.14,
        'Cospicua': 1.19
  },
 'Cospicua': 
    {
        'Valletta': 2.57,
        'Birkirkara': 5.73,
        'Sliema': 4.26,
        "St. Julian's": 5.35,
        'Mosta': 9.21,
        'Qormi': 4.27,
        'Żabbar': 1.39,
        'Żebbuġ': 7.08,
        'Marsaskala': 4.42,
        'Marsaxlokk': 4.53,
        'Mdina': 10.54,
        'Rabat': 10.91,
        'Birgu': 1.3,
        'Senglea': 1.19
    }
}


import time
import heapq

def dijkstra(graph, start, goal):
    # Öncelik kuyruğu (min-heap) oluştur
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))  # (mesafe, şehir)

    # Ziyaret edilen şehirler ve mesafeler
    distances = {city: float('inf') for city in graph}
    distances[start] = 0

    # Önceki şehirleri takip etmek için
    previous_nodes = {city: None for city in graph}

    while priority_queue:
        current_distance, current_city = heapq.heappop(priority_queue)

        # Eğer hedefe ulaşıldıysa, döngüyü kır
        if current_city == goal:
            break

        # Komşuları kontrol et
        for neighbor, weight in graph[current_city].items():
            distance = current_distance + weight

            # Daha kısa bir yol bulunduysa, güncelle
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_city
                heapq.heappush(priority_queue, (distance, neighbor))

    # En kısa yolu oluştur
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()

    return path, distances[goal]

import heapq

def a_star(graph, heuristics, start, goal):
    # Öncelik kuyruğu (min-heap) oluştur
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))  # (f(n), şehir)

    # Ziyaret edilen şehirler ve mesafeler
    g_scores = {city: float('inf') for city in graph}
    g_scores[start] = 0

    # Önceki şehirleri takip etmek için
    previous_nodes = {city: None for city in graph}

    while priority_queue:
        current_f_score, current_city = heapq.heappop(priority_queue)

        # Eğer hedefe ulaşıldıysa, döngüyü kır
        if current_city == goal:
            break

        # Komşuları kontrol et
        for neighbor, weight in graph[current_city].items():
            tentative_g_score = g_scores[current_city] + weight

            # Daha kısa bir yol bulunduysa, güncelle
            if tentative_g_score < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristics[neighbor].get(goal, float('inf'))
                heapq.heappush(priority_queue, (f_score, neighbor))
                previous_nodes[neighbor] = current_city

    # En kısa yolu oluştur
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()

    return path, g_scores[goal]

from collections import deque

def bidirectional_search(graph, start, goal):
    # İki yönlü arama için kuyruklar
    start_queue = deque([start])
    goal_queue = deque([goal])

    # Ziyaret edilen şehirler
    start_visited = {start: None}
    goal_visited = {goal: None}

    # İki yönlü arama sırasında kesişme noktası
    intersection = None

    while start_queue and goal_queue:
        # Başlangıçtan arama
        if start_queue:
            current_start = start_queue.popleft()
            for neighbor, _ in graph[current_start].items():
                if neighbor not in start_visited:
                    start_visited[neighbor] = current_start
                    start_queue.append(neighbor)
                    if neighbor in goal_visited:
                        intersection = neighbor
                        break

        # Hedeften arama
        if goal_queue:
            current_goal = goal_queue.popleft()
            for neighbor, _ in graph[current_goal].items():
                if neighbor not in goal_visited:
                    goal_visited[neighbor] = current_goal
                    goal_queue.append(neighbor)
                    if neighbor in start_visited:
                        intersection = neighbor
                        break

        # Eğer kesişme bulunduysa, döngüyü kır
        if intersection:
            break

    # Eğer kesişme yoksa, yol bulunamadı
    if not intersection:
        return None, float('inf')

    # En kısa yolu oluştur
    path = []
    current = intersection
    while current is not None:
        path.append(current)
        current = start_visited[current]
    path.reverse()

    current = goal_visited[intersection]
    while current is not None:
        path.append(current)
        current = goal_visited[current]

    # Toplam mesafeyi hesapla
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += graph[path[i]][path[i + 1]]

    return path, total_distance

from itertools import permutations

def csp_shortest_path(graph, start, goal):
    # Geçerli yolları bulmak için bir DFS (Derinlik Öncelikli Arama) fonksiyonu tanımlıyoruz
    def dfs(current, goal, visited, path, all_paths):
        # Eğer hedefe ulaşıldıysa, yolu kaydet
        if current == goal:
            all_paths.append(path[:])
            return

        # Komşuları ziyaret et
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, goal, visited, path, all_paths)
                # Geri izleme
                path.pop()
                visited.remove(neighbor)

    # Tüm geçerli yolları bul
    all_paths = []
    dfs(start, goal, {start}, [start], all_paths)

    # En kısa yolu ve mesafeyi bul
    shortest_path = None
    shortest_distance = float('inf')

    for path in all_paths:
        total_distance = 0
        valid = True

        # Yolun toplam mesafesini hesapla
        for i in range(len(path) - 1):
            if path[i + 1] in graph[path[i]]:
                total_distance += graph[path[i]][path[i + 1]]
            else:
                valid = False
                break

        # Eğer yol geçerliyse ve daha kısa bir mesafe bulduysak, güncelle
        if valid and total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = path

    return shortest_path, shortest_distance

def measure_execution_time_and_distance(func, *args, **kwargs):
    """
    Bir fonksiyonun çalışma süresini ve sonucunu ölçer.
    :param func: Çalıştırılacak fonksiyon
    :param args: Fonksiyonun argümanları
    :param kwargs: Fonksiyonun keyword argümanları
    :return: Fonksiyonun çıktısı (yol ve mesafe) ve çalışma süresi
    """
    start_time = time.time()  # Başlangıç zamanı
    result = func(*args, **kwargs)  # Fonksiyonu çalıştır
    end_time = time.time()  # Bitiş zamanı
    execution_time = end_time - start_time  # Çalışma süresi
    return result, execution_time

import time
from heapq import heappop, heappush

def bidirectional_a_star(graph, heuristics, start, goal):
    """
    Bidirectional A* algoritması.
    :param graph: Komşuluk listesi (şehirler arası mesafeler)
    :param heuristics: Heuristic dictionary (öklid mesafeleri)
    :param start: Başlangıç noktası
    :param goal: Hedef noktası
    :return: En kısa yol, toplam mesafe ve çalışma süresi
    """
    start_time = time.time()  # Çalışma süresini ölçmek için başlangıç zamanı

    # İleri ve geri arama için açık ve kapalı listeler
    open_forward = [(0, start, [start])]  # (öncelik, düğüm, yol)
    open_backward = [(0, goal, [goal])]
    closed_forward = {}
    closed_backward = {}

    # İleri ve geri maliyetler
    g_forward = {start: 0}
    g_backward = {goal: 0}

    # Kesişme noktası ve en kısa yol
    shortest_path = None
    shortest_distance = float('inf')

    while open_forward and open_backward:
        # İleri arama
        if open_forward:
            _, current_forward, path_forward = heappop(open_forward)
            if current_forward in closed_backward:
                # Kesişme bulundu
                total_distance = g_forward[current_forward] + g_backward[current_forward]
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
                    # İleri ve geri yolları birleştirirken kesişme düğümünü yalnızca bir kez ekle
                    shortest_path = path_forward + closed_backward[current_forward][::-1][1:]
            if current_forward not in closed_forward:
                closed_forward[current_forward] = path_forward
                for neighbor, distance in graph[current_forward].items():
                    tentative_g = g_forward[current_forward] + distance
                    if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                        g_forward[neighbor] = tentative_g
                        priority = tentative_g + heuristics[neighbor].get(goal, float('inf'))
                        heappush(open_forward, (priority, neighbor, path_forward + [neighbor]))

        # Geri arama
        if open_backward:
            _, current_backward, path_backward = heappop(open_backward)
            if current_backward in closed_forward:
                # Kesişme bulundu
                total_distance = g_backward[current_backward] + g_forward[current_backward]
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
                    # İleri ve geri yolları birleştirirken kesişme düğümünü yalnızca bir kez ekle
                    shortest_path = closed_forward[current_backward] + path_backward[::-1][1:]
            if current_backward not in closed_backward:
                closed_backward[current_backward] = path_backward
                for neighbor, distance in graph[current_backward].items():
                    tentative_g = g_backward[current_backward] + distance
                    if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                        g_backward[neighbor] = tentative_g
                        priority = tentative_g + heuristics[neighbor].get(start, float('inf'))
                        heappush(open_backward, (priority, neighbor, path_backward + [neighbor]))

    end_time = time.time()  # Çalışma süresini ölçmek için bitiş zamanı
    execution_time = end_time - start_time

    return shortest_path, shortest_distance, execution_time

from heapq import heappop, heappush
import time

def bidirectional_dijkstra(graph, start, goal):
    """
    Bidirectional Dijkstra algoritması.
    :param graph: Komşuluk listesi (şehirler arası mesafeler)
    :param start: Başlangıç noktası
    :param goal: Hedef noktası
    :return: En kısa yol, toplam mesafe ve çalışma süresi
    """
    start_time = time.time()  # Çalışma süresini ölçmek için başlangıç zamanı

    # İleri ve geri arama için açık ve kapalı listeler
    open_forward = [(0, start, [start])]  # (öncelik, düğüm, yol)
    open_backward = [(0, goal, [goal])]
    closed_forward = {}
    closed_backward = {}

    # İleri ve geri maliyetler
    g_forward = {start: 0}
    g_backward = {goal: 0}

    # Kesişme noktası ve en kısa yol
    shortest_path = None
    shortest_distance = float('inf')

    while open_forward and open_backward:
        # İleri arama
        if open_forward:
            current_distance, current_forward, path_forward = heappop(open_forward)
            if current_forward in closed_backward:
                # Kesişme bulundu
                total_distance = g_forward[current_forward] + g_backward[current_forward]
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
                    shortest_path = path_forward + closed_backward[current_forward][::-1][1:]
            if current_forward not in closed_forward:
                closed_forward[current_forward] = path_forward
                for neighbor, distance in graph[current_forward].items():
                    tentative_g = g_forward[current_forward] + distance
                    if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                        g_forward[neighbor] = tentative_g
                        heappush(open_forward, (tentative_g, neighbor, path_forward + [neighbor]))

        # Geri arama
        if open_backward:
            current_distance, current_backward, path_backward = heappop(open_backward)
            if current_backward in closed_forward:
                # Kesişme bulundu
                total_distance = g_backward[current_backward] + g_forward[current_backward]
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
                    shortest_path = closed_forward[current_backward] + path_backward[::-1][1:]
            if current_backward not in closed_backward:
                closed_backward[current_backward] = path_backward
                for neighbor, distance in graph[current_backward].items():
                    tentative_g = g_backward[current_backward] + distance
                    if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                        g_backward[neighbor] = tentative_g
                        heappush(open_backward, (tentative_g, neighbor, path_backward + [neighbor]))

    end_time = time.time()  # Çalışma süresini ölçmek için bitiş zamanı
    execution_time = end_time - start_time

    return shortest_path, shortest_distance, execution_time

# Mdina'dan Żabbar'a en kısa yolu hesapla
start = "Mdina"
goal = "Żabbar"
(path, distance), execution_time = measure_execution_time_and_distance(dijkstra, neighborhoods, "Mdina", "Żabbar")

print("           ")
print("           ")

print(f"En kısa yol dijkstra: {path}")
print(f"Toplam mesafe: {distance} km")
print(f"dijkstra algoritmasının çalışma süresi: {execution_time:.6f} saniye")

print("           ")
print("======================")
print("           ")

# (path, distance), execution_time = measure_execution_time_and_distance(bidirectional_search, neighborhoods, "Mdina", "Żabbar")

# print(f"En kısa yol bidirectional: {path}")
# print(f"Toplam mesafe: {distance} km")
# print(f"bidirectional algoritmasının çalışma süresi: {execution_time:.6f} saniye")

# print("           ")
# print("======================")
# print("           ")

(path, distance), execution_time = measure_execution_time_and_distance(a_star, neighborhoods, neighborhoods_heu, "Mdina", "Żabbar")

print(f"En kısa yol AStar: {path}")
print(f"Toplam mesafe: {distance} km")
print(f"AStar algoritmasının çalışma süresi: {execution_time:.6f} saniye")

# print("           ")
# print("======================")
# print("           ")

# (path, distance), execution_time = measure_execution_time_and_distance(csp_shortest_path, neighborhoods, "Mdina", "Żabbar")

# print(f"En kısa yol CSP: {path}")
# print(f"Toplam mesafe: {distance} km")
# print(f"CSP algoritmasının çalışma süresi: {execution_time:.6f} saniye")


print("           ")
print("======================")
print("           ")

# Bidirectional A* algoritmasını çalıştır
path, distance, execution_time = bidirectional_a_star(neighborhoods, neighborhoods_heu, start, goal)

print(f"En kısa yol bidirectional_a_star: {path}")
print(f"Toplam mesafe: {distance} km")
print(f"Çalışma süresi bidirectional_a_star: {execution_time:.6f} saniye")

print("           ")
print("======================")
print("           ")

# Bidirectional Dijkstra algoritmasını çalıştır
path, distance, execution_time = bidirectional_dijkstra(neighborhoods, start, goal)

print(f"En kısa yol: {path}")
print(f"Toplam mesafe: {distance} km")
print(f"Çalışma süresi: {execution_time:.6f} saniye")