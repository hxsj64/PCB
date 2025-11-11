import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import json
from collections import defaultdict, deque
import pickle

# 设置随机种子
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

def load_nodes(node_file):
    """加载节点信息"""
    nodes = {}
    with open(node_file, 'r') as f:
        lines = f.readlines()
        # 跳过标题行
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(',')
                node_id = int(parts[0].strip())
                width = float(parts[1].strip())
                height = float(parts[2].strip())
                nodes[node_id] = {'width': width, 'height': height}
    return nodes

def load_edges(edge_file):
    """加载边信息"""
    edges = []
    with open(edge_file, 'r') as f:
        lines = f.readlines()
        # 跳过标题行
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(',')
                start_node = int(parts[0].strip())
                end_node = int(parts[1].strip())
                edges.append((start_node, end_node))
    return edges

def check_nodes_overlap(nodes, node_positions, min_spacing=1.0):
    """检查节点间是否有重叠，并返回重叠信息"""
    overlaps = []
    node_ids = list(node_positions.keys())
    
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            node1_id = node_ids[i]
            node2_id = node_ids[j]
            
            x1, y1 = node_positions[node1_id]
            w1, h1 = nodes[node1_id]['width'], nodes[node1_id]['height']
            x2, y2 = node_positions[node2_id]
            w2, h2 = nodes[node2_id]['width'], nodes[node2_id]['height']
            
            # 检查矩形重叠（考虑最小间距）
            if (x1 - min_spacing < x2 + w2 + min_spacing and 
                x1 + w1 + min_spacing > x2 - min_spacing and 
                y1 - min_spacing < y2 + h2 + min_spacing and 
                y1 + h1 + min_spacing > y2 - min_spacing):
                
                # 计算重叠区域
                overlap_x = max(x1, x2)
                overlap_y = max(y1, y2)
                overlap_w = min(x1 + w1, x2 + w2) - overlap_x
                overlap_h = min(y1 + h1, y2 + h2) - overlap_y
                
                # 计算重叠面积
                overlap_area = overlap_w * overlap_h if overlap_w > 0 and overlap_h > 0 else 0
                
                overlaps.append({
                    'node1': node1_id,
                    'node2': node2_id,
                    'overlap_area': overlap_area,
                    'overlap_rect': (overlap_x, overlap_y, overlap_w, overlap_h)
                })
    
    return overlaps

def improved_bfs_alignment_layout(nodes, edges, D=150, min_spacing_factor=2.5):
    """改进的BFS布局算法"""
    print("📐 使用改进的BFS顺序布局...")
    
    # 构建邻接表
    graph = defaultdict(list)
    for start, end in edges:
        graph[start].append(end)
        graph[end].append(start)
    
    # 找到起始节点（选择度数最高的节点作为起点）
    if not graph:
        start_node = list(nodes.keys())[0]
    else:
        degrees = {node: -1 * len(neighbors) for node, neighbors in graph.items()}
        start_node = max(degrees.items(), key=lambda x: x[1])[0]
    
    # 计算平均节点尺寸
    avg_width = np.mean([node['width'] for node in nodes.values()])
    avg_height = np.mean([node['height'] for node in nodes.values()])
    
    # 优化间距计算
    horizontal_spacing = avg_width * min_spacing_factor * 1.2
    vertical_spacing = avg_height * min_spacing_factor
    
    print(f"  水平间距: {horizontal_spacing:.2f}")
    print(f"  垂直间距: {vertical_spacing:.2f}")
    
    # 初始化
    positions = {}
    visited = set()
    grid_positions = {}
    row_col_map = {}
    
    # 放置起始节点
    center_x = D / 2
    center_y = D / 2
    start_width = nodes[start_node]['width']
    start_height = nodes[start_node]['height']
    positions[start_node] = (center_x - start_width / 2, center_y - start_height / 2)
    grid_positions[(0, 0)] = start_node
    row_col_map[start_node] = (0, 0)
    visited.add(start_node)
    
    # 使用队列进行BFS
    queue = deque([(start_node, 0, 0)])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
    
    while queue:
        current_node, current_row, current_col = queue.popleft()
        
        # 处理邻居节点
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                
                placed = False
                # 按方向顺序尝试放置
                for dr, dc in directions:
                    new_row = current_row + dr
                    new_col = current_col + dc
                    
                    if (new_row, new_col) not in grid_positions:
                        # 计算坐标
                        x = center_x + new_col * horizontal_spacing
                        y = center_y - new_row * vertical_spacing
                        
                        width = nodes[neighbor]['width']
                        height = nodes[neighbor]['height']
                        x -= width / 2
                        y -= height / 2
                        
                        # 边界检查
                        x = max(0, min(D - width, x))
                        y = max(0, min(D - height, y))
                        
                        # 检查重叠
                        temp_positions = positions.copy()
                        temp_positions[neighbor] = (x, y)
                        overlaps = check_nodes_overlap(nodes, temp_positions)
                        
                        if not overlaps:
                            positions[neighbor] = (x, y)
                            grid_positions[(new_row, new_col)] = neighbor
                            row_col_map[neighbor] = (new_row, new_col)
                            queue.append((neighbor, new_row, new_col))
                            placed = True
                            break
                
                if not placed:
                    # 螺旋搜索空位
                    for radius in range(1, 8):
                        found = False
                        for dr in range(-radius, radius + 1):
                            for dc in range(-radius, radius + 1):
                                if abs(dr) + abs(dc) == radius:
                                    new_row = current_row + dr
                                    new_col = current_col + dc
                                    
                                    if (new_row, new_col) not in grid_positions:
                                        x = center_x + new_col * horizontal_spacing
                                        y = center_y - new_row * vertical_spacing
                                        
                                        width = nodes[neighbor]['width']
                                        height = nodes[neighbor]['height']
                                        x -= width / 2
                                        y -= height / 2
                                        
                                        x = max(0, min(D - width, x))
                                        y = max(0, min(D - height, y))
                                        
                                        temp_positions = positions.copy()
                                        temp_positions[neighbor] = (x, y)
                                        overlaps = check_nodes_overlap(nodes, temp_positions)
                                        
                                        if not overlaps:
                                            positions[neighbor] = (x, y)
                                            grid_positions[(new_row, new_col)] = neighbor
                                            row_col_map[neighbor] = (new_row, new_col)
                                            queue.append((neighbor, new_row, new_col))
                                            found = True
                                            break
                            if found:
                                break
                        if found:
                            break
    
    # 处理孤立节点
    all_nodes = set(nodes.keys())
    placed_nodes = set(positions.keys())
    isolated_nodes = all_nodes - placed_nodes
    
    if isolated_nodes:
        print(f"⚠️  处理 {len(isolated_nodes)} 个孤立节点")
        for node_id in isolated_nodes:
            width = nodes[node_id]['width']
            height = nodes[node_id]['height']
            
            # 在空白区域随机放置
            placed = False
            for attempt in range(100):
                x = random.uniform(0, D - width)
                y = random.uniform(0, D - height)
                
                temp_positions = positions.copy()
                temp_positions[node_id] = (x, y)
                overlaps = check_nodes_overlap(nodes, temp_positions)
                
                if not overlaps:
                    positions[node_id] = (x, y)
                    placed = True
                    break
            
            if not placed:
                # 强制放置
                positions[node_id] = (D/2, D/2)
    
    # 将布局居中
    positions = center_layout(nodes, positions, D)
    
    # 最终重叠检查
    final_overlaps = check_nodes_overlap(nodes, positions)
    if final_overlaps:
        print(f"⚠️  警告: 布局完成后仍有 {len(final_overlaps)} 处重叠")
    else:
        print("✅ 所有节点均无重叠")
    
    return positions

def center_layout(nodes, node_positions, D):
    """将布局居中到画布中心"""
    if not node_positions:
        return node_positions
    
    # 计算当前布局的边界
    min_x = min(pos[0] for pos in node_positions.values())
    max_x = max(pos[0] + nodes[node_id]['width'] for node_id, pos in node_positions.items())
    min_y = min(pos[1] for pos in node_positions.values())
    max_y = max(pos[1] + nodes[node_id]['height'] for node_id, pos in node_positions.items())
    
    # 计算当前布局的中心
    current_center_x = (min_x + max_x) / 2
    current_center_y = (min_y + max_y) / 2
    
    # 计算目标中心（画布中心）
    target_center_x = D / 2
    target_center_y = D / 2
    
    # 计算偏移量
    offset_x = target_center_x - current_center_x
    offset_y = target_center_y - current_center_y
    
    # 应用偏移量
    centered_positions = {}
    for node_id, pos in node_positions.items():
        new_x = pos[0] + offset_x
        new_y = pos[1] + offset_y
        
        # 边界检查
        width = nodes[node_id]['width']
        height = nodes[node_id]['height']
        new_x = max(0, min(D - width, new_x))
        new_y = max(0, min(D - height, new_y))
        
        centered_positions[node_id] = (new_x, new_y)
    
    return centered_positions

def manhattan_distance(p1, p2):
    """计算曼哈顿距离"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def calculate_hpwl(points):
    """计算一组点的半周线长（HPWL）"""
    if not points:
        return 0
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    return (max_x - min_x) + (max_y - min_y)

def generate_optimized_connection_points(node_pos, width, height, padding=0.5):
    """生成优化的连接点"""
    x, y = node_pos
    points = []
    
    # 在每个边界上生成多个点
    num_points_per_side = 2
    
    # 上边界
    for i in range(1, num_points_per_side + 1):
        ratio = i / (num_points_per_side + 1)
        points.append((x + ratio * width, y + height + padding))
    
    # 下边界
    for i in range(1, num_points_per_side + 1):
        ratio = i / (num_points_per_side + 1)
        points.append((x + ratio * width, y - padding))
    
    # 左边界
    for i in range(1, num_points_per_side + 1):
        ratio = i / (num_points_per_side + 1)
        points.append((x - padding, y + ratio * height))
    
    # 右边界
    for i in range(1, num_points_per_side + 1):
        ratio = i / (num_points_per_side + 1)
        points.append((x + width + padding, y + ratio * height))
    
    return points

def is_point_inside_rectangle(point, rect_pos, width, height):
    """检查点是否在矩形内部"""
    px, py = point
    rx, ry = rect_pos
    return (rx <= px <= rx + width) and (ry <= py <= ry + height)

def does_segment_intersect_rectangle(p1, p2, rect_pos, width, height):
    """检查线段是否与矩形相交"""
    rx, ry = rect_pos
    
    rect_edges = [
        ((rx, ry), (rx + width, ry)),
        ((rx, ry + height), (rx + width, ry + height)),
        ((rx, ry), (rx, ry + height)),
        ((rx + width, ry), (rx + width, ry + height))
    ]
    
    for edge in rect_edges:
        if do_segments_intersect(p1, p2, edge[0], edge[1]):
            return True
    
    if (is_point_inside_rectangle(p1, rect_pos, width, height) or 
        is_point_inside_rectangle(p2, rect_pos, width, height)):
        return True
    
    return False

def do_segments_intersect(p1, p2, p3, p4):
    """检查两个线段是否相交"""
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def on_segment(p, q, r):
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and 
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
            return True
        return False
    
    d1 = cross_product(p3, p4, p1)
    d2 = cross_product(p3, p4, p2)
    d3 = cross_product(p1, p2, p3)
    d4 = cross_product(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    
    return False

def is_path_valid(path, nodes, node_positions, source_node_id, target_node_id):
    """检查路径是否有效"""
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        
        for node_id, node_info in nodes.items():
            if node_id == source_node_id or node_id == target_node_id:
                continue
                
            node_x, node_y = node_positions[node_id]
            width = node_info['width']
            height = node_info['height']
            
            if does_segment_intersect_rectangle(p1, p2, (node_x, node_y), width, height):
                return False
    
    return True

def check_path_overlap(path, existing_paths, min_spacing=0.3):
    """检查路径是否与现有路径重叠，考虑最小间距"""
    for i in range(len(path) - 1):
        seg1 = (path[i], path[i+1])
        
        for existing_path in existing_paths:
            for j in range(len(existing_path) - 1):
                seg2 = (existing_path[j], existing_path[j+1])
                
                # 检查线段是否平行且距离过近
                if are_segments_parallel_and_close(seg1, seg2, min_spacing):
                    return True
                
                # 检查线段是否重叠
                if do_segments_overlap(seg1[0], seg1[1], seg2[0], seg2[1]):
                    return True
    
    return False

def are_segments_parallel_and_close(seg1, seg2, min_spacing=0.3):
    """检查两个线段是否平行且距离过近"""
    p1, p2 = seg1
    p3, p4 = seg2
    
    # 检查是否水平平行
    if abs(p1[1] - p2[1]) < 1e-6 and abs(p3[1] - p4[1]) < 1e-6:
        # 水平线段
        y1 = p1[1]
        y2 = p3[1]
        if abs(y1 - y2) < min_spacing:
            # 检查x范围是否有重叠
            x1_min, x1_max = min(p1[0], p2[0]), max(p1[0], p2[0])
            x2_min, x2_max = min(p3[0], p4[0]), max(p3[0], p4[0])
            return not (x1_max < x2_min or x2_max < x1_min)
    
    # 检查是否垂直平行
    elif abs(p1[0] - p2[0]) < 1e-6 and abs(p3[0] - p4[0]) < 1e-6:
        # 垂直线段
        x1 = p1[0]
        x2 = p3[0]
        if abs(x1 - x2) < min_spacing:
            # 检查y范围是否有重叠
            y1_min, y1_max = min(p1[1], p2[1]), max(p1[1], p2[1])
            y2_min, y2_max = min(p3[1], p4[1]), max(p3[1], p4[1])
            return not (y1_max < y2_min or y2_max < y1_min)
    
    return False

def do_segments_overlap(p1, p2, p3, p4, tolerance=1e-6):
    """检查两个线段是否有重叠（不仅仅是交点）"""
    # 检查两个线段是否共线
    if not are_segments_collinear(p1, p2, p3, p4, tolerance):
        return False
    
    # 如果共线，检查是否有重叠部分
    seg1_x = sorted([p1[0], p2[0]])
    seg1_y = sorted([p1[1], p2[1]])
    seg2_x = sorted([p3[0], p4[0]])
    seg2_y = sorted([p3[1], p4[1]])
    
    # 检查x轴和y轴上的重叠
    x_overlap = max(0, min(seg1_x[1], seg2_x[1]) - max(seg1_x[0], seg2_x[0]))
    y_overlap = max(0, min(seg1_y[1], seg2_y[1]) - max(seg1_y[0], seg2_y[0]))
    
    # 如果有显著重叠，返回True
    return x_overlap > tolerance and y_overlap > tolerance

def are_segments_collinear(p1, p2, p3, p4, tolerance=1e-6):
    """检查两个线段是否共线"""
    # 计算三个点的向量
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    v3 = (p4[0] - p1[0], p4[1] - p1[1])
    
    # 计算叉积
    cross1 = v1[0] * v2[1] - v1[1] * v2[0]
    cross2 = v1[0] * v3[1] - v1[1] * v3[0]
    
    # 如果叉积都很小，则四个点共线
    return abs(cross1) < tolerance and abs(cross2) < tolerance

def generate_straight_paths(start, end, horizontal_first=True):
    """生成直线路径"""
    x1, y1 = start
    x2, y2 = end
    
    if horizontal_first:
        return [[start, (x2, y1), end]]
    else:
        return [[start, (x1, y2), end]]

def generate_detour_paths(start, end, min_spacing=0.3):
    """生成绕路路径，确保间距"""
    x1, y1 = start
    x2, y2 = end
    
    paths = []
    
    # 计算基本偏移
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    detour_dist = max(min_spacing * 2, min(dx, dy) * 0.3)
    
    # 水平优先的绕路
    if dx > dy:
        # 向上绕路
        paths.append([start, (x1, y1 - detour_dist), (x2, y1 - detour_dist), end])
        # 向下绕路
        paths.append([start, (x1, y1 + detour_dist), (x2, y1 + detour_dist), end])
        # Z字形绕路
        mid_x = (x1 + x2) / 2
        paths.append([start, (mid_x, y1), (mid_x, y2), end])
    else:
        # 向左绕路
        paths.append([start, (x1 - detour_dist, y1), (x1 - detour_dist, y2), end])
        # 向右绕路
        paths.append([start, (x1 + detour_dist, y1), (x1 + detour_dist, y2), end])
        # Z字形绕路
        mid_y = (y1 + y2) / 2
        paths.append([start, (x1, mid_y), (x2, mid_y), end])
    
    return paths

def select_optimized_connection_points(node1_pos, node2_pos, node1_info, node2_info):
    """选择优化的连接点"""
    x1, y1 = node1_pos
    w1, h1 = node1_info['width'], node1_info['height']
    x2, y2 = node2_pos
    w2, h2 = node2_info['width'], node2_info['height']
    
    points1 = generate_optimized_connection_points(node1_pos, w1, h1)
    points2 = generate_optimized_connection_points(node2_pos, w2, h2)
    
    return points1, points2

def optimized_hpwl_router(node1_pos, node2_pos, node1_info, node2_info, nodes, node_positions, existing_paths=[]):
    """优化的HPWL布线器，避免重叠"""
    candidate_connections = []
    
    # 选择连接点
    node1_points, node2_points = select_optimized_connection_points(
        node1_pos, node2_pos, node1_info, node2_info)
    
    # 确定优先方向
    dx = abs(node2_pos[0] - node1_pos[0])
    dy = abs(node2_pos[1] - node1_pos[1])
    horizontal_first = dx > dy
    
    for p1 in node1_points:
        for p2 in node2_points:
            # 生成直线路径
            possible_paths = generate_straight_paths(p1, p2, horizontal_first)
            
            # 生成绕路路径
            detour_paths = generate_detour_paths(p1, p2)
            possible_paths.extend(detour_paths)
            
            for path in possible_paths:
                if is_path_valid(path, nodes, node_positions, None, None):
                    # 检查路径重叠
                    if check_path_overlap(path, existing_paths):
                        continue
                    
                    # 计算路径长度
                    path_length = sum(manhattan_distance(path[i], path[i+1]) for i in range(len(path)-1))
                    
                    # 计算转弯次数
                    turn_count = 0
                    for i in range(1, len(path)-1):
                        prev_segment = (path[i-1][0] - path[i][0], path[i-1][1] - path[i][1])
                        next_segment = (path[i][0] - path[i+1][0], path[i][1] - path[i+1][1])
                        if prev_segment[0] != next_segment[0] or prev_segment[1] != next_segment[1]:
                            turn_count += 1
                    
                    # 计算与其他路径的交点数量
                    intersections = count_path_intersections(path, existing_paths)
                    
                    # 综合评分：优先短路径、少转弯、少交点、无重叠
                    score = (path_length * 2 + 
                            turn_count * 50 + 
                            intersections * 100)
                    
                    candidate_connections.append((p1, p2, path, path_length, turn_count, intersections, score))
    
    # 选择最优连接
    if candidate_connections:
        candidate_connections.sort(key=lambda c: c[6])  # 按分数排序
        return candidate_connections[0]
    
    return None

def count_path_intersections(path, existing_paths):
    """计算路径与其他路径的交点数量"""
    intersections = 0
    for i in range(len(path) - 1):
        seg1 = (path[i], path[i+1])
        for existing_path in existing_paths:
            for j in range(len(existing_path) - 1):
                seg2 = (existing_path[j], existing_path[j+1])
                if do_segments_intersect(seg1[0], seg1[1], seg2[0], seg2[1]):
                    intersections += 1
    return intersections

def adjust_wire_spacing(node_positions, nodes, edges, D, min_spacing=0.3):
    """调整连线间距，确保平行连线间距至少为min_spacing"""
    print("📏 调整连线间距...")
    
    # 收集所有路径
    existing_paths = []
    all_segments = []
    
    for start_node, end_node in edges:
        if start_node in node_positions and end_node in node_positions:
            start_pos = node_positions[start_node]
            end_pos = node_positions[end_node]
            start_info = nodes[start_node]
            end_info = nodes[end_node]
            
            connection_info = optimized_hpwl_router(
                start_pos, end_pos, start_info, end_info, nodes, node_positions, existing_paths
            )
            
            if connection_info:
                p1, p2, path, path_length, turn_count, intersections, score = connection_info
                existing_paths.append(path)
                
                # 收集所有线段
                for i in range(len(path) - 1):
                    all_segments.append((path[i], path[i+1]))
    
    # 分组平行线段
    horizontal_segments = []
    vertical_segments = []
    
    for seg in all_segments:
        p1, p2 = seg
        if abs(p1[1] - p2[1]) < 1e-6:  # 水平线段
            horizontal_segments.append(seg)
        elif abs(p1[0] - p2[0]) < 1e-6:  # 垂直线段
            vertical_segments.append(seg)
    
    # 调整水平线段间距
    horizontal_groups = defaultdict(list)
    for seg in horizontal_segments:
        y = (seg[0][1] + seg[1][1]) / 2
        y_key = round(y / min_spacing) * min_spacing
        horizontal_groups[y_key].append(seg)
    
    for y_key, segments in horizontal_groups.items():
        if len(segments) > 1:
            # 按x坐标排序
            segments.sort(key=lambda s: min(s[0][0], s[1][0]))
            
            # 重新分配y坐标，确保间距
            target_ys = np.linspace(y_key - min_spacing * (len(segments)-1)/2, 
                                  y_key + min_spacing * (len(segments)-1)/2, len(segments))
            
            for i, seg in enumerate(segments):
                old_y = (seg[0][1] + seg[1][1]) / 2
                new_y = target_ys[i]
                delta_y = new_y - old_y
                
                # 更新路径中对应的点
                for path in existing_paths:
                    for j in range(len(path)):
                        if abs(path[j][1] - old_y) < 1e-6 and (
                            (min(seg[0][0], seg[1][0]) <= path[j][0] <= max(seg[0][0], seg[1][0]))):
                            path[j] = (path[j][0], new_y)
    
    # 调整垂直线段间距
    vertical_groups = defaultdict(list)
    for seg in vertical_segments:
        x = (seg[0][0] + seg[1][0]) / 2
        x_key = round(x / min_spacing) * min_spacing
        vertical_groups[x_key].append(seg)
    
    for x_key, segments in vertical_groups.items():
        if len(segments) > 1:
            # 按y坐标排序
            segments.sort(key=lambda s: min(s[0][1], s[1][1]))
            
            # 重新分配x坐标，确保间距
            target_xs = np.linspace(x_key - min_spacing * (len(segments)-1)/2, 
                                  x_key + min_spacing * (len(segments)-1)/2, len(segments))
            
            for i, seg in enumerate(segments):
                old_x = (seg[0][0] + seg[1][0]) / 2
                new_x = target_xs[i]
                delta_x = new_x - old_x
                
                # 更新路径中对应的点
                for path in existing_paths:
                    for j in range(len(path)):
                        if abs(path[j][0] - old_x) < 1e-6 and (
                            (min(seg[0][1], seg[1][1]) <= path[j][1] <= max(seg[0][1], seg[1][1]))):
                            path[j] = (new_x, path[j][1])
    
    return existing_paths

def evaluate_wiring_quality(nodes, node_positions, edges, existing_paths=None):
    """评估连线质量"""
    if existing_paths is None:
        existing_paths = []
    
    connection_stats = {
        'total_connections': 0,
        'successful_connections': 0,
        'failed_connections': 0,
        'total_path_length': 0,
        'total_turns': 0,
        'total_intersections': 0,
        'connection_details': []
    }
    
    for start_node, end_node in edges:
        if start_node in node_positions and end_node in node_positions:
            start_pos = node_positions[start_node]
            end_pos = node_positions[end_node]
            start_info = nodes[start_node]
            end_info = nodes[end_node]
            
            connection_info = optimized_hpwl_router(
                start_pos, end_pos, start_info, end_info, nodes, node_positions, existing_paths
            )
            
            connection_stats['total_connections'] += 1
            
            if connection_info:
                p1, p2, path, path_length, turn_count, intersections, score = connection_info
                
                connection_stats['successful_connections'] += 1
                connection_stats['total_path_length'] += path_length
                connection_stats['total_turns'] += turn_count
                connection_stats['total_intersections'] += intersections
                
                existing_paths.append(path)
                
                connection_stats['connection_details'].append({
                    'start_node': start_node,
                    'end_node': end_node,
                    'path_length': path_length,
                    'turns': turn_count,
                    'intersections': intersections,
                    'success': True
                })
            else:
                connection_stats['failed_connections'] += 1
                connection_stats['connection_details'].append({
                    'start_node': start_node,
                    'end_node': end_node,
                    'success': False
                })
    
    # 计算综合分数
    completion_rate = connection_stats['successful_connections'] / connection_stats['total_connections'] if connection_stats['total_connections'] > 0 else 0
    avg_path_length = connection_stats['total_path_length'] / connection_stats['successful_connections'] if connection_stats['successful_connections'] > 0 else 0
    avg_turns = connection_stats['total_turns'] / connection_stats['successful_connections'] if connection_stats['successful_connections'] > 0 else 0
    avg_intersections = connection_stats['total_intersections'] / connection_stats['successful_connections'] if connection_stats['successful_connections'] > 0 else 0
    
    score = ((1 - completion_rate) * 1000 + 
             avg_path_length * 0.1 + 
             avg_turns * 10 + 
             avg_intersections * 50)
    
    return score, connection_stats, existing_paths

def create_optimized_visualization(nodes, node_positions, edges, save_path, D=100, title="Network Layout"):
    """创建优化的可视化 - 避免连线重叠"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    # 绘制节点
    for node_id, pos in node_positions.items():
        width = nodes[node_id]['width']
        height = nodes[node_id]['height']
        x, y = pos
        
        rect = plt.Rectangle((x, y), width, height, color='lightblue', alpha=0.8, 
                             edgecolor='darkblue', linewidth=1.0)
        ax.add_patch(rect)
        
        ax.text(x + width/2, y + height/2, f'{node_id}', 
                ha='center', va='center', fontweight='bold', color='darkblue', fontsize=8)
    
    # 布线
    existing_paths = []
    connection_stats = {
        'total_connections': 0,
        'successful_connections': 0,
        'failed_connections': 0,
        'total_path_length': 0,
        'total_turns': 0,
        'total_intersections': 0
    }
    
    # 首先收集所有路径
    all_paths = []
    for start_node, end_node in edges:
        if start_node in node_positions and end_node in node_positions:
            start_pos = node_positions[start_node]
            end_pos = node_positions[end_node]
            start_info = nodes[start_node]
            end_info = nodes[end_node]
            
            connection_info = optimized_hpwl_router(
                start_pos, end_pos, start_info, end_info, nodes, node_positions, existing_paths
            )
            
            connection_stats['total_connections'] += 1
            
            if connection_info:
                p1, p2, path, path_length, turn_count, intersections, score = connection_info
                
                connection_stats['successful_connections'] += 1
                connection_stats['total_path_length'] += path_length
                connection_stats['total_turns'] += turn_count
                connection_stats['total_intersections'] += intersections
                
                all_paths.append((path, turn_count, intersections))
                existing_paths.append(path)
            else:
                connection_stats['failed_connections'] += 1
    
    # 调整连线间距
    adjusted_paths = adjust_wire_spacing(node_positions, nodes, edges, D)
    
    # 绘制调整后的路径
    for path, turn_count, intersections in all_paths:
        # 根据转弯次数选择颜色
        if turn_count <= 1:
            line_color = 'green'
        elif turn_count == 2:
            line_color = 'orange'
        else:
            line_color = 'red'
        
        # 根据交点数量调整线宽
        line_width = max(0.3, 1.0 - intersections * 0.1)
        
        # 绘制路径
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax.plot(x_coords, y_coords, color=line_color, alpha=0.7, 
               linewidth=line_width, marker='o', markersize=1.5)
        
        # 标记连接点
        if len(path) > 0:
            ax.plot(path[0][0], path[0][1], 's', color='darkgreen', markersize=2.0)
            ax.plot(path[-1][0], path[-1][1], 's', color='darkviolet', markersize=2.0)
    
    # 添加统计信息
    completion_rate = (connection_stats['successful_connections'] / connection_stats['total_connections'] * 100) if connection_stats['total_connections'] > 0 else 0
    
    stats_text = f"Nodes: {len(nodes)}\n"
    stats_text += f"Edges: {len(edges)}\n"
    stats_text += f"Connections: {connection_stats['successful_connections']}/{connection_stats['total_connections']}\n"
    stats_text += f"Success Rate: {completion_rate:.1f}%\n"
    stats_text += f"Total Intersections: {connection_stats['total_intersections']}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return connection_stats

def optimized_main():
    """优化的主函数 - 避免连线重叠"""
    output_dir = "non_overlapping_wires_layout"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading node and edge data...")
    nodes = load_nodes('_node.txt')
    edges = load_edges('_edge.txt')
    
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
    
    # 计算布局区域
    max_width = max(node['width'] for node in nodes.values())
    max_height = max(node['height'] for node in nodes.values())
    total_area = sum(node['width'] * node['height'] for node in nodes.values())
    D = max(210, int((total_area * 5) ** 0.5))
    
    print(f"Layout area size: {D}x{D}")
    
    # 使用改进的布局算法
    print("📐 Using optimized BFS layout...")
    node_positions = improved_bfs_alignment_layout(nodes, edges, D, min_spacing_factor=2.5)
    
    # 创建可视化
    print("Creating non-overlapping wires visualization...")
    viz_path = os.path.join(output_dir, "non_overlapping_wires_layout.png")
    connection_stats = create_optimized_visualization(
        nodes, node_positions, edges, viz_path, D,
        title="Non-Overlapping Wires Layout (Min Spacing: 0.3)"
    )
    
    # 评估连线质量
    print("Evaluating wiring quality...")
    score, detailed_stats, existing_paths = evaluate_wiring_quality(nodes, node_positions, edges)
    
    # 保存结果
    summary_path = os.path.join(output_dir, "non_overlapping_layout_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Non-Overlapping Wires Layout Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total nodes: {len(nodes)}\n")
        f.write(f"Total edges: {len(edges)}\n")
        f.write(f"Layout area: {D}x{D}\n")
        f.write(f"Wiring quality score: {score:.2f}\n")
        f.write(f"Successful connections: {connection_stats['successful_connections']}/{connection_stats['total_connections']}\n")
        f.write(f"Completion rate: {(connection_stats['successful_connections']/connection_stats['total_connections']*100):.1f}%\n")
        f.write(f"Failed connections: {connection_stats['failed_connections']}\n")
        f.write(f"Total intersections: {connection_stats['total_intersections']}\n")
        
        successful_details = [d for d in detailed_stats['connection_details'] if d['success']]
        if successful_details:
            avg_length = sum(d['path_length'] for d in successful_details) / len(successful_details)
            avg_turns = sum(d['turns'] for d in successful_details) / len(successful_details)
            avg_intersections = sum(d['intersections'] for d in successful_details) / len(successful_details)
            f.write(f"Average path length: {avg_length:.2f}\n")
            f.write(f"Average turns per connection: {avg_turns:.2f}\n")
            f.write(f"Average intersections per connection: {avg_intersections:.2f}\n")
        
        f.write("\nConnection details:\n")
        for detail in detailed_stats['connection_details']:
            if detail['success']:
                f.write(f"  {detail['start_node']}-{detail['end_node']}: Length: {detail['path_length']:.2f}, Turns: {detail['turns']}, Intersections: {detail['intersections']}\n")
            else:
                f.write(f"  {detail['start_node']}-{detail['end_node']}: FAILED\n")
    
    print(f"Results saved in '{output_dir}' directory")
    print(f"Wiring quality score: {score:.2f}")
    print(f"Successful connections: {connection_stats['successful_connections']}/{connection_stats['total_connections']}")
    print(f"Failed connections: {connection_stats['failed_connections']}")
    print(f"Total intersections: {connection_stats['total_intersections']}")
    
    return node_positions, connection_stats, score

if __name__ == "__main__":
    optimized_main()