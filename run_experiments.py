import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from distance_metrics import DISTANCE_METRICS
from environment import PowerGridEnv
from visualizer import PowerGridVisualizer, create_training_gif
from benchmark_algorithms import (
    generate_optimized_substations, 
    generate_training_history,
    kmeans_solution,
    random_solution
)

# 设置随机种子
random.seed(123)
np.random.seed(123)

def generate_generators(N, D, K):
    """生成不重叠的发电机位置"""
    generators = []
    while len(generators) < N:
        x = random.randint(0, D - K)
        y = random.randint(0, D - K)
        new_point = (x, y)
        
        # 检查重叠
        overlap = False
        for px, py in generators:
            if max(abs(x - px), abs(y - py)) < K:
                overlap = True
                break
        
        if not overlap:
            generators.append(new_point)
    
    return generators

def generate_non_overlapping_substations(generators, M, D, K, max_attempts=1000):
    """生成不与发电机重叠且不在同一横线或竖线的变电站位置"""
    substations = []
    used_x = set()  # 记录已使用的x坐标
    used_y = set()  # 记录已使用的y坐标
    attempts = 0
    
    while len(substations) < M and attempts < max_attempts:
        x = random.randint(0, D - K)
        y = random.randint(0, D - K)
        new_point = (x, y)
        
        # 检查与发电机的重叠
        overlap_with_generator = False
        for gx, gy in generators:
            if max(abs(x - gx), abs(y - gy)) < K:
                overlap_with_generator = True
                break
        
        # 检查与其他变电站的重叠
        overlap_with_substation = False
        for sx, sy in substations:
            if max(abs(x - sx), abs(y - sy)) < K:
                overlap_with_substation = True
                break
        
        # 检查是否在同一横线或竖线
        same_line = (x in used_x) or (y in used_y)
        
        if not overlap_with_generator and not overlap_with_substation and not same_line:
            substations.append(new_point)
            used_x.add(x)
            used_y.add(y)
        
        attempts += 1
    
    # 如果无法生成足够的不重叠且不在同一线的变电站，放宽条件
    if len(substations) < M:
        print(f"Warning: Could not generate {M} non-overlapping and non-aligned substations after {max_attempts} attempts.")
        print("Trying with relaxed constraints (allowing same line but not same position)...")
        
        # 放宽条件：允许在同一线，但不能在同一位置
        while len(substations) < M and attempts < max_attempts * 2:
            x = random.randint(0, D - K)
            y = random.randint(0, D - K)
            new_point = (x, y)
            
            # 检查重叠
            overlap = False
            for gx, gy in generators:
                if max(abs(x - gx), abs(y - gy)) < K:
                    overlap = True
                    break
            
            for sx, sy in substations:
                if max(abs(x - sx), abs(y - sy)) < K:
                    overlap = True
                    break
            
            if not overlap:
                substations.append(new_point)
                used_x.add(x)
                used_y.add(y)
            
            attempts += 1
    
    # 如果仍然无法生成足够变电站，使用网格搜索方法
    if len(substations) < M:
        print("Using grid search method...")
        all_positions = [(x, y) for x in range(0, D-K+1) for y in range(0, D-K+1)]
        random.shuffle(all_positions)
        
        for pos in all_positions:
            if len(substations) >= M:
                break
                
            x, y = pos
            
            # 优先选择不在已使用线上的位置
            if x not in used_x and y not in used_y:
                overlap = False
                
                # 检查与发电机的重叠
                for gx, gy in generators:
                    if max(abs(x - gx), abs(y - gy)) < K:
                        overlap = True
                        break
                
                # 检查与其他变电站的重叠
                for sx, sy in substations:
                    if max(abs(x - sx), abs(y - sy)) < K:
                        overlap = True
                        break
                
                if not overlap:
                    substations.append((x, y))
                    used_x.add(x)
                    used_y.add(y)
        
        # 如果仍然不够，放宽到允许在同一线
        if len(substations) < M:
            for pos in all_positions:
                if len(substations) >= M:
                    break
                    
                x, y = pos
                overlap = False
                
                for gx, gy in generators:
                    if max(abs(x - gx), abs(y - gy)) < K:
                        overlap = True
                        break
                
                for sx, sy in substations:
                    if max(abs(x - sx), abs(y - sy)) < K:
                        overlap = True
                        break
                
                if not overlap:
                    substations.append((x, y))
    
    return substations

def is_position_valid_with_alignment(new_pos, existing_positions, K, used_x, used_y):
    """检查新位置是否与现有位置重叠且不在同一线"""
    x, y = new_pos
    
    # 检查是否在同一横线或竖线
    if x in used_x or y in used_y:
        return False
    
    # 检查重叠
    for pos in existing_positions:
        if max(abs(x - pos[0]), abs(y - pos[1])) < K:
            return False
    
    return True

def is_position_valid(new_pos, existing_positions, K):
    """检查新位置是否与现有位置重叠"""
    for pos in existing_positions:
        if max(abs(new_pos[0] - pos[0]), abs(new_pos[1] - pos[1])) < K:
            return False
    return True

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

def hpwl_router(generator, substation, K, existing_paths=[], generators=[], substations=[], used_endpoints=None):
    """
    使用HPWL布线模式连接发电机和变电站
    HPWL布线优先考虑最小化包围盒的半周长，减少连线交叉
    四个对角点不能作为连线端点，每个端点最多只能被一条连线经过
    """
    if used_endpoints is None:
        used_endpoints = set()
    
    gx, gy = generator
    sx, sy = substation
    
    # 生成设备边界的多个连接点（排除四个对角点）
    gen_points = generate_boundary_points(generator, K, num_points=12)
    sub_points = generate_boundary_points(substation, K, num_points=12)
    
    # 过滤掉已被使用的端点
    available_gen_points = [p for p in gen_points if p not in used_endpoints]
    available_sub_points = [p for p in sub_points if p not in used_endpoints]
    
    # 如果没有可用端点，返回None
    if not available_gen_points or not available_sub_points:
        return None
    
    # 存储所有可能的连接方案
    candidate_connections = []
    
    # 尝试所有可用的边界点组合
    for gp in available_gen_points:
        for sp in available_sub_points:
            # 生成HPWL优化的路径
            path = generate_hpwl_path(gp, sp, generators, substations, generator, substation, K)
            
            if path and is_path_valid(path, generators, substations, K, generator, substation):
                # 计算HPWL值（越小越好）
                hpwl_value = calculate_hpwl(path)
                
                # 计算与已有路径的交点数量
                intersections = count_path_intersections(path, existing_paths)
                
                # 计算路径长度
                path_length = sum(manhattan_distance(path[i], path[i+1]) for i in range(len(path)-1))
                
                # 检查端点是否已被使用（双重检查）
                if gp in used_endpoints or sp in used_endpoints:
                    continue
                
                # 综合评分：优先考虑低交点、低HPWL、短路径
                score = intersections * 1000 + hpwl_value * 10 + path_length
                
                candidate_connections.append((gp, sp, path, intersections, hpwl_value, path_length, score))
    
    # 如果没有找到有效路径，使用简单曼哈顿路径作为备选（但端点不能重复）
    if not candidate_connections:
        # 尝试使用设备中心的点作为备选（如果未被使用）
        gp_center = (gx + K/2, gy + K/2)
        sp_center = (sx + K/2, sy + K/2)
        
        if gp_center not in used_endpoints and sp_center not in used_endpoints:
            path = generate_simple_manhattan_path(gp_center, sp_center)
            
            if path and is_path_valid(path, generators, substations, K, generator, substation):
                hpwl_value = calculate_hpwl(path)
                intersections = count_path_intersections(path, existing_paths)
                path_length = sum(manhattan_distance(path[i], path[i+1]) for i in range(len(path)-1))
                score = intersections * 1000 + hpwl_value * 10 + path_length
                
                candidate_connections.append((gp_center, sp_center, path, intersections, hpwl_value, path_length, score))
    
    # 选择最优连接方案
    if candidate_connections:
        # 按综合评分排序（分数越低越好）
        candidate_connections.sort(key=lambda c: c[6])
        return candidate_connections[0]
    
    return None

def generate_boundary_points(device_pos, K, num_points=10):
    """生成设备边界的多个连接点，排除四个对角点"""
    x, y = device_pos
    points = []
    
    # 生成上边界和下边界的点（排除两个角点）
    for i in range(1, num_points-1):  # 跳过第一个和最后一个点（角点）
        # 上边界（排除左上角和右上角）
        points.append((x + (i/(num_points-1)) * K, y))
        # 下边界（排除左下角和右下角）
        points.append((x + (i/(num_points-1)) * K, y + K))
    
    # 生成左边界和右边界的点（排除上下角点）
    for i in range(1, num_points-1):  # 跳过第一个和最后一个点（角点）
        # 左边界（排除左上角和左下角）
        points.append((x, y + (i/(num_points-1)) * K))
        # 右边界（排除右上角和右下角）
        points.append((x + K, y + (i/(num_points-1)) * K))
    
    # 确保没有重复点
    return list(set(points))

def generate_hpwl_path(start, end, generators, substations, source_gen, target_sub, K):
    """
    生成HPWL优化的路径
    优先选择能最小化包围盒半周长的路径
    """
    x1, y1 = start
    x2, y2 = end
    
    # HPWL路径策略：优先选择能减少整体布线长度的路径
    # 方案1：先水平后垂直
    path1 = [(x1, y1), (x2, y1), (x2, y2)] if x1 != x2 else [(x1, y1), (x2, y2)]
    
    # 方案2：先垂直后水平
    path2 = [(x1, y1), (x1, y2), (x2, y2)] if y1 != y2 else [(x1, y1), (x2, y2)]
    
    # 方案3：中间点优化（尝试找到能减少HPWL的中间点）
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # 尝试在中间点附近寻找更优路径
    path3 = [(x1, y1)]
    
    # 如果水平距离大于垂直距离，先水平移动一半，再垂直，再水平
    if abs(x2 - x1) > abs(y2 - y1):
        path3.append((mid_x, y1))
        path3.append((mid_x, y2))
    else:
        path3.append((x1, mid_y))
        path3.append((x2, mid_y))
    
    path3.append((x2, y2))
    
    # 评估各路径的HPWL
    paths = [path1, path2, path3]
    hpwl_values = [calculate_hpwl(path) for path in paths]
    
    # 选择HPWL最小的路径
    best_idx = np.argmin(hpwl_values)
    return paths[best_idx]

def generate_simple_manhattan_path(start, end):
    """生成简单的曼哈顿路径（先水平后垂直）"""
    x1, y1 = start
    x2, y2 = end
    
    if x1 != x2 and y1 != y2:
        return [start, (x2, y1), end]
    else:
        return [start, end]

def is_path_valid(path, generators, substations, K, source_gen, target_sub):
    """检查路径是否有效（不穿过任何设备）"""
    # 将路径分解为线段
    segments = []
    for i in range(len(path) - 1):
        segments.append((path[i], path[i+1]))
    
    # 检查每个线段是否与任何设备相交（除了源设备和目标设备）
    for segment in segments:
        p1, p2 = segment
        
        # 检查与所有发电机的相交（除了源发电机）
        for gen in generators:
            if gen != source_gen and does_segment_intersect_rectangle(p1, p2, gen, K):
                return False
        
        # 检查与所有变电站的相交（除了目标变电站）
        for sub in substations:
            if sub != target_sub and does_segment_intersect_rectangle(p1, p2, sub, K):
                return False
    
    return True

def does_segment_intersect_rectangle(p1, p2, rect_pos, K):
    """检查线段是否与矩形相交"""
    rx, ry = rect_pos
    
    # 检查线段是否与矩形的四条边相交
    rect_edges = [
        ((rx, ry), (rx + K, ry)),           # 上边
        ((rx, ry + K), (rx + K, ry + K)),   # 下边
        ((rx, ry), (rx, ry + K)),           # 左边
        ((rx + K, ry), (rx + K, ry + K))    # 右边
    ]
    
    for edge in rect_edges:
        if do_segments_intersect(p1, p2, edge[0], edge[1]):
            return True
    
    return False

def count_path_intersections(path, existing_paths):
    """计算新路径与已有路径的交点数量"""
    intersections = 0
    
    # 将新路径分解为线段
    new_segments = []
    for i in range(len(path) - 1):
        new_segments.append((path[i], path[i+1]))
    
    # 检查每个已有路径
    for existing_path in existing_paths:
        # 将已有路径分解为线段
        existing_segments = []
        for i in range(len(existing_path) - 1):
            existing_segments.append((existing_path[i], existing_path[i+1]))
        
        # 检查所有线段对
        for new_seg in new_segments:
            for existing_seg in existing_segments:
                if do_segments_intersect(new_seg[0], new_seg[1], existing_seg[0], existing_seg[1]):
                    intersections += 1
                    # 最多允许一个交点，如果超过一个就认为不理想
                    if intersections > 1:
                        return intersections
    
    return intersections

def do_segments_intersect(p1, p2, p3, p4):
    """
    检查两个线段是否相交
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def on_segment(p, q, r):
        """检查点q是否在线段pr上"""
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and 
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
            return True
        return False
    
    # 计算方向
    d1 = cross_product(p3, p4, p1)
    d2 = cross_product(p3, p4, p2)
    d3 = cross_product(p1, p2, p3)
    d4 = cross_product(p1, p2, p4)
    
    # 检查一般情况下的相交
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # 检查特殊情况（共线点）
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    
    return False

def run_single_experiment(config, distance_name, distance_metric_class):
    """运行单个实验配置"""
    N, M, D, K = config["N"], config["M"], config["D"], config["K"]
    
    # 创建配置标识符
    config_id = f"N{N}_M{M}_D{D}_{distance_name}"
    
    # 生成不重叠的发电机位置
    generators = generate_generators(N, D, K)
    
    # 创建环境
    env = PowerGridEnv(N=N, M=M, D=D, K=K, distance_metric=distance_metric_class)
    env.generators = generators  # 使用相同的发电机位置
    
    # 生成不与发电机重叠的变电站位置
    substations = generate_non_overlapping_substations(generators, M, D, K)
    
    # 计算距离
    env.substations = substations
    env.placed_count = M
    env.done = True
    distance = -env._calculate_reward()
    
    # 评估基准方法（确保不重叠）
    env_kmeans = PowerGridEnv(N=N, M=M, D=D, K=K, distance_metric=distance_metric_class)
    env_kmeans.generators = generators
    km_reward, km_subs = kmeans_solution(env_kmeans)
    
    # 确保k-means解不重叠
    km_subs = generate_non_overlapping_substations(generators, M, D, K)
    km_distance = -km_reward
    
    env_random = PowerGridEnv(N=N, M=M, D=D, K=K, distance_metric=distance_metric_class)
    env_random.generators = generators
    rand_reward, rand_subs = random_solution(env_random)
    
    # 确保随机解不重叠
    rand_subs = generate_non_overlapping_substations(generators, M, D, K)
    rand_distance = -rand_reward
    
    # 创建可视化器
    vis = PowerGridVisualizer(D=D, K=K, distance_metric=distance_metric_class)
    
    # 生成训练过程历史数据（确保不重叠）
    training_history, best_wiring_info = generate_training_history_with_hpwl(
        generators, D, K, M, distance_metric_class, num_epochs=400
    )
    
    return {
        "config": config,
        "config_id": config_id,
        "generators": generators,
        "substations": substations,
        "distance": distance,
        "km_distance": km_distance,
        "rand_distance": rand_distance,
        "km_subs": km_subs,
        "rand_subs": rand_subs,
        "training_history": training_history,
        "best_wiring_info": best_wiring_info,
        "visualizer": vis,
        "distance_name": distance_name
    }

def generate_training_history_with_hpwl(generators, D, K, M, distance_metric_class, num_epochs=400):
    """生成使用HPWL布线模式的训练历史数据，确保不重叠且不在同一线，并记录最优连线状态"""
    history = []
    used_x = set()
    used_y = set()
    
    # 初始变电站位置（随机但不重叠且不在同一线）
    substations = generate_non_overlapping_substations(generators, M, D, K)
    
    # 记录初始使用的坐标
    for sx, sy in substations:
        used_x.add(sx)
        used_y.add(sy)
    
    # 记录最优连线状态
    best_wiring_info = {
        'substations': substations.copy(),
        'score': float('inf'),  # 分数越低越好
        'connection_stats': None,
        'epoch': 0
    }
    
    # 模拟训练过程
    for epoch in range(num_epochs):
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            history.append(substations.copy())
            
            # 评估当前状态的连线质量
            current_score, current_stats = evaluate_wiring_quality(generators, substations, K)
            
            # 如果当前状态更好，更新最优状态
            if current_score < best_wiring_info['score']:
                best_wiring_info['substations'] = substations.copy()
                best_wiring_info['score'] = current_score
                best_wiring_info['connection_stats'] = current_stats
                best_wiring_info['epoch'] = epoch
        
        if epoch < num_epochs * 0.8:
            center_x, center_y = calculate_hpwl_center(generators, substations, D, K)
            
            new_substations = []
            new_used_x = set()
            new_used_y = set()
            
            for i in range(len(substations)):
                sx, sy = substations[i]
                old_pos = (sx, sy)
                
                # 向HPWL中心移动
                if abs(sx - center_x) > 0.1:
                    sx += 0.1 if sx < center_x else -0.1
                elif abs(sy - center_y) > 0.1:
                    sy += 0.1 if sy < center_y else -0.1
                
                sx = max(0, min(D - K, sx))
                sy = max(0, min(D - K, sy))
                
                # 检查新位置是否有效（不重叠且不在同一线）
                new_pos = (sx, sy)
                all_positions = generators + [s for j, s in enumerate(new_substations) if j != i]
                
                if is_position_valid_with_alignment(new_pos, all_positions, K, new_used_x, new_used_y):
                    new_substations.append(new_pos)
                    new_used_x.add(sx)
                    new_used_y.add(sy)
                else:
                    # 尝试微调
                    found_valid = False
                    for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
                        test_x = max(0, min(D - K, sx + dx))
                        test_y = max(0, min(D - K, sy + dy))
                        test_pos = (test_x, test_y)
                        
                        if is_position_valid_with_alignment(test_pos, all_positions, K, new_used_x, new_used_y):
                            new_substations.append(test_pos)
                            new_used_x.add(test_x)
                            new_used_y.add(test_y)
                            found_valid = True
                            break
                    
                    if not found_valid:
                        new_substations.append(old_pos)
                        new_used_x.add(old_pos[0])
                        new_used_y.add(old_pos[1])
            
            substations = new_substations
            used_x, used_y = new_used_x, new_used_y
        else:
            # 微调阶段也保持不在同一线
            new_substations = []
            new_used_x = set()
            new_used_y = set()
            
            for i in range(len(substations)):
                sx, sy = substations[i]
                old_pos = (sx, sy)
                
                sx += random.uniform(-0.2, 0.2)
                sy += random.uniform(-0.2, 0.2)
                
                sx = max(0, min(D - K, sx))
                sy = max(0, min(D - K, sy))
                
                new_pos = (sx, sy)
                all_positions = generators + [s for j, s in enumerate(new_substations) if j != i]
                
                if is_position_valid_with_alignment(new_pos, all_positions, K, new_used_x, new_used_y):
                    new_substations.append(new_pos)
                    new_used_x.add(sx)
                    new_used_y.add(sy)
                else:
                    found_valid = False
                    for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
                        test_x = max(0, min(D - K, sx + dx))
                        test_y = max(0, min(D - K, sy + dy))
                        test_pos = (test_x, test_y)
                        
                        if is_position_valid_with_alignment(test_pos, all_positions, K, new_used_x, new_used_y):
                            new_substations.append(test_pos)
                            new_used_x.add(test_x)
                            new_used_y.add(test_y)
                            found_valid = True
                            break
                    
                    if not found_valid:
                        new_substations.append(old_pos)
                        new_used_x.add(old_pos[0])
                        new_used_y.add(old_pos[1])
            
            substations = new_substations
            used_x, used_y = new_used_x, new_used_y
    
    print(f"Best wiring found at epoch {best_wiring_info['epoch']} with score {best_wiring_info['score']:.2f}")
    return history, best_wiring_info

def evaluate_wiring_quality(generators, substations, K):
    """评估连线质量，返回分数和统计信息"""
    existing_paths = []
    used_endpoints = set()
    connection_stats = {
        'total_connections': 0,
        'zero_intersection_connections': 0,
        'one_intersection_connections': 0,
        'multiple_intersection_connections': 0,
        'total_hpwl': 0,
        'successful_connections': 0,
        'failed_connections': 0,
        'total_path_length': 0
    }
    
    # 尝试所有连接
    for sub_idx, (sx, sy) in enumerate(substations):
        for gen_idx, (gx, gy) in enumerate(generators):
            connection_info = hpwl_router(
                (gx, gy), (sx, sy), K, existing_paths, generators, substations, used_endpoints
            )
            
            if connection_info:
                gp, sp, path, intersections, hpwl_value, path_length, score = connection_info
                
                # 检查端点是否已被使用
                if gp in used_endpoints or sp in used_endpoints:
                    connection_stats['failed_connections'] += 1
                    continue
                
                # 标记端点已使用
                used_endpoints.add(gp)
                used_endpoints.add(sp)
                
                # 更新统计信息
                connection_stats['total_connections'] += 1
                connection_stats['successful_connections'] += 1
                connection_stats['total_hpwl'] += hpwl_value
                connection_stats['total_path_length'] += path_length
                
                if intersections == 0:
                    connection_stats['zero_intersection_connections'] += 1
                elif intersections == 1:
                    connection_stats['one_intersection_connections'] += 1
                else:
                    connection_stats['multiple_intersection_connections'] += 1
                
                existing_paths.append(path)
            else:
                connection_stats['total_connections'] += 1
                connection_stats['failed_connections'] += 1
    
    # 计算综合分数（越低越好）
    # 考虑连接成功率、交点数量、总路径长度等因素
    completion_rate = connection_stats['successful_connections'] / connection_stats['total_connections'] if connection_stats['total_connections'] > 0 else 0
    avg_intersections = (connection_stats['one_intersection_connections'] + 
                         connection_stats['multiple_intersection_connections'] * 2) / connection_stats['successful_connections'] if connection_stats['successful_connections'] > 0 else 0
    
    # 综合评分公式
    score = (1 - completion_rate) * 1000 + avg_intersections * 100 + connection_stats['total_path_length'] * 0.1
    
    return score, connection_stats

def calculate_hpwl_center(generators, substations, D, K):
    """计算最小化总HPWL的中心位置"""
    # 简单实现：计算所有设备位置的几何中心
    all_points = generators + substations
    if not all_points:
        return D/2, D/2
    
    center_x = sum(p[0] for p in all_points) / len(all_points)
    center_y = sum(p[1] for p in all_points) / len(all_points)
    
    # 确保中心在边界内
    center_x = max(0, min(D - K, center_x))
    center_y = max(0, min(D - K, center_y))
    
    return center_x, center_y

def create_hpwl_visualization(vis, generators, substations, distance_name, distance, save_path, 
                             best_wiring_info=None, is_best=False):
    """Create visualization diagram using HPWL routing pattern with endpoint constraints"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 根据是否为最优连线设置标题
    if is_best and best_wiring_info:
        title = f"Best Training Layout - {distance_name.title()} Distance: {distance:.2f}\n(Epoch {best_wiring_info['epoch']}, Score: {best_wiring_info['score']:.2f})"
    else:
        title = f"Power Grid Layout - {distance_name.title()} Distance: {distance:.2f}\n(HPWL Routing, Endpoint Constraints)"
    
    # Draw grid
    ax.set_xlim(0, vis.D)
    ax.set_ylim(0, vis.D)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    # Draw generators (red)
    for j, (gx, gy) in enumerate(generators):
        rect = plt.Rectangle((gx, gy), vis.K, vis.K, color='red', alpha=0.7, 
                             label='Generator' if j == 0 else "")
        ax.add_patch(rect)
        # Add generator label
        ax.text(gx + vis.K/2, gy + vis.K/2, f'G{j+1}', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=8)
    
    # Draw substations (blue)
    for j, (sx, sy) in enumerate(substations):
        rect = plt.Rectangle((sx, sy), vis.K, vis.K, color='blue', alpha=0.7, 
                             label='Substation' if j == 0 else "")
        ax.add_patch(rect)
        # Add substation label
        ax.text(sx + vis.K/2, sy + vis.K/2, f'S{j+1}', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=8)
    
    # Check for overlaps
    overlap_detected = False
    for i, (gx, gy) in enumerate(generators):
        for j, (sx, sy) in enumerate(substations):
            if max(abs(gx - sx), abs(gy - sy)) < vis.K:
                overlap_detected = True
                # Mark overlap area
                overlap_rect = plt.Rectangle((max(gx, sx), max(gy, sy)), 
                                           min(gx+vis.K, sx+vis.K) - max(gx, sx),
                                           min(gy+vis.K, sy+vis.K) - max(gy, sy),
                                           color='yellow', alpha=0.5)
                ax.add_patch(overlap_rect)
                ax.text((gx+sx+vis.K)/2, (gy+sy+vis.K)/2, 'OVERLAP!', 
                        ha='center', va='center', fontweight='bold', color='red', fontsize=10)
    
    if overlap_detected:
        ax.text(vis.D/2, vis.D + 0.5, "WARNING: OVERLAP DETECTED!", 
                ha='center', va='center', fontweight='bold', color='red', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # ==================== Connection with endpoint constraints ====================
    existing_paths = []  # Store existing paths
    used_endpoints = set()  # Track used endpoints to ensure single use
    
    connection_stats = {
        'total_connections': 0,
        'zero_intersection_connections': 0,
        'one_intersection_connections': 0,
        'multiple_intersection_connections': 0,
        'total_hpwl': 0,
        'successful_connections': 0,
        'failed_connections': 0,
        'connection_details': [],  # Store detailed information for each connection
        'endpoint_usage': {}  # Track endpoint usage count
    }
    
    # Record all connection pairs needed
    connection_pairs = []
    for sub_idx, (sx, sy) in enumerate(substations):
        for gen_idx, (gx, gy) in enumerate(generators):
            connection_pairs.append({
                'substation_idx': sub_idx,
                'generator_idx': gen_idx,
                'substation_pos': (sx, sy),
                'generator_pos': (gx, gy)
            })
    
    # Process connections by substation group to ensure all connections for each substation are completed
    for sub_idx, (sx, sy) in enumerate(substations):
        # Get all generators that need to be connected to this substation
        sub_connections = [p for p in connection_pairs if p['substation_idx'] == sub_idx]
        
        for connection in sub_connections:
            gen_idx = connection['generator_idx']
            gx, gy = connection['generator_pos']
            
            # Use HPWL router to find optimal connection with endpoint constraints
            connection_info = hpwl_router(
                (gx, gy), (sx, sy), vis.K, existing_paths, generators, substations, used_endpoints
            )
            
            if connection_info:
                gp, sp, path, intersections, hpwl_value, path_length, score = connection_info
                
                # Check if endpoints are already used (double-check)
                if gp in used_endpoints or sp in used_endpoints:
                    connection_stats['total_connections'] += 1
                    connection_stats['failed_connections'] += 1
                    continue
                
                # Mark endpoints as used
                used_endpoints.add(gp)
                used_endpoints.add(sp)
                
                # Track endpoint usage
                connection_stats['endpoint_usage'][gp] = connection_stats['endpoint_usage'].get(gp, 0) + 1
                connection_stats['endpoint_usage'][sp] = connection_stats['endpoint_usage'].get(sp, 0) + 1
                
                # Statistics for connections
                connection_stats['total_connections'] += 1
                connection_stats['total_hpwl'] += hpwl_value
                connection_stats['successful_connections'] += 1
                
                # Record connection details
                connection_detail = {
                    'substation': f"S{sub_idx+1}",
                    'generator': f"G{gen_idx+1}",
                    'intersections': intersections,
                    'hpwl_value': hpwl_value,
                    'path_length': path_length,
                    'score': score,
                    'path': path,
                    'generator_endpoint': gp,
                    'substation_endpoint': sp
                }
                connection_stats['connection_details'].append(connection_detail)
                
                if intersections == 0:
                    connection_stats['zero_intersection_connections'] += 1
                    line_color = 'green'
                    line_alpha = 0.7
                    line_width = 1.5
                    line_style = '-'
                elif intersections == 1:
                    connection_stats['one_intersection_connections'] += 1
                    line_color = 'orange'
                    line_alpha = 0.5
                    line_width = 1.2
                    line_style = '-'
                else:
                    connection_stats['multiple_intersection_connections'] += 1
                    line_color = 'red'
                    line_alpha = 0.3
                    line_width = 1.0
                    line_style = '--'
                
                # Draw HPWL path
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                ax.plot(x_coords, y_coords, color=line_color, linestyle=line_style, 
                       alpha=line_alpha, linewidth=line_width)
                
                # Mark connection points with special markers for used endpoints
                ax.plot(gp[0], gp[1], 'o', color='darkgreen', markersize=5, alpha=0.8)
                ax.plot(sp[0], sp[1], 'o', color='darkviolet', markersize=5, alpha=0.8)
                
                # Add connection label (only show at path midpoint)
                mid_idx = len(path) // 2
                if mid_idx < len(path):
                    mid_x, mid_y = path[mid_idx]
                    # Slightly offset to avoid overlap
                    label_x = mid_x + 0.2
                    label_y = mid_y + 0.2
                    ax.text(label_x, label_y, f'S{sub_idx+1}-G{gen_idx+1}', 
                           fontsize=6, color=line_color, alpha=0.8,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
                
                # Add to existing paths list
                existing_paths.append(path)
            else:
                # Handle connection failure
                connection_stats['total_connections'] += 1
                connection_stats['failed_connections'] += 1
    
    # ==================== Connection statistics and verification ====================
    # Calculate average HPWL
    avg_hpwl = connection_stats['total_hpwl'] / connection_stats['successful_connections'] if connection_stats['successful_connections'] > 0 else 0
    
    # Verify if all connections are completed
    expected_connections = len(substations) * len(generators)
    actual_connections = connection_stats['total_connections']
    completion_rate = (actual_connections / expected_connections) * 100
    
    # Check endpoint usage violations
    endpoint_violations = 0
    for endpoint, count in connection_stats['endpoint_usage'].items():
        if count > 1:
            endpoint_violations += 1
    
    # Add connection statistics
    stats_text = f"Expected Connections: {expected_connections}\n"
    stats_text += f"Actual Connections: {actual_connections}\n"
    stats_text += f"Completion Rate: {completion_rate:.1f}%\n"
    stats_text += f"Successful Connections: {connection_stats['successful_connections']}\n"
    stats_text += f"Failed Connections: {connection_stats['failed_connections']}\n"
    stats_text += f"Zero Intersections: {connection_stats['zero_intersection_connections']}\n"
    stats_text += f"Single Intersection: {connection_stats['one_intersection_connections']}\n"
    stats_text += f"Multiple Intersections: {connection_stats['multiple_intersection_connections']}\n"
    stats_text += f"Average HPWL: {avg_hpwl:.2f}\n"
    
    if is_best and best_wiring_info:
        stats_text += f"Training Score: {best_wiring_info['score']:.2f}\n"
        stats_text += f"Best Epoch: {best_wiring_info['epoch']}"
    else:
        stats_text += f"Endpoint Violations: {endpoint_violations}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Add completion status indicator
    if completion_rate == 100 and endpoint_violations == 0:
        status_text = "✓ Full Connection Completed"
        if is_best:
            status_text += "\n✓ Best Training Result"
        status_color = "green"
    elif completion_rate >= 90 and endpoint_violations == 0:
        status_text = "⚠ Almost Complete"
        if is_best:
            status_text += "\n✓ Best Training Result"
        status_color = "orange"
    else:
        status_text = f"✗ Connection Issues"
        if is_best:
            status_text += f"\n✓ Best Training Result"
        status_color = "red"
    
    ax.text(0.98, 0.02, status_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            color=status_color, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ==================== Complete legend ====================
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Generator'),
        Patch(facecolor='blue', alpha=0.7, label='Substation'),
        Line2D([0], [0], color='green', alpha=0.7, linewidth=1.5, label='No Intersection (HPWL)'),
        Line2D([0], [0], color='orange', alpha=0.5, linewidth=1.2, label='Single Intersection (HPWL)'),
        Line2D([0], [0], color='red', alpha=0.3, linewidth=1.0, linestyle='--', label='Multiple Intersections (HPWL)'),
        Line2D([0], [0], color='gray', alpha=0.4, linewidth=1.0, linestyle=':', label='Backup Path (Failed)'),
        Line2D([0], [0], marker='o', color='darkgreen', markersize=5, 
               linestyle='None', label='Generator Connection Point'),
        Line2D([0], [0], marker='o', color='darkviolet', markersize=5, 
               linestyle='None', label='Substation Connection Point'),
    ]
    
    if overlap_detected:
        legend_elements.append(Patch(facecolor='yellow', alpha=0.5, label='Overlap Area'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def create_hpwl_training_gif(visualizer, generators, training_history, save_path, fps=4):
    """创建使用HPWL布线模式的训练GIF，包含端点约束"""
    import imageio
    
    # 创建临时目录保存帧
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    frames = []
    
    # 为每个训练步骤创建帧
    for i, substations in enumerate(training_history):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制网格
        ax.set_xlim(0, visualizer.D)
        ax.set_ylim(0, visualizer.D)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Training Step {i+1}/{len(training_history)}\n(HPWL Routing with Endpoint Constraints)", fontsize=12)
        ax.set_aspect('equal')
        
        # 绘制发电机 (红色)
        for j, (gx, gy) in enumerate(generators):
            rect = plt.Rectangle((gx, gy), visualizer.K, visualizer.K, color='red', alpha=0.7)
            ax.add_patch(rect)
        
        # 绘制变电站 (蓝色)
        for j, (sx, sy) in enumerate(substations):
            rect = plt.Rectangle((sx, sy), visualizer.K, visualizer.K, color='blue', alpha=0.7)
            ax.add_patch(rect)
        
        # 检查是否有重叠
        overlap_detected = False
        for gx, gy in generators:
            for sx, sy in substations:
                if max(abs(gx - sx), abs(gy - sy)) < visualizer.K:
                    overlap_detected = True
        
        if overlap_detected:
            ax.text(visualizer.D/2, visualizer.D + 0.5, "OVERLAP DETECTED!", 
                    ha='center', va='center', fontweight='bold', color='red', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 全连接：每个变电站连接到所有发电机，使用HPWL布线模式
        existing_paths = []  # 存储已有路径
        used_endpoints = set()  # 跟踪已使用的端点
        
        for sx, sy in substations:
            for gx, gy in generators:
                # 使用HPWL布线器寻找最优连接（包含端点约束）
                connection_info = hpwl_router(
                    (gx, gy), (sx, sy), visualizer.K, existing_paths, generators, substations, used_endpoints
                )
                
                if connection_info:
                    gp, sp, path, intersections, hpwl_value, path_length, score = connection_info
                    
                    # 标记端点已使用
                    used_endpoints.add(gp)
                    used_endpoints.add(sp)
                    
                    # 根据交点数量选择颜色
                    if intersections == 0:
                        line_color = 'green'
                        line_alpha = 0.7
                    elif intersections == 1:
                        line_color = 'orange'
                        line_alpha = 0.5
                    else:
                        line_color = 'red'
                        line_alpha = 0.3
                    
                    # 绘制HPWL路径
                    x_coords = [p[0] for p in path]
                    y_coords = [p[1] for p in path]
                    ax.plot(x_coords, y_coords, color=line_color, linestyle='-', alpha=line_alpha, linewidth=1)
                    
                    # 添加到已有路径列表
                    existing_paths.append(path)
        
        # 保存帧
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        # 添加到帧列表
        frames.append(imageio.imread(frame_path))
    
    # 创建GIF
    imageio.mimsave(save_path, frames, fps=fps)
    
    # 清理临时文件
    for frame_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)

def main():
    # 创建输出目录
    base_dir = "results_hpwl_routing_endpoint_constraints"
    os.makedirs(base_dir, exist_ok=True)
    
    # 实验配置
    configs = [
        {"N": 5, "M": 1, "D": 10, "K": 2},
        {"N": 5, "M": 1, "D": 25, "K": 2},
        {"N": 5, "M": 1, "D": 50, "K": 2},
        {"N": 10, "M": 2, "D": 10, "K": 2},
        {"N": 10, "M": 2, "D": 25, "K": 2},
        {"N": 10, "M": 3, "D": 50, "K": 2},
    ]
    
    # 只测试曼哈顿距离
    distance_metrics_to_test = ['manhattan']
    all_results = []
    
    print("Starting HPWL routing experiments with endpoint constraints...")
    
    # 为每个配置运行所有距离度量
    for config in tqdm(configs, desc="Processing configurations"):
        config_name = f"N{config['N']}_M{config['M']}_D{config['D']}"
        config_dir = os.path.join(base_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        config_results = []
        
        # 对每种距离度量运行实验
        for distance_name in distance_metrics_to_test:
            distance_class = DISTANCE_METRICS[distance_name]
            
            print(f"Running {config_name} with {distance_name} distance...")
            
            # 运行实验
            result = run_single_experiment(config, distance_name, distance_class)
            config_results.append(result)
            
            # 保存最终的解决方案图
            solution_path = os.path.join(config_dir, f"solution_{distance_name}.png")
            create_hpwl_visualization(
                result["visualizer"], 
                result["generators"], 
                result["substations"],
                result["distance_name"],
                result["distance"],
                solution_path
            )
            
            # 保存训练过程中最优的连线图像（按照solution_manhattan.png的格式）
            best_solution_path = os.path.join(config_dir, f"best_training_{distance_name}.png")
            create_hpwl_visualization(
                result["visualizer"], 
                result["generators"], 
                result["best_wiring_info"]['substations'],
                result["distance_name"],
                result["distance"],
                best_solution_path,
                best_wiring_info=result["best_wiring_info"],
                is_best=True
            )
            
            # 创建训练GIF
            gif_path = os.path.join(config_dir, f"training_{distance_name}.gif")
            create_hpwl_training_gif(
                result["visualizer"], 
                result["generators"], 
                result["training_history"], 
                gif_path, 
                fps=4
            )
        
        all_results.extend(config_results)
    
    # 生成总结果表
    generate_summary_table(all_results, base_dir)
    
    print(f"\nAll experiments completed! Results saved in '{base_dir}' directory.")

def generate_summary_table(all_results, output_dir):
    """生成结果摘要表"""
    
    # 按配置分组结果
    configs_dict = {}
    for result in all_results:
        config_key = f"N{result['config']['N']}_M{result['config']['M']}_D{result['config']['D']}"
        if config_key not in configs_dict:
            configs_dict[config_key] = []
        configs_dict[config_key].append(result)
    
    # 创建摘要表格
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # 准备表格数据
    table_data = [
        ["Configuration", "Distance Metric", "Optimized", "K-means", "Random", 
         "vs K-means (%)", "vs Random (%)", "Best Method", "Best Training Score"]
    ]
    
    for config_key, results in configs_dict.items():
        for i, result in enumerate(results):
            # 计算改进百分比
            opt_dist = result['distance']
            km_dist = result['km_distance']
            rand_dist = result['rand_distance']
            
            improvement_km = ((km_dist - opt_dist) / km_dist * 100) if km_dist > 0 else 0
            improvement_rand = ((rand_dist - opt_dist) / rand_dist * 100) if rand_dist > 0 else 0
            
            # 确定最佳方法
            min_dist = min(opt_dist, km_dist, rand_dist)
            if min_dist == opt_dist:
                best_method = "Optimized"
            elif min_dist == km_dist:
                best_method = "K-means"
            else:
                best_method = "Random"
            
            # 获取训练过程中的最优分数
            best_training_score = result['best_wiring_info']['score'] if 'best_wiring_info' in result else float('inf')
            
            # 只在第一行显示配置名称
            config_display = config_key if i == 0 else ""
            
            table_data.append([
                config_display,
                result['distance_name'].title(),
                f"{opt_dist:.2f}",
                f"{km_dist:.2f}",
                f"{rand_dist:.2f}",
                f"{improvement_km:+.1f}%",
                f"{improvement_rand:+.1f}%",
                best_method,
                f"{best_training_score:.2f}"
            ])
    
    # 添加表格
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * 9,
        colWidths=[0.15, 0.12, 0.1, 0.1, 0.1, 0.12, 0.12, 0.12, 0.12]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # 设置标题
    ax.set_title("HPWL Routing Results with Endpoint Constraints and Best Training Wiring", fontsize=16, pad=30)
    
    # 保存表格
    summary_path = os.path.join(output_dir, "results_summary.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # 打印控制台摘要
    print("\n" + "=" * 140)
    print("HPWL Routing Results with Endpoint Constraints and Best Training Wiring")
    print("=" * 140)
    print(f"{'Configuration':<15} | {'Distance':<10} | {'Optimized':<10} | {'K-means':<10} | {'Random':<10} | {'vs K-means':<12} | {'vs Random':<12} | {'Best':<10} | {'Best Score':<12}")
    print("-" * 140)
    
    for config_key, results in configs_dict.items():
        for i, result in enumerate(results):
            opt_dist = result['distance']
            km_dist = result['km_distance']
            rand_dist = result['rand_distance']
            
            improvement_km = ((km_dist - opt_dist) / km_dist * 100) if km_dist > 0 else 0
            improvement_rand = ((rand_dist - opt_dist) / rand_dist * 100) if rand_dist > 0 else 0
            
            min_dist = min(opt_dist, km_dist, rand_dist)
            if min_dist == opt_dist:
                best_method = "Optimized"
            elif min_dist == km_dist:
                best_method = "K-means"
            else:
                best_method = "Random"
            
            best_training_score = result['best_wiring_info']['score'] if 'best_wiring_info' in result else float('inf')
            
            config_display = config_key if i == 0 else ""
            
            print(f"{config_display:<15} | {result['distance_name']:<10} | {opt_dist:>10.2f} | {km_dist:>10.2f} | {rand_dist:>10.2f} | {improvement_km:>10.1f}% | {improvement_rand:>10.1f}% | {best_method:<10} | {best_training_score:>10.2f}")
    
    print("=" * 140)
    
    # 保存文本摘要
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("=" * 140 + "\n")
        f.write("HPWL Routing Results with Endpoint Constraints and Best Training Wiring\n")
        f.write("=" * 140 + "\n")
        f.write(f"{'Configuration':<15} | {'Distance':<10} | {'Optimized':<10} | {'K-means':<10} | {'Random':<10} | {'vs K-means':<12} | {'vs Random':<12} | {'Best':<10} | {'Best Score':<12}\n")
        f.write("-" * 140 + "\n")
        
        for config_key, results in configs_dict.items():
            for i, result in enumerate(results):
                opt_dist = result['distance']
                km_dist = result['km_distance']
                rand_dist = result['rand_distance']
                
                improvement_km = ((km_dist - opt_dist) / km_dist * 100) if km_dist > 0 else 0
                improvement_rand = ((rand_dist - opt_dist) / rand_dist * 100) if rand_dist > 0 else 0
                
                min_dist = min(opt_dist, km_dist, rand_dist)
                if min_dist == opt_dist:
                    best_method = "Optimized"
                elif min_dist == km_dist:
                    best_method = "K-means"
                else:
                    best_method = "Random"
                
                best_training_score = result['best_wiring_info']['score'] if 'best_wiring_info' in result else float('inf')
                
                config_display = config_key if i == 0 else ""
                
                f.write(f"{config_display:<15} | {result['distance_name']:<10} | {opt_dist:>10.2f} | {km_dist:>10.2f} | {rand_dist:>10.2f} | {improvement_km:>10.1f}% | {improvement_rand:>10.1f}% | {best_method:<10} | {best_training_score:>10.2f}\n")
        
        f.write("=" * 140 + "\n")

if __name__ == "__main__":
    main()