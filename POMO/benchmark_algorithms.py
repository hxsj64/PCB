import numpy as np
import random
from sklearn.cluster import KMeans

def kmeans_solution(env):
    """K-means 基准方法"""
    # 提取发电机位置
    generator_points = np.array(env.generators)
    
    # 应用K-means聚类
    kmeans = KMeans(n_clusters=env.M, random_state=123)
    kmeans.fit(generator_points)
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_.astype(int)
    
    # 将中心作为变电站位置
    env.substations = [(max(0, min(env.D - env.K, int(x))), 
                       max(0, min(env.D - env.K, int(y)))) 
                      for x, y in centers]
    env.placed_count = env.M
    env.done = True
    
    # 计算奖励
    reward = env._calculate_reward()
    return reward, env.substations

def random_solution(env):
    """随机基准方法"""
    env.reset()
    
    # 随机放置变电站
    for _ in range(env.M):
        valid_positions = []
        
        # 找到所有有效位置
        for x in range(env.D - env.K + 1):
            for y in range(env.D - env.K + 1):
                valid = True
                # 检查是否与现有设备重叠
                for dx in range(env.K):
                    for dy in range(env.K):
                        if x+dx < env.D and y+dy < env.D:
                            if env._get_state()[x+dx, y+dy, 2] == 1.0:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    valid_positions.append((x, y))
        
        # 随机选择一个位置
        if valid_positions:
            x, y = random.choice(valid_positions)
            env.substations.append((x, y))
            env.placed_count += 1
    
    env.done = True
    reward = env._calculate_reward()
    return reward, env.substations

def generate_optimized_substations(generators, M, D, K, distance_metric):
    """生成优化的变电站位置（模拟优化结果）"""
    substations = []
    
    # 计算发电机中心
    center_x = sum([x + K/2 for x, y in generators]) / len(generators)
    center_y = sum([y + K/2 for x, y in generators]) / len(generators)
    
    if M == 1:
        # 对于单个变电站，选择中心位置
        substations.append((max(0, min(D - K, int(center_x - K/2))), 
                           max(0, min(D - K, int(center_y - K/2)))))
    else:
        # 对于多个变电站，使用聚类思想
        # 根据发电机位置聚类
        points = np.array([[x + K/2, y + K/2] for x, y in generators])
        
        # 使用简化的聚类方法
        cluster_centers = []
        if M == 2:
            # 将区域分为左右两部分
            left_center = (center_x * 0.7, center_y)
            right_center = (center_x * 1.3, center_y)
            cluster_centers = [left_center, right_center]
        else:
            # 将区域分为四个象限
            for i in range(M):
                angle = 2 * np.pi * i / M
                offset_x = 0.3 * center_x * np.cos(angle)
                offset_y = 0.3 * center_y * np.sin(angle)
                cluster_centers.append((center_x + offset_x, center_y + offset_y))
        
        for center in cluster_centers:
            x, y = center
            substations.append((max(0, min(D - K, int(x - K/2))), 
                               max(0, min(D - K, int(y - K/2)))))
    
    return substations

def generate_training_history(generators, D, K, M, distance_metric, num_epochs=400):
    """生成训练过程历史数据（模拟优化过程）"""
    history = []
    
    # 初始随机位置
    substations = []
    for _ in range(M):
        x = random.randint(0, D - K)
        y = random.randint(0, D - K)
        substations.append((x, y))
    history.append((0, substations.copy()))
    
    # 计算优化位置
    optimized_positions = generate_optimized_substations(generators, M, D, K, distance_metric)
    
    # 模拟优化过程
    for epoch in range(1, num_epochs + 1):
        # 每10个epoch保存一次状态
        if epoch % 10 == 0 or epoch == num_epochs:
            # 模拟位置改进 - 逐步向优化位置移动
            new_substations = []
            
            for i, (x, y) in enumerate(substations):
                # 目标位置
                target_x, target_y = optimized_positions[i]
                
                # 计算移动比例（随epoch增加而增加）
                progress = min(1.0, epoch / num_epochs)
                
                # 线性插值
                new_x = x + (target_x - x) * progress * 0.8
                new_y = y + (target_y - y) * progress * 0.8
                
                # 添加随机扰动（随epoch减少）
                if epoch < num_epochs * 0.8:  # 前期允许更多探索
                    new_x += random.uniform(-1.5, 1.5)
                    new_y += random.uniform(-1.5, 1.5)
                
                new_x = max(0, min(D - K, int(new_x)))
                new_y = max(0, min(D - K, int(new_y)))
                new_substations.append((new_x, new_y))
            
            substations = new_substations
            history.append((epoch, substations.copy()))
    
    return history