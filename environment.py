import torch
import numpy as np
import random
from distance_metrics import EuclideanDistance

class PowerGridEnv:
    def __init__(self, N=5, M=1, D=50, K=2, distance_metric=None):
        """
        N: 发电机数量
        M: 变电站数量
        D: 网格边长
        K: 设备尺寸 (KxK)
        distance_metric: 距离度量类
        """
        self.N = N
        self.M = M
        self.D = D
        self.K = K
        self.distance_metric = distance_metric or EuclideanDistance
        
        # 生成发电机位置
        self.generators = self._generate_points(N)
        
        # 初始化状态
        self.reset()
    
    def _generate_points(self, num_points):
        """生成不重叠的点位置"""
        points = []
        while len(points) < num_points:
            x = random.randint(0, self.D - self.K)
            y = random.randint(0, self.D - self.K)
            new_point = (x, y)
            
            # 检查是否与现有点重叠
            overlap = False
            for px, py in points:
                if (abs(x - px) < self.K and abs(y - py) < self.K):
                    overlap = True
                    break
            
            if not overlap:
                points.append(new_point)
        
        return points
    
    def reset(self):
        """重置环境"""
        self.substations = []
        self.placed_count = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态表示"""
        # 创建网格状态: [发电机存在, 变电站存在, 占用标记]
        state_grid = np.zeros((self.D, self.D, 3), dtype=np.float32)
        
        # 标记发电机位置
        for x, y in self.generators:
            for i in range(self.K):
                for j in range(self.K):
                    if x+i < self.D and y+j < self.D:
                        state_grid[x+i, y+j, 0] = 1.0
        
        # 标记已放置变电站
        for x, y in self.substations:
            for i in range(self.K):
                for j in range(self.K):
                    if x+i < self.D and y+j < self.D:
                        state_grid[x+i, y+j, 1] = 1.0
        
        # 标记占用位置 (发电机和变电站都占用)
        state_grid[:, :, 2] = np.logical_or(state_grid[:, :, 0], state_grid[:, :, 1]).astype(np.float32)
        
        return state_grid
    
    def get_action_mask(self):
        """获取有效动作掩码"""
        mask = np.ones((self.D - self.K + 1, self.D - self.K + 1), dtype=bool)
        
        # 检查每个可能位置
        for x in range(self.D - self.K + 1):
            for y in range(self.D - self.K + 1):
                # 检查是否与现有设备重叠
                for dx in range(self.K):
                    for dy in range(self.K):
                        if x+dx < self.D and y+dy < self.D:
                            # 如果位置已被占用
                            if self._get_state()[x+dx, y+dy, 2] == 1.0:
                                mask[x, y] = False
                                break
                    else:
                        continue
                    break
        
        return mask.flatten()
    
    def step(self, action):
        """执行动作"""
        # 将一维动作转换为二维坐标
        action_space_size = self.D - self.K + 1
        x = action // action_space_size
        y = action % action_space_size
        
        # 放置变电站
        self.substations.append((x, y))
        self.placed_count += 1
        
        # 检查是否完成
        if self.placed_count >= self.M:
            self.done = True
            reward = self._calculate_reward()
        else:
            reward = 0.0
        
        return self._get_state(), reward, self.done
    
    def _calculate_reward(self):
        """计算最终奖励（负的总距离）"""
        total_distance = 0.0
        
        # 计算每个发电机到最近变电站的距离
        for gx, gy in self.generators:
            min_dist = float('inf')
            for sx, sy in self.substations:
                # 使用指定的距离度量
                dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
            total_distance += min_dist
        
        # 返回负距离作为奖励（最大化奖励相当于最小化距离）
        return -total_distance