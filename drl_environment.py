"""
深度强化学习环境 - 修复初始化问题
"""
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class PowerGridRLEnv:
    def __init__(self, N=5, M=1, D=50, K=2, distance_metric=None, device='cpu'):
        """
        深度强化学习环境
        Args:
            N: 发电机数量
            M: 变电站数量  
            D: 网格边长
            K: 设备尺寸
            distance_metric: 距离度量函数
            device: 计算设备
        """
        self.N = N
        self.M = M
        self.D = D
        self.K = K
        self.distance_metric = distance_metric
        self.device = device
        
        # 状态维度: [generator_map, substation_map, occupancy_map, available_positions]
        self.state_dim = 4
        
        # 动作空间: 可放置变电站的位置数量
        self.action_space_size = (D - K + 1) ** 2
        
        # 初始化发电机位置
        self.generator_positions = self._generate_generators()
        
        # 初始化其他属性
        self.substation_positions = []
        self.placed_count = 0
        self.done = False
        self.step_count = 0
        self.state = None  # 先初始化为None
        
        # 重置环境
        self.reset()
    
    def _generate_generators(self):
        """生成固定的发电机位置"""
        generators = []
        max_attempts = 1000
        attempts = 0
        
        while len(generators) < self.N and attempts < max_attempts:
            x = random.randint(0, self.D - self.K)
            y = random.randint(0, self.D - self.K)
            
            # 检查重叠
            overlap = False
            for gx, gy in generators:
                if max(abs(x - gx), abs(y - gy)) < self.K:
                    overlap = True
                    break
            
            if not overlap:
                generators.append((x, y))
            attempts += 1
        
        return generators
    
    def reset(self):
        """重置环境到初始状态"""
        self.substation_positions = []
        self.placed_count = 0
        self.done = False
        self.step_count = 0
        
        # 初始化状态
        self.state = self._get_state_representation()
        
        return self.state
    
    def _get_state_representation(self):
        """获取状态表示 - 多通道特征图"""
        state = torch.zeros(self.state_dim, self.D, self.D, dtype=torch.float32, device=self.device)
        
        # 通道0: 发电机位置
        for gx, gy in self.generator_positions:
            for i in range(self.K):
                for j in range(self.K):
                    if gx+i < self.D and gy+j < self.D:
                        state[0, gx+i, gy+j] = 1.0
        
        # 通道1: 已放置的变电站
        for sx, sy in self.substation_positions:
            for i in range(self.K):
                for j in range(self.K):
                    if sx+i < self.D and sy+j < self.D:
                        state[1, sx+i, sy+j] = 1.0
        
        # 通道2: 占用位置 (发电机 + 变电站)
        state[2] = torch.clamp(state[0] + state[1], 0, 1)
        
        # 通道3: 可用位置的价值估计 (热图)
        if self.generator_positions:
            state[3] = self._compute_value_heatmap(state[2])  # 传入占用图
        
        return state
    
    def _compute_value_heatmap(self, occupancy_map=None):
        """
        计算位置价值热图 - 受SOFTDIST启发
        Args:
            occupancy_map: 占用位置图，如果为None则使用当前状态
        """
        heatmap = torch.zeros(self.D, self.D, dtype=torch.float32, device=self.device)
        
        # 如果没有传入占用图，尝试从当前状态获取
        if occupancy_map is None and self.state is not None:
            occupancy_map = self.state[2]
        
        for x in range(self.D - self.K + 1):
            for y in range(self.D - self.K + 1):
                if self._is_valid_position_with_occupancy(x, y, occupancy_map):
                    # 计算该位置作为变电站的价值
                    value = self._calculate_position_value(x, y)
                    
                    # 在KxK区域内设置价值
                    for i in range(self.K):
                        for j in range(self.K):
                            if x+i < self.D and y+j < self.D:
                                heatmap[x+i, y+j] = max(heatmap[x+i, y+j], value)
        
        # 归一化热图
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def _calculate_position_value(self, x, y):
        """计算位置价值 - 基于到发电机的距离"""
        if not self.generator_positions:
            return 0.0
        
        total_value = 0.0
        
        for gx, gy in self.generator_positions:
            # 计算距离
            dist = self.distance_metric.calculate((gx, gy), (x, y))
            
            # 距离越近价值越高 (使用指数衰减)
            value = np.exp(-dist / 10.0)  # 可调参数
            total_value += value
        
        return total_value
    
    def _is_valid_position_with_occupancy(self, x, y, occupancy_map):
        """
        检查位置是否有效 - 使用传入的占用图
        Args:
            x, y: 位置坐标
            occupancy_map: 占用位置图
        """
        # 边界检查
        if x < 0 or y < 0 or x + self.K > self.D or y + self.K > self.D:
            return False
        
        # 重叠检查
        if occupancy_map is not None:
            for i in range(self.K):
                for j in range(self.K):
                    if occupancy_map[x+i, y+j] > 0:  # 已被占用
                        return False
        
        return True
    
    def _is_valid_position(self, x, y):
        """检查位置是否有效 - 使用当前状态"""
        # 边界检查
        if x < 0 or y < 0 or x + self.K > self.D or y + self.K > self.D:
            return False
        
        # 重叠检查 - 使用当前状态
        if self.state is not None:
            for i in range(self.K):
                for j in range(self.K):
                    if self.state[2, x+i, y+j] > 0:  # 已被占用
                        return False
        else:
            # 如果状态还未初始化，直接检查与发电机的重叠
            for gx, gy in self.generator_positions:
                if (abs(x - gx) < self.K and abs(y - gy) < self.K):
                    return False
        
        return True
    
    def get_action_mask(self):
        """获取有效动作掩码"""
        mask = torch.zeros(self.action_space_size, dtype=torch.bool, device=self.device)
        
        for idx in range(self.action_space_size):
            x, y = self._action_to_position(idx)
            if self._is_valid_position(x, y):
                mask[idx] = True
        
        return mask
    
    def _action_to_position(self, action):
        """将动作索引转换为位置坐标"""
        grid_size = self.D - self.K + 1
        x = action // grid_size
        y = action % grid_size
        return x, y
    
    def _position_to_action(self, x, y):
        """将位置坐标转换为动作索引"""
        grid_size = self.D - self.K + 1
        return x * grid_size + y
    
    def step(self, action):
        """执行动作"""
        self.step_count += 1
        
        # 将动作转换为位置
        x, y = self._action_to_position(action)
        
        # 检查动作有效性
        if not self._is_valid_position(x, y):
            # 无效动作，给予负奖励
            reward = -10.0
            info = {'invalid_action': True, 'action_position': (x, y)}
            return self.state, reward, self.done, info
        
        # 放置变电站
        self.substation_positions.append((x, y))
        self.placed_count += 1
        
        # 更新状态
        self.state = self._get_state_representation()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否完成
        if self.placed_count >= self.M:
            self.done = True
            # 给予最终奖励
            reward += self._calculate_final_reward()
        
        info = {
            'placed_count': self.placed_count,
            'target_count': self.M,
            'step_count': self.step_count,
            'action_position': (x, y),
            'invalid_action': False
        }
        
        return self.state, reward, self.done, info
    
    def _calculate_reward(self):
        """计算即时奖励"""
        if not self.substation_positions:
            return 0.0
        
        # 基于当前变电站配置计算奖励
        total_distance = 0.0
        
        for gx, gy in self.generator_positions:
            min_dist = float('inf')
            for sx, sy in self.substation_positions:
                dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
            total_distance += min_dist
        
        # 归一化奖励 (负距离，因为要最小化)
        reward = -total_distance / len(self.generator_positions)
        
        # 添加变电站覆盖奖励
        coverage_bonus = self._calculate_coverage_bonus()
        
        return reward + coverage_bonus
    
    def _calculate_coverage_bonus(self):
        """计算覆盖奖励 - 鼓励变电站分布均匀"""
        if len(self.substation_positions) <= 1:
            return 0.0
        
        # 计算变电站之间的距离
        min_inter_distance = float('inf')
        
        for i, (sx1, sy1) in enumerate(self.substation_positions):
            for j, (sx2, sy2) in enumerate(self.substation_positions[i+1:], i+1):
                dist = self.distance_metric.calculate((sx1, sy1), (sx2, sy2))
                if dist < min_inter_distance:
                    min_inter_distance = dist
        
        # 鼓励适度的变电站间距
        optimal_distance = self.D / (2 * np.sqrt(self.M))
        distance_score = 1.0 - abs(min_inter_distance - optimal_distance) / optimal_distance
        
        return max(0, distance_score) * 2.0  # 覆盖奖励系数
    
    def _calculate_final_reward(self):
        """计算最终奖励"""
        if len(self.substation_positions) != self.M:
            return -50.0  # 惩罚未完成的情况
        
        # 计算总的服务质量
        total_distance = 0.0
        max_distance = 0.0
        
        for gx, gy in self.generator_positions:
            min_dist = float('inf')
            for sx, sy in self.substation_positions:
                dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
            
            total_distance += min_dist
            max_distance = max(max_distance, min_dist)
        
        # 综合奖励: 总距离 + 最大距离 (考虑公平性)
        avg_distance = total_distance / len(self.generator_positions)
        fairness_penalty = max_distance - avg_distance
        
        final_reward = -(avg_distance + 0.5 * fairness_penalty)
        
        # 完成奖励
        completion_bonus = 10.0
        
        return final_reward + completion_bonus
    
    def render(self, save_path=None):
        """可视化当前状态"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制网格
        ax.set_xlim(0, self.D)
        ax.set_ylim(0, self.D)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f'Power Grid RL Environment (Step: {self.step_count})')
        
        # 绘制价值热图
        if hasattr(self, 'state') and self.state is not None:
            heatmap = self.state[3].cpu().numpy()
            im = ax.imshow(heatmap.T, origin='lower', alpha=0.3, cmap='viridis')
            plt.colorbar(im, ax=ax, label='Position Value')
        
        # 绘制发电机
        for i, (gx, gy) in enumerate(self.generator_positions):
            rect = Rectangle((gx, gy), self.K, self.K, color='red', alpha=0.8)
            ax.add_patch(rect)
            ax.text(gx + self.K/2, gy + self.K/2, f'G{i+1}', 
                   ha='center', va='center', fontweight='bold', color='white')
        
        # 绘制变电站
        for i, (sx, sy) in enumerate(self.substation_positions):
            rect = Rectangle((sx, sy), self.K, self.K, color='blue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(sx + self.K/2, sy + self.K/2, f'S{i+1}', 
                   ha='center', va='center', fontweight='bold', color='white')
        
        # 绘制连接线
        for gx, gy in self.generator_positions:
            if self.substation_positions:
                min_dist = float('inf')
                closest_sub = None
                for sx, sy in self.substation_positions:
                    dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                    if dist < min_dist:
                        min_dist = dist
                        closest_sub = (sx, sy)
                
                if closest_sub:
                    sx, sy = closest_sub
                    ax.plot([gx + self.K/2, sx + self.K/2], 
                           [gy + self.K/2, sy + self.K/2], 
                           'gray', linestyle='-', alpha=0.6)
        
        # 添加信息
        info_text = f"Generators: {len(self.generator_positions)}\n"
        info_text += f"Substations: {len(self.substation_positions)}/{self.M}\n"
        info_text += f"Steps: {self.step_count}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()