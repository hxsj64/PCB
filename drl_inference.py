"""
DRL模型推理脚本 - 用于部署和实时推理
"""
import torch
import numpy as np
import json
import os
from pathlib import Path

from distance_metrics import DISTANCE_METRICS
from drl_environment import PowerGridRLEnv
from attention_network import POMO_PowerGrid

class DRLInference:
    def __init__(self, model_path, config_path=None, device='cpu'):
        """
        DRL推理器
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
            device: 计算设备
        """
        self.device = device
        self.model_path = model_path
        
        # 加载配置
        if config_path is None:
            config_path = Path(model_path).parent / "experiment_info.json"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            self.config = config_data.get('config', config_data)
            self.model_params = config_data.get('model_params', {})
        else:
            # 默认配置
            print(f"⚠️ 未找到配置文件，使用默认配置")
            self.config = {"N": 5, "M": 1, "D": 25, "K": 2, "name": "Default"}
            self.model_params = {'d_model': 256, 'n_head': 8, 'n_layers': 6}
        
        # 初始化模型
        self.model = self._load_model()
        
        print(f"✅ DRL推理器初始化完成")
        print(f"📁 模型: {model_path}")
        print(f"⚙️ 配置: {self.config}")
        print(f"🔧 设备: {device}")
    
    def _load_model(self):
        """加载模型"""
        model = POMO_PowerGrid(
            grid_size=self.config['D'],
            input_dim=4,
            d_model=self.model_params.get('d_model', 256),
            n_head=self.model_params.get('n_head', 8),
            n_layers=self.model_params.get('n_layers', 6),
            K=self.config['K']
        ).to(self.device)
        
        # 加载权重
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✅ 模型权重加载成功")
        except Exception as e:
            print(f"❌ 模型权重加载失败: {str(e)}")
            # 使用随机初始化的模型
            print(f"⚠️ 使用随机初始化的模型")
        
        return model
    
    def solve(self, generators, distance_metric_name='manhattan', deterministic=True, max_steps=20):
        """
        求解电力网格优化问题
        Args:
            generators: 发电机位置列表 [(x1,y1), (x2,y2), ...]
            distance_metric_name: 距离度量名称
            deterministic: 是否使用确定性策略
            max_steps: 最大步数
        Returns:
            dict: 包含解决方案和统计信息
        """
        # 创建环境
        distance_metric = DISTANCE_METRICS[distance_metric_name]
        
        env = PowerGridRLEnv(
            N=self.config['N'],
            M=self.config['M'],
            D=self.config['D'],
            K=self.config['K'],
            distance_metric=distance_metric,
            device=self.device
        )
        
        # 设置发电机位置
        env.generator_positions = generators[:self.config['N']]  # 截取到指定数量
        
        # 开始求解
        state = env.reset()
        solution_steps = []
        total_reward = 0
        
        step_count = 0
        while not env.done and step_count < max_steps:
            # 模型推理
            if isinstance(state, torch.Tensor):
                if state.device != self.device:
                    state = state.to(self.device)
                state_tensor = state.unsqueeze(0)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            action_mask = env.get_action_mask()
            if action_mask.device != self.device:
                action_mask = action_mask.to(self.device)
            action_mask = action_mask.unsqueeze(0)
            
            with torch.no_grad():
                action, policy_logits, value, probs = self.model(
                    state_tensor, action_mask, deterministic=deterministic
                )
            
            # 执行动作
            next_state, reward, done, info = env.step(action.item())
            
            # 记录步骤信息
            step_info = {
                'step': step_count,
                'action': action.item(),
                'position': env._action_to_position(action.item()),
                'reward': reward,
                'value_estimate': value.item(),
                'action_probability': probs[0, action.item()].item()
            }
            solution_steps.append(step_info)
            
            total_reward += reward
            state = next_state
            step_count += 1
        
        # 计算最终距离
        final_distance = self._calculate_total_distance(
            env.generator_positions, 
            env.substation_positions, 
            distance_metric
        )
        
        # 构建结果
        result = {
            'success': env.done and len(env.substation_positions) == self.config['M'],
            'generators': env.generator_positions,
            'substations': env.substation_positions,
            'total_distance': final_distance,
            'total_reward': total_reward,
            'steps_taken': step_count,
            'distance_metric': distance_metric_name,
            'solution_steps': solution_steps,
        }
        
        return result
    
    def _calculate_total_distance(self, generators, substations, distance_metric):
        """计算总距离"""
        if not substations:
            return float('inf')
        
        total_distance = 0
        for gx, gy in generators:
            min_dist = float('inf')
            for sx, sy in substations:
                dist = distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
            total_distance += min_dist
        
        return total_distance