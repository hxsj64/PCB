"""
修复后的DRL训练可视化器 - 生成网格布局GIF
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio
import os
from tqdm import tqdm
import json

class DRLTrainingVisualizer:
    def __init__(self, model_path, env_config, distance_metric, device='cpu'):
        """
        初始化可视化器
        """
        self.model_path = model_path
        self.env_config = env_config
        self.distance_metric = distance_metric
        self.device = device
        
        # 加载模型
        self.model = self._load_model()
        
        # 创建环境
        from drl_environment import PowerGridRLEnv
        self.env = PowerGridRLEnv(
            N=env_config['N'],
            M=env_config['M'],
            D=env_config['D'],
            K=env_config['K'],
            distance_metric=distance_metric,
            device=device
        )
    
    def _load_model(self):
        """加载训练好的模型"""
        from attention_network import POMO_PowerGrid
        
        # 从配置文件读取模型参数
        config_path = os.path.join(os.path.dirname(self.model_path), 'experiment_info.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_params = config.get('model_params', {})
        else:
            model_params = {'d_model': 256, 'n_head': 8, 'n_layers': 6}
        
        model = POMO_PowerGrid(
            grid_size=self.env_config['D'],
            input_dim=4,
            d_model=model_params.get('d_model', 256),
            n_head=model_params.get('n_head', 8),
            n_layers=model_params.get('n_layers', 6),
            K=self.env_config['K']
        ).to(self.device)
        
        # 加载权重
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        except Exception as e:
            print(f"⚠️ 模型加载失败，使用随机权重: {str(e)}")
        
        return model
    
    def generate_step_by_step_placement_gif(self, save_dir):
        """生成逐步放置变电站的网格GIF"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 重置环境
        state = self.env.reset()
        images = []
        step_count = 0
        
        print(f"🎬 生成逐步放置GIF...")
        
        # 记录初始状态
        images.append(self._create_grid_frame(step_count, "Initial State", 0.0))
        
        while not self.env.done and step_count < 20:
            # 模型推理
            if isinstance(state, torch.Tensor):
                if state.device != self.device:
                    state = state.to(self.device)
                state_tensor = state.unsqueeze(0)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            action_mask = self.env.get_action_mask()
            if action_mask.device != self.device:
                action_mask = action_mask.to(self.device)
            action_mask = action_mask.unsqueeze(0)
            
            with torch.no_grad():
                action, policy_logits, value, probs = self.model(
                    state_tensor, action_mask, deterministic=True
                )
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action.item())
            step_count += 1
            
            # 创建当前步骤的帧
            action_pos = self.env._action_to_position(action.item())
            step_info = f"Step {step_count}: Place substation at ({action_pos[0]}, {action_pos[1]})"
            images.append(self._create_grid_frame(step_count, step_info, reward, 
                                               highlight_pos=action_pos, probs=probs[0].cpu().numpy()))
            
            state = next_state
        
        # 创建最终状态帧
        if self.env.done:
            final_distance = self._calculate_final_distance()
            images.append(self._create_grid_frame(step_count, f"Final Solution (Distance: {final_distance:.2f})", 0.0, final=True))
        
        # 保存GIF
        if images:
            gif_path = os.path.join(save_dir, "step_by_step_placement.gif")
            imageio.mimsave(gif_path, images, fps=2, loop=0)
            print(f"✅ 逐步放置GIF已保存: {gif_path}")
            
            # 复制到标准文件名
            standard_gif_path = os.path.join(save_dir, "decision_making_process.gif")
            imageio.mimsave(standard_gif_path, images, fps=2, loop=0)
        
        return len(images)
    
    def _create_grid_frame(self, step, title, reward, highlight_pos=None, probs=None, final=False):
        """创建单个网格帧"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        D = self.env_config['D']
        K = self.env_config['K']
        
        # 左侧：网格布局
        ax1.set_xlim(0, D)
        ax1.set_ylim(0, D)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_title(f"{title}\nReward: {reward:.2f}")
        
        # 绘制发电机（红色）
        for i, (gx, gy) in enumerate(self.env.generator_positions):
            rect = Rectangle((gx, gy), K, K, color='red', alpha=0.8, edgecolor='darkred', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(gx + K/2, gy + K/2, f'G{i+1}', 
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        # 绘制已放置的变电站（蓝色）
        for i, (sx, sy) in enumerate(self.env.substation_positions):
            rect = Rectangle((sx, sy), K, K, color='blue', alpha=0.8, edgecolor='darkblue', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(sx + K/2, sy + K/2, f'S{i+1}', 
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        # 高亮当前动作位置（绿色边框）
        if highlight_pos is not None:
            hx, hy = highlight_pos
            highlight_rect = Rectangle((hx, hy), K, K, fill=False, 
                                     edgecolor='lime', linewidth=4, alpha=0.8)
            ax1.add_patch(highlight_rect)
            ax1.text(hx + K/2, hy - 0.5, '🆕 NEW', ha='center', va='top', 
                    fontweight='bold', color='lime', fontsize=12)
        
        # 绘制连接线
        self._draw_connections_on_grid(ax1)
        
        # 添加网格信息
        info_text = f"Grid: {D}×{D}\nGenerators: {len(self.env.generator_positions)}\nSubstations: {len(self.env.substation_positions)}/{self.env_config['M']}"
        if final:
            final_distance = self._calculate_final_distance()
            info_text += f"\nFinal Distance: {final_distance:.2f}"
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # 右侧：策略概率热图（如果有的话）
        if probs is not None and not final:
            action_grid_size = D - K + 1
            if len(probs) == action_grid_size ** 2:
                policy_map = probs.reshape(action_grid_size, action_grid_size)
                im = ax2.imshow(policy_map.T, origin='lower', cmap='viridis', alpha=0.8)
                ax2.set_title("Action Probability Heatmap")
                ax2.set_xlabel('X Position')
                ax2.set_ylabel('Y Position')
                
                # 标记选择的动作
                if highlight_pos is not None:
                    hx, hy = highlight_pos
                    ax2.plot(hx, hy, 'r*', markersize=20, label='Selected Action')
                    ax2.legend()
                
                plt.colorbar(im, ax=ax2, label='Probability')
            else:
                ax2.text(0.5, 0.5, 'Probability map\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("Action Probabilities")
        else:
            # 显示当前网格状态统计
            ax2.axis('off')
            if final:
                stats_text = self._generate_final_stats()
            else:
                stats_text = self._generate_current_stats(step, reward)
            
            ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            ax2.set_title("Status Information")
        
        plt.tight_layout()
        
        # 转换为图像
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return image
    
    def _draw_connections_on_grid(self, ax):
        """在网格上绘制连接线"""
        K = self.env_config['K']
        
        for gx, gy in self.env.generator_positions:
            if self.env.substation_positions:
                min_dist = float('inf')
                closest_sub = None
                
                for sx, sy in self.env.substation_positions:
                    dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                    if dist < min_dist:
                        min_dist = dist
                        closest_sub = (sx, sy)
                
                if closest_sub:
                    sx, sy = closest_sub
                    
                    # 根据距离度量类型绘制不同样式的连接线
                    if self.distance_metric.name().lower() == "manhattan":
                        # L形路径
                        start_x, start_y = gx + K/2, gy + K/2
                        end_x, end_y = sx + K/2, sy + K/2
                        ax.plot([start_x, end_x], [start_y, start_y], 
                               'gray', linestyle='-', linewidth=2, alpha=0.7)
                        ax.plot([end_x, end_x], [start_y, end_y], 
                               'gray', linestyle='-', linewidth=2, alpha=0.7)
                        ax.plot(end_x, start_y, 'o', color='gray', markersize=4)
                    else:
                        # 直线连接
                        line_style = '--' if self.distance_metric.name().lower() == "chebyshev" else '-'
                        ax.plot([gx + K/2, sx + K/2], [gy + K/2, sy + K/2], 
                               'gray', linestyle=line_style, linewidth=2, alpha=0.7)
    
    def _calculate_final_distance(self):
        """计算最终总距离"""
        total_distance = 0
        for gx, gy in self.env.generator_positions:
            min_dist = float('inf')
            for sx, sy in self.env.substation_positions:
                dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
            total_distance += min_dist
        return total_distance
    
    def _generate_current_stats(self, step, reward):
        """生成当前状态统计信息"""
        stats = f"""
Step: {step}
Current Reward: {reward:.2f}
Placed Substations: {len(self.env.substation_positions)}/{self.env_config['M']}

Distance Metric: {self.distance_metric.name().title()}
Grid Size: {self.env_config['D']}×{self.env_config['D']}
Device Size: {self.env_config['K']}×{self.env_config['K']}

Current Distances:"""
        
        if self.env.substation_positions:
            for i, (gx, gy) in enumerate(self.env.generator_positions):
                min_dist = float('inf')
                for sx, sy in self.env.substation_positions:
                    dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                    if dist < min_dist:
                        min_dist = dist
                stats += f"\nG{i+1}: {min_dist:.1f}"
        else:
            stats += "\nNo substations placed yet"
        
        return stats
    
    def _generate_final_stats(self):
        """生成最终统计信息"""
        total_distance = self._calculate_final_distance()
        avg_distance = total_distance / len(self.env.generator_positions)
        
        stats = f"""
🎉 SOLUTION COMPLETE!

Final Results:
Total Distance: {total_distance:.2f}
Average Distance: {avg_distance:.2f}
Substations Placed: {len(self.env.substation_positions)}

Distance Breakdown:"""
        
        for i, (gx, gy) in enumerate(self.env.generator_positions):
            min_dist = float('inf')
            closest_sub = None
            for sx, sy in self.env.substation_positions:
                dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
                    closest_sub = (sx, sy)
            
            stats += f"\nG{i+1}({gx},{gy}) → S{closest_sub}: {min_dist:.1f}"
        
        return stats
    
    def generate_decision_making_visualization(self, save_dir):
        """生成决策过程可视化 - 主要接口"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成逐步放置GIF
        num_frames = self.generate_step_by_step_placement_gif(save_dir)
        
        # 生成分析报告
        self._generate_placement_analysis(save_dir)
        
        print(f"✅ 决策可视化完成，共生成 {num_frames} 帧")
        return []  # 返回空列表保持兼容性
    
    def _generate_placement_analysis(self, save_dir):
        """生成放置分析报告"""
        analysis_path = os.path.join(save_dir, "placement_analysis.txt")
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("DRL变电站放置分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"配置信息:\n")
            f.write(f"- 网格大小: {self.env_config['D']}×{self.env_config['D']}\n")
            f.write(f"- 发电机数量: {self.env_config['N']}\n")
            f.write(f"- 变电站数量: {self.env_config['M']}\n")
            f.write(f"- 设备尺寸: {self.env_config['K']}×{self.env_config['K']}\n")
            f.write(f"- 距离度量: {self.distance_metric.name()}\n\n")
            
            f.write(f"发电机位置:\n")
            for i, (gx, gy) in enumerate(self.env.generator_positions):
                f.write(f"- G{i+1}: ({gx}, {gy})\n")
            
            f.write(f"\n变电站位置:\n")
            for i, (sx, sy) in enumerate(self.env.substation_positions):
                f.write(f"- S{i+1}: ({sx}, {sy})\n")
            
            if self.env.substation_positions:
                total_distance = self._calculate_final_distance()
                f.write(f"\n最终结果:\n")
                f.write(f"- 总距离: {total_distance:.2f}\n")
                f.write(f"- 平均距离: {total_distance/len(self.env.generator_positions):.2f}\n")
        
        print(f"📋 分析报告已保存: {analysis_path}")

    def compare_with_baselines(self, save_dir, num_episodes=5):
        """与基准方法比较 - 保持接口兼容性"""
        print(f"📊 基准比较功能暂时简化，主要专注于网格可视化")
        return True