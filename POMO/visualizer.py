import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import random
from tqdm import tqdm
import imageio

class PowerGridVisualizer:
    def __init__(self, D, K, distance_metric):
        self.D = D
        self.K = K
        self.distance_metric = distance_metric
    
    def _draw_connection_line(self, ax, start_pos, end_pos):
        """
        根据距离度量类型绘制不同样式的连接线
        """
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        if self.distance_metric.name().lower() == "manhattan":
            # 曼哈顿距离：绘制L形路径（横线+竖线）
            # 方案1：先水平后垂直
            ax.plot([start_x, end_x], [start_y, start_y], 'gray', linestyle='-', alpha=0.6, linewidth=2)
            ax.plot([end_x, end_x], [start_y, end_y], 'gray', linestyle='-', alpha=0.6, linewidth=2)
            
            # 添加转折点标记
            ax.plot(end_x, start_y, 'o', color='gray', markersize=3, alpha=0.7)
            
        elif self.distance_metric.name().lower() == "chebyshev":
            # 切比雪夫距离：绘制阶梯状路径
            steps = max(abs(end_x - start_x), abs(end_y - start_y))
            if steps > 0:
                step_x = (end_x - start_x) / steps
                step_y = (end_y - start_y) / steps
                
                x_coords = [start_x]
                y_coords = [start_y]
                
                for i in range(int(steps)):
                    x_coords.append(start_x + (i + 1) * step_x)
                    y_coords.append(start_y + (i + 1) * step_y)
                
                ax.plot(x_coords, y_coords, 'gray', linestyle='--', alpha=0.6, linewidth=2)
            else:
                ax.plot([start_x, end_x], [start_y, end_y], 'gray', linestyle='--', alpha=0.6, linewidth=2)
                
        else:
            # 欧几里得距离：直线连接
            ax.plot([start_x, end_x], [start_y, end_y], 'gray', linestyle='-', alpha=0.5, linewidth=1.5)
    
    def visualize_solution(self, generators, substations, title="", save_path=None):
        """可视化单个解决方案"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制网格
        ax.set_xlim(0, self.D)
        ax.set_ylim(0, self.D)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title} ({self.distance_metric.name()} Distance)", fontsize=14)
        ax.set_aspect('equal')
        
        # 计算总距离
        total_distance = 0
        connections = []
        
        # 绘制发电机 (红色)
        for i, (gx, gy) in enumerate(generators):
            rect = Rectangle((gx, gy), self.K, self.K, color='red', alpha=0.7, 
                             label='Generator' if i == 0 else "")
            ax.add_patch(rect)
            
            # 找到最近的变电站
            min_dist = float('inf')
            closest_sub = None
            for j, (sx, sy) in enumerate(substations):
                dist = self.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
                    closest_sub = (sx, sy)
            
            total_distance += min_dist
            if closest_sub:
                sx, sy = closest_sub
                connections.append(((gx + self.K/2, gy + self.K/2), 
                                    (sx + self.K/2, sy + self.K/2)))
        
        # 绘制变电站 (蓝色)
        for i, (sx, sy) in enumerate(substations):
            rect = Rectangle((sx, sy), self.K, self.K, color='blue', alpha=0.7, 
                             label='Substation' if i == 0 else "")
            ax.add_patch(rect)
        
        # 绘制连接线
        for (start, end) in connections:
            self._draw_connection_line(ax, start, end)
        
        # 添加距离信息和路径说明
        distance_info = f"Total {self.distance_metric.name()} Distance: {total_distance:.2f}"
        if self.distance_metric.name().lower() == "manhattan":
            distance_info += "\n(L-shaped paths: horizontal + vertical)"
        elif self.distance_metric.name().lower() == "chebyshev":
            distance_info += "\n(Diagonal + orthogonal paths)"
        else:
            distance_info += "\n(Straight-line paths)"
            
        ax.text(0.05, 0.95, distance_info, 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:  # 确保图例不为空
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            plt.show()
        
        return total_distance

def create_training_gif(visualizer, generators, training_history, save_path, fps=4):
    """创建训练过程GIF"""
    images = []
    
    for epoch, substations in tqdm(training_history, desc=f"Creating GIF for {save_path}"):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        ax.set_xlim(0, visualizer.D)
        ax.set_ylim(0, visualizer.D)
        ax.grid(True, alpha=0.3)
        
        # 标题包含路径类型信息
        path_type = ""
        if visualizer.distance_metric.name().lower() == "manhattan":
            path_type = " (L-shaped paths)"
        elif visualizer.distance_metric.name().lower() == "chebyshev":
            path_type = " (Diagonal paths)"
        else:
            path_type = " (Straight paths)"
            
        ax.set_title(f"Training Progress - {visualizer.distance_metric.name()} Distance{path_type} (Epoch: {epoch})", fontsize=10)
        ax.set_aspect('equal')
        
        # 计算总距离
        total_distance = 0
        
        # 绘制发电机 (红色)
        for gx, gy in generators:
            rect = Rectangle((gx, gy), visualizer.K, visualizer.K, color='red', alpha=0.7)
            ax.add_patch(rect)
            
            # 找到最近的变电站
            min_dist = float('inf')
            closest_sub = None
            for sx, sy in substations:
                dist = visualizer.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
                    closest_sub = (sx, sy)
            
            total_distance += min_dist
            if closest_sub:
                sx, sy = closest_sub
                # 使用新的连接线绘制方法
                visualizer._draw_connection_line(
                    ax, 
                    (gx + visualizer.K/2, gy + visualizer.K/2),
                    (sx + visualizer.K/2, sy + visualizer.K/2)
                )
        
        # 绘制变电站 (蓝色)
        for i, (sx, sy) in enumerate(substations):
            rect = Rectangle((sx, sy), visualizer.K, visualizer.K, color='blue', alpha=0.7,
                             label=f'Substation {i+1}' if i < 3 else "")  # 限制图例数量
            ax.add_patch(rect)
        
        # 添加距离信息
        ax.text(0.05, 0.95, f"Total {visualizer.distance_metric.name()} Distance: {total_distance:.2f}", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # 添加图例（仅显示前几个变电站以避免拥挤）
        handles, labels = ax.get_legend_handles_labels()
        if handles:  # 确保有图例可添加
            # 只显示发电机和前几个变电站的图例
            unique_labels = []
            unique_handles = []
            seen_labels = set()
            for handle, label in zip(handles, labels):
                if label not in seen_labels:
                    unique_handles.append(handle)
                    unique_labels.append(label)
                    seen_labels.add(label)
            
            if unique_handles:
                ax.legend(unique_handles, unique_labels, loc='lower right')
        
        # 保存图像到内存
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    
    # 保存为GIF
    imageio.mimsave(save_path, images, fps=fps, loop=0)

def create_comparison_visualization_with_paths(results, config, output_dir):
    """创建包含不同路径样式的比较可视化"""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    
    fig.suptitle(f"Distance Metrics & Path Types Comparison (N={config['N']}, M={config['M']}, D={config['D']})", fontsize=16)
    
    for i, result in enumerate(results):
        ax = axes[i]
        vis = result["visualizer"]
        generators = result["generators"]
        substations = result["substations"]
        
        # 绘制网格
        ax.set_xlim(0, vis.D)
        ax.set_ylim(0, vis.D)
        ax.grid(True, alpha=0.3)
        
        # 标题包含路径类型信息
        path_type = ""
        if vis.distance_metric.name().lower() == "manhattan":
            path_type = "\n(L-shaped paths)"
        elif vis.distance_metric.name().lower() == "chebyshev":
            path_type = "\n(Diagonal paths)"
        else:
            path_type = "\n(Straight paths)"
            
        ax.set_title(f"{result['distance_name']} Distance: {result['distance']:.2f}{path_type}")
        ax.set_aspect('equal')
        
        # 绘制发电机 (红色)
        for j, (gx, gy) in enumerate(generators):
            rect = plt.Rectangle((gx, gy), vis.K, vis.K, color='red', alpha=0.7, 
                                 label='Generator' if j == 0 else "")
            ax.add_patch(rect)
        
        # 绘制变电站 (蓝色)
        for j, (sx, sy) in enumerate(substations):
            rect = plt.Rectangle((sx, sy), vis.K, vis.K, color='blue', alpha=0.7, 
                                 label='Substation' if j == 0 else "")
            ax.add_patch(rect)
        
        # 绘制连接线
        for gx, gy in generators:
            min_dist = float('inf')
            closest_sub = None
            for sx, sy in substations:
                dist = vis.distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
                    closest_sub = (sx, sy)
            
            if closest_sub:
                sx, sy = closest_sub
                vis._draw_connection_line(
                    ax,
                    (gx + vis.K/2, gy + vis.K/2),
                    (sx + vis.K/2, sy + vis.K/2)
                )
        
        # 添加图例
        if i == 0:  # 只在第一个子图添加图例
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_comparison_with_paths.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)