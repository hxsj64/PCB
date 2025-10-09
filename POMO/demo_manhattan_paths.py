"""
专门演示曼哈顿距离L形路径的脚本
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from distance_metrics import ManhattanDistance
from visualizer import PowerGridVisualizer

def demo_manhattan_paths():
    """演示曼哈顿距离的L形路径"""
    
    # 创建一个简单的示例
    generators = [(2, 2), (8, 7), (3, 8)]
    substations = [(5, 5)]
    D, K = 10, 1
    
    # 创建可视化器
    vis = PowerGridVisualizer(D=D, K=K, distance_metric=ManhattanDistance)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 左图：显示L形路径
    ax1.set_xlim(0, D)
    ax1.set_ylim(0, D)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Manhattan Distance - L-shaped Paths", fontsize=14)
    ax1.set_aspect('equal')
    
    # 绘制发电机和变电站
    for i, (gx, gy) in enumerate(generators):
        rect = Rectangle((gx, gy), K, K, color='red', alpha=0.7, 
                         label='Generator' if i == 0 else "")
        ax1.add_patch(rect)
    
    for i, (sx, sy) in enumerate(substations):
        rect = Rectangle((sx, sy), K, K, color='blue', alpha=0.7, 
                         label='Substation' if i == 0 else "")
        ax1.add_patch(rect)
    
    # 绘制L形路径
    colors = ['green', 'orange', 'purple']
    total_distance = 0
    
    for i, (gx, gy) in enumerate(generators):
        sx, sy = substations[0]  # 只有一个变电站
        
        # 计算距离
        distance = ManhattanDistance.calculate((gx, gy), (sx, sy))
        total_distance += distance
        
        # 绘制L形路径
        center_g = (gx + K/2, gy + K/2)
        center_s = (sx + K/2, sy + K/2)
        
        # 水平线
        ax1.plot([center_g[0], center_s[0]], [center_g[1], center_g[1]], 
                 color=colors[i], linestyle='-', linewidth=3, alpha=0.8,
                 label=f'Path {i+1} (dist: {distance:.1f})' if i < 3 else "")
        
        # 垂直线
        ax1.plot([center_s[0], center_s[0]], [center_g[1], center_s[1]], 
                 color=colors[i], linestyle='-', linewidth=3, alpha=0.8)
        
        # 转折点
        ax1.plot(center_s[0], center_g[1], 'o', color=colors[i], markersize=6)
        
        # 添加距离标注
        mid_x = (center_g[0] + center_s[0]) / 2
        mid_y = center_g[1] + 0.2
        ax1.text(mid_x, mid_y, f'd={distance:.1f}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
    
    ax1.text(0.05, 0.95, f"Total Manhattan Distance: {total_distance:.1f}", 
             transform=ax1.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    ax1.legend(loc='upper right')
    
    # 右图：对比直线路径（欧几里得）
    ax2.set_xlim(0, D)
    ax2.set_ylim(0, D)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Euclidean Distance - Straight Paths", fontsize=14)
    ax2.set_aspect('equal')
    
    # 绘制发电机和变电站
    for i, (gx, gy) in enumerate(generators):
        rect = Rectangle((gx, gy), K, K, color='red', alpha=0.7)
        ax2.add_patch(rect)
    
    for i, (sx, sy) in enumerate(substations):
        rect = Rectangle((sx, sy), K, K, color='blue', alpha=0.7)
        ax2.add_patch(rect)
    
    # 绘制直线路径
    total_euclidean = 0
    for i, (gx, gy) in enumerate(generators):
        sx, sy = substations[0]
        
        # 计算欧几里得距离
        distance = np.sqrt((gx - sx)**2 + (gy - sy)**2)
        total_euclidean += distance
        
        center_g = (gx + K/2, gy + K/2)
        center_s = (sx + K/2, sy + K/2)
        
        ax2.plot([center_g[0], center_s[0]], [center_g[1], center_s[1]], 
                color=colors[i], linestyle='-', linewidth=3, alpha=0.8)
        
        # 添加距离标注
        mid_x = (center_g[0] + center_s[0]) / 2
        mid_y = (center_g[1] + center_s[1]) / 2
        ax2.text(mid_x, mid_y, f'd={distance:.1f}', 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
    
    ax2.text(0.05, 0.95, f"Total Euclidean Distance: {total_euclidean:.1f}", 
             transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("manhattan_vs_euclidean_paths_demo.png", bbox_inches='tight', dpi=150)
    plt.show()
    
    print(f"曼哈顿距离总和: {total_distance:.2f}")
    print(f"欧几里得距离总和: {total_euclidean:.2f}")
    print(f"差异: {total_distance - total_euclidean:.2f}")

if __name__ == "__main__":
    demo_manhattan_paths()