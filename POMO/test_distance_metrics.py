import os
import random
import numpy as np
import matplotlib.pyplot as plt
from distance_metrics import DISTANCE_METRICS
from environment import PowerGridEnv
from visualizer import PowerGridVisualizer, create_training_gif
from benchmark_algorithms import generate_optimized_substations, generate_training_history

# 设置随机种子
random.seed(123)
np.random.seed(123)

def test_single_distance_metric(distance_name, config):
    """测试单个距离度量"""
    print(f"\nTesting {distance_name} distance metric...")
    
    distance_class = DISTANCE_METRICS[distance_name]
    N, M, D, K = config["N"], config["M"], config["D"], config["K"]
    
    # 创建输出目录
    output_dir = f"results_{distance_name}_distance"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成发电机位置
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
    
    # 创建环境
    env = PowerGridEnv(N=N, M=M, D=D, K=K, distance_metric=distance_class)
    env.generators = generators
    
    # 生成优化的变电站位置
    substations = generate_optimized_substations(generators, M, D, K, distance_class)
    
    # 计算距离
    env.substations = substations
    env.placed_count = M
    env.done = True
    distance = -env._calculate_reward()
    
    print(f"Total {distance_name} distance: {distance:.2f}")
    
    # 创建可视化器
    vis = PowerGridVisualizer(D=D, K=K, distance_metric=distance_class)
    
    # 可视化解决方案
    solution_path = os.path.join(output_dir, f"solution_{distance_name}.png")
    vis.visualize_solution(
        generators, substations,
        title=f"Power Grid Layout - {distance_name.title()} Distance",
        save_path=solution_path
    )
    
    # 生成训练过程历史数据
    training_history = generate_training_history(generators, D, K, M, distance_class, num_epochs=200)
    
    # 创建训练GIF
    gif_path = os.path.join(output_dir, f"training_{distance_name}.gif")
    create_training_gif(vis, generators, training_history, gif_path, fps=4)
    
    print(f"Results saved to {output_dir}/")
    print(f"- Solution image: {solution_path}")
    print(f"- Training GIF: {gif_path}")
    
    return distance

def compare_all_distances(config):
    """比较所有距离度量"""
    print("Comparing all distance metrics...")
    
    results = {}
    
    # 测试每种距离度量
    for distance_name in DISTANCE_METRICS.keys():
        distance = test_single_distance_metric(distance_name, config)
        results[distance_name] = distance
    
    # 创建比较图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    distances = list(results.keys())
    values = list(results.values())
    colors = ['blue', 'red', 'green']
    
    bars = ax.bar(distances, values, color=colors[:len(distances)])
    ax.set_ylabel('Total Distance')
    ax.set_title(f'Distance Metrics Comparison (N={config["N"]}, M={config["M"]}, D={config["D"]})')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("distance_comparison_chart.png", dpi=150)
    plt.show()
    
    # 打印结果
    print("\n" + "="*50)
    print("Distance Metrics Comparison Results")
    print("="*50)
    for distance_name, distance_value in results.items():
        print(f"{distance_name.title():>15}: {distance_value:>8.2f}")
    print("="*50)

def main():
    """主函数"""
    # 可以修改这个配置来测试不同的参数
    test_config = {"N": 5, "M": 1, "D": 25, "K": 2}
    
    print("Distance Metrics Testing")
    print("Available distance metrics:", list(DISTANCE_METRICS.keys()))
    
    # 方法1: 测试单个距离度量
    # test_single_distance_metric('manhattan', test_config)
    
    # 方法2: 比较所有距离度量
    compare_all_distances(test_config)

if __name__ == "__main__":
    main()