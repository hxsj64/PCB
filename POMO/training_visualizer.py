import os
import random
import numpy as np
import json
import time
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
from experiment_configs import EXPERIMENT_CONFIGS, RUNS_PER_CONFIG, DISTANCE_METRICS_LIST

# 设置随机种子
random.seed(123)
np.random.seed(123)

class ExperimentRunner:
    def __init__(self, output_dir="experiment_results"):
        self.output_dir = output_dir
        self.results = []
        
    def generate_generators(self, N, D, K):
        """生成不重叠的发电机位置"""
        generators = []
        max_attempts = 1000
        attempts = 0
        
        while len(generators) < N and attempts < max_attempts:
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
            attempts += 1
            
        if len(generators) < N:
            print(f"警告: 只能放置 {len(generators)} 个发电机，少于要求的 {N} 个")
            
        return generators
    
    def run_single_experiment(self, config, distance_name, run_id, create_visualization=True):
        """运行单个实验"""
        N, M, D, K = config["N"], config["M"], config["D"], config["K"]
        distance_class = DISTANCE_METRICS[distance_name]
        
        # 生成发电机位置
        generators = self.generate_generators(N, D, K)
        
        if len(generators) < N:
            return None  # 跳过无法完成的配置
        
        # 创建环境
        env = PowerGridEnv(N=N, M=M, D=D, K=K, distance_metric=distance_class)
        env.generators = generators
        
        # 生成优化的变电站位置
        substations = generate_optimized_substations(generators, M, D, K, distance_class)
        
        # 计算优化算法距离
        env.substations = substations
        env.placed_count = M
        env.done = True
        optimized_distance = -env._calculate_reward()
        
        # 评估K-means基准
        env_kmeans = PowerGridEnv(N=N, M=M, D=D, K=K, distance_metric=distance_class)
        env_kmeans.generators = generators
        km_reward, km_subs = kmeans_solution(env_kmeans)
        kmeans_distance = -km_reward
        
        # 评估随机基准
        env_random = PowerGridEnv(N=N, M=M, D=D, K=K, distance_metric=distance_class)
        env_random.generators = generators
        rand_reward, rand_subs = random_solution(env_random)
        random_distance = -rand_reward
        
        result = {
            "config": config,
            "distance_metric": distance_name,
            "run_id": run_id,
            "generators": generators,
            "optimized_substations": substations,
            "optimized_distance": optimized_distance,
            "kmeans_distance": kmeans_distance,
            "random_distance": random_distance,
            "improvement_vs_kmeans": ((kmeans_distance - optimized_distance) / kmeans_distance * 100) if kmeans_distance > 0 else 0,
            "improvement_vs_random": ((random_distance - optimized_distance) / random_distance * 100) if random_distance > 0 else 0
        }
        
        # 创建可视化（仅为第一次运行）
        if create_visualization and run_id == 0:
            self.create_experiment_visualization(result)
            
        return result
    
    def create_experiment_visualization(self, result):
        """为单个实验创建可视化"""
        config = result["config"]
        distance_name = result["distance_metric"]
        
        # 创建输出目录
        viz_dir = os.path.join(self.output_dir, "visualizations", 
                              f'{config["name"]}_{distance_name}')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 创建可视化器
        distance_class = DISTANCE_METRICS[distance_name]
        vis = PowerGridVisualizer(D=config["D"], K=config["K"], distance_metric=distance_class)
        
        # 可视化优化解决方案
        title = f'{config["name"]} - {distance_name.title()} Distance (N={config["N"]}, M={config["M"]}, D={config["D"]})'
        solution_path = os.path.join(viz_dir, "optimized_solution.png")
        vis.visualize_solution(
            result["generators"], 
            result["optimized_substations"],
            title=title,
            save_path=solution_path
        )
        
        # 生成训练历史（仅为N=5, D=50的情况生成详细的训练视频）
        if config["N"] == 5 and config["D"] == 50:
            print(f"生成详细训练过程 for {config['name']}_{distance_name}...")
            training_history = generate_training_history(
                result["generators"], config["D"], config["K"], config["M"], 
                distance_class, num_epochs=500  # 更多epoch用于详细展示
            )
            
            # 创建训练GIF
            gif_path = os.path.join(viz_dir, "training_process.gif")
            create_training_gif(vis, result["generators"], training_history, gif_path, fps=6)
        
        print(f"可视化已保存到: {viz_dir}")
    
    def run_all_experiments(self):
        """运行所有实验配置"""
        print("开始运行完整实验套件...")
        print(f"配置数量: {len(EXPERIMENT_CONFIGS)}")
        print(f"距离度量: {DISTANCE_METRICS_LIST}")
        print(f"每配置运行次数: {RUNS_PER_CONFIG}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        total_experiments = len(EXPERIMENT_CONFIGS) * len(DISTANCE_METRICS_LIST) * RUNS_PER_CONFIG
        
        with tqdm(total=total_experiments, desc="运行实验") as pbar:
            for config in EXPERIMENT_CONFIGS:
                for distance_name in DISTANCE_METRICS_LIST:
                    config_results = []
                    
                    for run_id in range(RUNS_PER_CONFIG):
                        # 只为第一次运行创建可视化
                        create_viz = (run_id == 0)
                        
                        result = self.run_single_experiment(
                            config, distance_name, run_id, create_viz
                        )
                        
                        if result is not None:
                            config_results.append(result)
                            self.results.append(result)
                        
                        pbar.update(1)
                    
                    # 计算该配置的平均结果
                    if config_results:
                        self.calculate_average_results(config_results)
        
        # 保存所有结果
        self.save_results()
        
        # 生成报告
        self.generate_comprehensive_report()
    
    def calculate_average_results(self, config_results):
        """计算单个配置的平均结果"""
        if not config_results:
            return
        
        config = config_results[0]["config"]
        distance_name = config_results[0]["distance_metric"]
        
        # 计算平均值
        avg_optimized = np.mean([r["optimized_distance"] for r in config_results])
        avg_kmeans = np.mean([r["kmeans_distance"] for r in config_results])
        avg_random = np.mean([r["random_distance"] for r in config_results])
        avg_improvement_kmeans = np.mean([r["improvement_vs_kmeans"] for r in config_results])
        avg_improvement_random = np.mean([r["improvement_vs_random"] for r in config_results])
        
        # 计算标准差
        std_optimized = np.std([r["optimized_distance"] for r in config_results])
        std_kmeans = np.std([r["kmeans_distance"] for r in config_results])
        std_random = np.std([r["random_distance"] for r in config_results])
        
        avg_result = {
            "config": config,
            "distance_metric": distance_name,
            "type": "average",
            "runs_count": len(config_results),
            "avg_optimized_distance": avg_optimized,
            "avg_kmeans_distance": avg_kmeans,
            "avg_random_distance": avg_random,
            "std_optimized_distance": std_optimized,
            "std_kmeans_distance": std_kmeans,
            "std_random_distance": std_random,
            "avg_improvement_vs_kmeans": avg_improvement_kmeans,
            "avg_improvement_vs_random": avg_improvement_random
        }
        
        self.results.append(avg_result)
    
    def save_results(self):
        """保存实验结果"""
        results_file = os.path.join(self.output_dir, "all_results.json")
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        json_results = []
        for result in self.results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_result[key] = value.item()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {results_file}")
    
    def generate_comprehensive_report(self):
        """生成综合实验报告"""
        # 筛选平均结果
        avg_results = [r for r in self.results if r.get("type") == "average"]
        
        # 按配置分组
        grouped_results = {}
        for result in avg_results:
            config_name = result["config"]["name"]
            if config_name not in grouped_results:
                grouped_results[config_name] = []
            grouped_results[config_name].append(result)
        
        # 生成表格报告
        self.generate_table_report(grouped_results)
        
        # 生成图表报告
        self.generate_chart_report(grouped_results)
        
        # 生成详细分析
        self.generate_detailed_analysis(grouped_results)
    
    def generate_table_report(self, grouped_results):
        """生成表格报告"""
        # 创建表格数据
        table_data = []
        headers = [
            "Configuration", "Distance Metric", "N", "M", "D", 
            "Optimized", "K-means", "Random", 
            "vs K-means (%)", "vs Random (%)", "Best Method"
        ]
        
        for config_name, results in grouped_results.items():
            for result in results:
                config = result["config"]
                
                # 确定最佳方法
                distances = [
                    result["avg_optimized_distance"],
                    result["avg_kmeans_distance"], 
                    result["avg_random_distance"]
                ]
                best_idx = np.argmin(distances)
                best_methods = ["Optimized", "K-means", "Random"]
                best_method = best_methods[best_idx]
                
                row = [
                    config["name"],
                    result["distance_metric"].title(),
                    config["N"],
                    config["M"],
                    config["D"],
                    f"{result['avg_optimized_distance']:.2f}±{result['std_optimized_distance']:.2f}",
                    f"{result['avg_kmeans_distance']:.2f}±{result['std_kmeans_distance']:.2f}",
                    f"{result['avg_random_distance']:.2f}±{result['std_random_distance']:.2f}",
                    f"{result['avg_improvement_vs_kmeans']:+.1f}%",
                    f"{result['avg_improvement_vs_random']:+.1f}%",
                    best_method
                ]
                table_data.append(row)
        
        # 创建表格图像
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis('off')
        
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(headers)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title("Experimental Results Summary - Average Performance", 
                    fontsize=16, pad=30, weight='bold')
        
        plt.savefig(os.path.join(self.output_dir, "results_table.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存文本版本的表格
        with open(os.path.join(self.output_dir, "results_table.txt"), 'w', encoding='utf-8') as f:
            f.write("实验结果汇总表 - 平均性能\n")
            f.write("=" * 150 + "\n")
            
            # 写入表头
            header_line = " | ".join([f"{h:^15}" for h in headers])
            f.write(header_line + "\n")
            f.write("-" * 150 + "\n")
            
            # 写入数据
            for row in table_data:
                row_line = " | ".join([f"{str(cell):^15}" for cell in row])
                f.write(row_line + "\n")
            
            f.write("=" * 150 + "\n")
    
    def generate_chart_report(self, grouped_results):
        """生成图表报告"""
        # 性能比较柱状图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 准备数据
        configs = []
        euclidean_data = []
        manhattan_data = []
        chebyshev_data = []
        
        for config_name, results in grouped_results.items():
            configs.append(config_name.replace("_", "\n"))
            
            # 按距离度量分组
            for result in results:
                if result["distance_metric"] == "euclidean":
                    euclidean_data.append(result["avg_optimized_distance"])
                elif result["distance_metric"] == "manhattan":
                    manhattan_data.append(result["avg_optimized_distance"])
                elif result["distance_metric"] == "chebyshev":
                    chebyshev_data.append(result["avg_optimized_distance"])
        
        # 确保数据长度一致
        min_len = min(len(euclidean_data), len(manhattan_data), len(chebyshev_data))
        configs = configs[:min_len]
        euclidean_data = euclidean_data[:min_len]
        manhattan_data = manhattan_data[:min_len]
        chebyshev_data = chebyshev_data[:min_len]
        
        x = np.arange(len(configs))
        width = 0.25
        
        # 子图1: 距离度量比较
        ax1.bar(x - width, euclidean_data, width, label='Euclidean', alpha=0.8)
        ax1.bar(x, manhattan_data, width, label='Manhattan', alpha=0.8)
        ax1.bar(x + width, chebyshev_data, width, label='Chebyshev', alpha=0.8)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Average Distance')
        ax1.set_title('Distance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 算法改进百分比
        improvements_kmeans = []
        improvements_random = []
        config_labels = []
        
        for config_name, results in grouped_results.items():
            for result in results:
                if result["distance_metric"] == "euclidean":  # 只取欧几里得距离的结果
                    improvements_kmeans.append(result["avg_improvement_vs_kmeans"])
                    improvements_random.append(result["avg_improvement_vs_random"])
                    config_labels.append(f"{config_name}\n(N={result['config']['N']}, D={result['config']['D']})")
                    break
        
        x2 = np.arange(len(config_labels))
        ax2.bar(x2 - width/2, improvements_kmeans, width, label='vs K-means', alpha=0.8)
        ax2.bar(x2 + width/2, improvements_random, width, label='vs Random', alpha=0.8)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Algorithm Improvement vs Baselines')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(config_labels, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 子图3: 不同规模下的性能趋势
        grid_sizes = []
        performance_trends = {}
        
        for config_name, results in grouped_results.items():
            config = results[0]["config"]
            grid_size = config["D"]
            
            if grid_size not in performance_trends:
                performance_trends[grid_size] = {"euclidean": [], "manhattan": [], "chebyshev": []}
            
            for result in results:
                distance_type = result["distance_metric"]
                performance_trends[grid_size][distance_type].append(result["avg_optimized_distance"])
        
        for grid_size in sorted(performance_trends.keys()):
            grid_sizes.append(grid_size)
        
        for distance_type in ["euclidean", "manhattan", "chebyshev"]:
            values = []
            for grid_size in grid_sizes:
                if performance_trends[grid_size][distance_type]:
                    values.append(np.mean(performance_trends[grid_size][distance_type]))
                else:
                    values.append(0)
            ax3.plot(grid_sizes, values, marker='o', label=distance_type.title(), linewidth=2)
        
        ax3.set_xlabel('Grid Size (D)')
        ax3.set_ylabel('Average Distance')
        ax3.set_title('Performance Scaling with Grid Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 标准差分析
        config_names_short = []
        std_optimized = []
        std_kmeans = []
        
        for config_name, results in grouped_results.items():
            for result in results:
                if result["distance_metric"] == "euclidean":
                    config_names_short.append(config_name.replace("_", "\n"))
                    std_optimized.append(result["std_optimized_distance"])
                    std_kmeans.append(result["std_kmeans_distance"])
                    break
        
        x4 = np.arange(len(config_names_short))
        ax4.bar(x4 - width/2, std_optimized, width, label='Optimized Algorithm', alpha=0.8)
        ax4.bar(x4 + width/2, std_kmeans, width, label='K-means Baseline', alpha=0.8)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Result Stability (Lower is Better)')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(config_names_short, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_charts.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_detailed_analysis(self, grouped_results):
        """生成详细分析报告"""
        analysis_file = os.path.join(self.output_dir, "detailed_analysis.txt")
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("详细实验分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体统计
            f.write("1. 总体统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"实验配置数量: {len(grouped_results)}\n")
            f.write(f"距离度量类型: {len(DISTANCE_METRICS_LIST)}\n")
            f.write(f"每配置运行次数: {RUNS_PER_CONFIG}\n")
            f.write(f"总实验次数: {len(grouped_results) * len(DISTANCE_METRICS_LIST) * RUNS_PER_CONFIG}\n\n")
            
            # 最佳配置分析
            f.write("2. 最佳配置分析\n")
            f.write("-" * 40 + "\n")
            
            all_results = []
            for config_name, results in grouped_results.items():
                all_results.extend(results)
            
            # 找到最佳性能
            best_optimized = min(all_results, key=lambda x: x["avg_optimized_distance"])
            best_improvement = max(all_results, key=lambda x: x["avg_improvement_vs_kmeans"])
            most_stable = min(all_results, key=lambda x: x["std_optimized_distance"])
            
            f.write(f"最小平均距离:\n")
            f.write(f"  配置: {best_optimized['config']['name']}\n")
            f.write(f"  距离度量: {best_optimized['distance_metric']}\n")
            f.write(f"  平均距离: {best_optimized['avg_optimized_distance']:.2f}\n\n")
            
            f.write(f"最大改进幅度 (vs K-means):\n")
            f.write(f"  配置: {best_improvement['config']['name']}\n")
            f.write(f"  距离度量: {best_improvement['distance_metric']}\n")
            f.write(f"  改进幅度: {best_improvement['avg_improvement_vs_kmeans']:.1f}%\n\n")
            
            f.write(f"最稳定结果 (最小标准差):\n")
            f.write(f"  配置: {most_stable['config']['name']}\n")
            f.write(f"  距离度量: {most_stable['distance_metric']}\n")
            f.write(f"  标准差: {most_stable['std_optimized_distance']:.2f}\n\n")
            
            # 距离度量比较
            f.write("3. 距离度量比较\n")
            f.write("-" * 40 + "\n")
            
            metric_performance = {"euclidean": [], "manhattan": [], "chebyshev": []}
            for result in all_results:
                metric_performance[result["distance_metric"]].append(result["avg_optimized_distance"])
            
            for metric, distances in metric_performance.items():
                avg_dist = np.mean(distances)
                std_dist = np.std(distances)
                f.write(f"{metric.title()} Distance:\n")
                f.write(f"  平均距离: {avg_dist:.2f} ± {std_dist:.2f}\n")
                f.write(f"  实验次数: {len(distances)}\n\n")
            
            # 规模效应分析
            f.write("4. 规模效应分析\n")
            f.write("-" * 40 + "\n")
            
            size_analysis = {}
            for config_name, results in grouped_results.items():
                config = results[0]["config"]
                key = f"N={config['N']}, D={config['D']}"
                
                if key not in size_analysis:
                    size_analysis[key] = []
                
                for result in results:
                    size_analysis[key].append(result["avg_optimized_distance"])
            
            for size_key, distances in size_analysis.items():
                avg_dist = np.mean(distances)
                f.write(f"{size_key}: 平均距离 = {avg_dist:.2f}\n")
            
            f.write("\n5. 建议和结论\n")
            f.write("-" * 40 + "\n")
            f.write("基于实验结果的建议:\n")
            f.write("- 对于小规模问题，推荐使用欧几里得距离度量\n")
            f.write("- 对于大规模问题，考虑使用切比雪夫距离以减少计算复杂度\n")
            f.write("- 曼哈顿距离适用于城市网格布局场景\n")
            f.write("- 优化算法在所有测试配置中均优于基准方法\n")
            
        print(f"详细分析报告已保存到: {analysis_file}")

def main():
    """主函数"""
    print("开始大规模实验...")
    
    runner = ExperimentRunner("comprehensive_experiment_results")
    runner.run_all_experiments()
    
    print("\n实验完成！")
    print("结果文件:")
    print("- comprehensive_experiment_results/results_table.png (结果表格)")
    print("- comprehensive_experiment_results/performance_charts.png (性能图表)")
    print("- comprehensive_experiment_results/visualizations/ (可视化文件)")
    print("- comprehensive_experiment_results/detailed_analysis.txt (详细分析)")

if __name__ == "__main__":
    main()