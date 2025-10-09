"""
修复后的完整实验运行脚本 - 生成网格布局GIF
"""
import os
import time
import json
import shutil
from datetime import datetime
import torch
import numpy as np
import random
from tqdm import tqdm
from matplotlib.patches import Rectangle

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# 导入模块
from distance_metrics import DISTANCE_METRICS
from drl_environment import PowerGridRLEnv
from attention_network import POMO_PowerGrid
from ppo_trainer import PPOTrainer
from fixed_drl_visualizer import DRLTrainingVisualizer  # 使用修复后的可视化器
from drl_inference import DRLInference

class FixedExperimentRunner:
    def __init__(self, output_dir="grid_layout_experiments"):
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 网格布局实验运行器初始化完成")
        print(f"📁 输出目录: {output_dir}")
        print(f"🔧 计算设备: {self.device}")
    
    def get_experiment_configs(self):
        """获取实验配置"""
        configs = [
            # 小规模快速测试
            {"N": 3, "M": 1, "D": 10, "K": 2, "name": "Tiny_Grid"},
            {"N": 5, "M": 1, "D": 10, "K": 2, "name": "Small_Grid"},
            {"N": 5, "M": 2, "D": 15, "K": 2, "name": "Medium_Grid"},
            {"N": 5, "M": 2, "D": 25, "K": 2, "name": "Large_Grid"},
        ]
        distance_metrics = ['manhattan']
        #distance_metrics = ['manhattan', 'euclidean', 'chebyshev']
        return configs, distance_metrics
    
    def train_drl_model(self, config, distance_name, training_steps=3000):
        """训练DRL模型"""
        print(f"\n🎯 训练: {config['name']} - {distance_name}")
        
        try:
            # 创建环境
            distance_metric = DISTANCE_METRICS[distance_name]
            env = PowerGridRLEnv(
                N=config['N'], M=config['M'], D=config['D'], K=config['K'],
                distance_metric=distance_metric, device=self.device
            )
            
            # 创建模型（小规模用于快速测试）
            model = POMO_PowerGrid(
                grid_size=config['D'], input_dim=4, d_model=64, n_head=4, n_layers=2, K=config['K']
            ).to(self.device)
            
            # 训练器
            trainer = PPOTrainer(model=model, env=env, lr=1e-3, device=self.device)
            
            # 输出目录
            exp_name = f"{config['name']}_{distance_name}"
            exp_dir = os.path.join(self.output_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            # 训练
            best_reward = trainer.train(
                total_steps=training_steps,
                steps_per_update=256,
                eval_interval=5,
                save_interval=10,
                output_dir=exp_dir
            )
            
            # 保存实验信息
            exp_info = {
                'config': config,
                'distance_metric': distance_name,
                'model_params': {'d_model': 64, 'n_head': 4, 'n_layers': 2},
                'best_reward': best_reward,
                'generators': env.generator_positions
            }
            
            with open(os.path.join(exp_dir, 'experiment_info.json'), 'w') as f:
                json.dump(exp_info, f, indent=2)
            
            result = {
                'config': config, 'distance_metric': distance_name, 'exp_dir': exp_dir,
                'best_reward': best_reward, 'model_path': os.path.join(exp_dir, 'best_model.pth'),
                'success': True, 'generators': env.generator_positions
            }
            
            print(f"✅ 训练完成! 最佳奖励: {best_reward:.2f}")
            return result
            
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
        return {'config': config, 'distance_metric': distance_name, 'error': str(e), 'success': False}
    
    def generate_grid_layout_gif(self, result):
        """生成网格布局GIF"""
        if not result['success']:
            return False
        
        print(f"🎬 生成网格布局GIF: {result['config']['name']} - {result['distance_metric']}")
        
        try:
            # 创建可视化器
            visualizer = DRLTrainingVisualizer(
                model_path=result['model_path'],
                env_config=result['config'],
                distance_metric=DISTANCE_METRICS[result['distance_metric']],
                device=self.device
            )
            
            # 生成逐步放置网格GIF
            gif_dir = os.path.join(result['exp_dir'], "grid_placement_gif")
            visualizer.generate_decision_making_visualization(gif_dir)
            
            # 复制GIF到主目录
            main_gif_path = os.path.join(self.output_dir, f"{result['config']['name']}_{result['distance_metric']}_grid_layout.gif")
            source_gif = os.path.join(gif_dir, "decision_making_process.gif")
            
            if os.path.exists(source_gif):
                shutil.copy2(source_gif, main_gif_path)
                print(f"✅ 网格布局GIF已保存: {main_gif_path}")
                return True
            else:
                print(f"⚠️ 未找到源GIF文件")
                return False
                
        except Exception as e:
            print(f"❌ GIF生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_static_comparison_image(self, result):
        """创建静态对比图片"""
        if not result['success']:
            return False
        
        try:
            print(f"📊 生成对比图片: {result['config']['name']} - {result['distance_metric']}")
            
            # 使用推理器求解
            inference = DRLInference(model_path=result['model_path'], device=self.device)
            solution = inference.solve(
                generators=result['generators'],
                distance_metric_name=result['distance_metric'],
                deterministic=True
            )
            
            if solution['success']:
                self._create_three_method_comparison(result, solution)
                return True
            else:
                print(f"⚠️ DRL求解失败")
                return False
                
        except Exception as e:
            print(f"❌ 对比图片生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_three_method_comparison(self, result, drl_solution):
        """创建三种方法对比图"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        config = result['config']
        distance_metric = DISTANCE_METRICS[result['distance_metric']]
        
        # 运行基准算法
        kmeans_subs, kmeans_distance = self._run_kmeans_baseline(result)
        random_subs, random_distance = self._run_random_baseline(result)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Method Comparison: {config['name']} - {result['distance_metric'].title()} Distance", 
                    fontsize=16, fontweight='bold')
        
        methods = [
            ("DRL Algorithm", drl_solution['substations'], drl_solution['total_distance'], 'green'),
            ("K-means Baseline", kmeans_subs, kmeans_distance, 'blue'),
            ("Random Baseline", random_subs, random_distance, 'red')
        ]
        
        for i, (method_name, substations, distance, color) in enumerate(methods):
            ax = axes[i]
            self._draw_grid_solution(ax, result['generators'], substations, config, distance_metric, 
                                   method_name, distance, color)
        
        # 保存图片
        comparison_path = os.path.join(self.output_dir, f"{config['name']}_{result['distance_metric']}_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 对比图已保存: {comparison_path}")
    
    def _draw_grid_solution(self, ax, generators, substations, config, distance_metric, 
                           method_name, distance, color):
        """绘制单个方法的网格解决方案"""
        D, K = config['D'], config['K']
        
        ax.set_xlim(0, D)
        ax.set_ylim(0, D)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f"{method_name}\nDistance: {distance:.2f}")
        
        # 绘制发电机
        for j, (gx, gy) in enumerate(generators):
            rect = Rectangle((gx, gy), K, K, color='red', alpha=0.8, edgecolor='darkred')
            ax.add_patch(rect)
            ax.text(gx + K/2, gy + K/2, f'G{j+1}', 
                   ha='center', va='center', fontweight='bold', color='white')
        
        # 绘制变电站
        for j, (sx, sy) in enumerate(substations):
            rect = Rectangle((sx, sy), K, K, color=color, alpha=0.8, edgecolor='black')
            ax.add_patch(rect)
            ax.text(sx + K/2, sy + K/2, f'S{j+1}', 
                   ha='center', va='center', fontweight='bold', color='white')
        
        # 绘制连接线
        self._draw_connections(ax, generators, substations, distance_metric, K)
    
    def _draw_connections(self, ax, generators, substations, distance_metric, K):
        """绘制连接线"""
        if not substations:
            return
        
        for gx, gy in generators:
            min_dist = float('inf')
            closest_sub = None
            
            for sx, sy in substations:
                dist = distance_metric.calculate((gx, gy), (sx, sy))
                if dist < min_dist:
                    min_dist = dist
                    closest_sub = (sx, sy)
            
            if closest_sub:
                sx, sy = closest_sub
                
                if distance_metric.name().lower() == "manhattan":
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
                    line_style = '--' if distance_metric.name().lower() == "chebyshev" else '-'
                    ax.plot([gx + K/2, sx + K/2], [gy + K/2, sy + K/2], 
                           'gray', linestyle=line_style, linewidth=2, alpha=0.7)
    
    def _run_kmeans_baseline(self, result):
        """运行K-means基准"""
        try:
            from sklearn.cluster import KMeans
            
            config = result['config']
            generators = np.array(result['generators'])
            
            if len(generators) < config['M']:
                return [], float('inf')
            
            kmeans = KMeans(n_clusters=config['M'], random_state=42, n_init=10)
            kmeans.fit(generators)
            
            # 将聚类中心调整到有效位置
            substations = []
            for center in kmeans.cluster_centers_:
                x, y = center
                x = max(0, min(config['D'] - config['K'], int(round(x))))
                y = max(0, min(config['D'] - config['K'], int(round(y))))
                substations.append((x, y))
            
            # 计算距离
            distance_metric = DISTANCE_METRICS[result['distance_metric']]
            total_distance = 0
            for gx, gy in result['generators']:
                min_dist = float('inf')
                for sx, sy in substations:
                    dist = distance_metric.calculate((gx, gy), (sx, sy))
                    if dist < min_dist:
                        min_dist = dist
                total_distance += min_dist
            
            return substations, total_distance
            
        except Exception as e:
            print(f"⚠️ K-means基准失败: {str(e)}")
            return [], float('inf')
    
    def _run_random_baseline(self, result):
        """运行随机基准"""
        try:
            config = result['config']
            
            # 生成有效位置列表
            valid_positions = []
            for x in range(config['D'] - config['K'] + 1):
                for y in range(config['D'] - config['K'] + 1):
                    # 检查是否与发电机重叠
                    overlap = False
                    for gx, gy in result['generators']:
                        if (abs(x - gx) < config['K'] and abs(y - gy) < config['K']):
                            overlap = True
                            break
                    if not overlap:
                        valid_positions.append((x, y))
            
            # 随机选择位置
            if len(valid_positions) >= config['M']:
                substations = random.sample(valid_positions, config['M'])
            else:
                substations = valid_positions  # 如果位置不够，用所有可用位置
            
            # 计算距离
            distance_metric = DISTANCE_METRICS[result['distance_metric']]
            total_distance = 0
            for gx, gy in result['generators']:
                min_dist = float('inf')
                for sx, sy in substations:
                    dist = distance_metric.calculate((gx, gy), (sx, sy))
                    if dist < min_dist:
                        min_dist = dist
                total_distance += min_dist
            
            return substations, total_distance
            
        except Exception as e:
            print(f"⚠️ 随机基准失败: {str(e)}")
            return [], float('inf')
    
    def generate_summary_report(self):
        """生成总结报告"""
        print(f"\n📋 生成总结报告...")
        
        successful_results = [r for r in self.results if r['success']]
        failed_results = [r for r in self.results if not r['success']]
        
        report_path = os.path.join(self.output_dir, "GRID_LAYOUT_EXPERIMENT_SUMMARY.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 电力网格布局优化实验报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 实验概述\n\n")
            f.write("本实验展示了深度强化学习(DRL)算法在电力网格变电站选址问题上的应用。\n")
            f.write("每个实验都生成了**网格布局GIF**，显示算法逐步放置变电站的决策过程。\n\n")
            
            f.write(f"- 总实验数: {len(self.results)}\n")
            f.write(f"- 成功实验: {len(successful_results)}\n")
            f.write(f"- 失败实验: {len(failed_results)}\n")
            f.write(f"- 成功率: {len(successful_results)/len(self.results)*100:.1f}%\n\n")
            
            if successful_results:
                f.write("## 🎬 网格布局动画结果\n\n")
                f.write("| 配置 | 距离度量 | 网格大小 | 网格布局GIF | 方法对比图 |\n")
                f.write("|------|----------|----------|-------------|------------|\n")
                
                for result in successful_results:
                    config = result['config']
                    gif_file = f"{config['name']}_{result['distance_metric']}_grid_layout.gif"
                    comparison_file = f"{config['name']}_{result['distance_metric']}_comparison.png"
                    
                    f.write(f"| {config['name']} | {result['distance_metric'].title()} | ")
                    f.write(f"{config['D']}×{config['D']} | [🎬 网格布局动画]({gif_file}) | ")
                    f.write(f"[📊 三方法对比]({comparison_file}) |\n")
            
            f.write("\n## 📋 配置说明\n\n")
            for result in successful_results:
                config = result['config']
                f.write(f"### {config['name']}\n")
                f.write(f"- 网格大小: {config['D']}×{config['D']}\n")
                f.write(f"- 发电机数量: {config['N']}\n")
                f.write(f"- 变电站数量: {config['M']}\n")
                f.write(f"- 设备尺寸: {config['K']}×{config['K']}\n\n")
            
            f.write("## 🎯 GIF内容说明\n\n")
            f.write("每个网格布局GIF包含以下内容：\n\n")
            f.write("- **左侧网格图**: 显示发电机(红色)和逐步放置的变电站(蓝色)\n")
            f.write("- **连接线**: 根据距离度量显示不同样式的连接\n")
            f.write("  - 曼哈顿距离: L形路径(横线+竖线)\n")
            f.write("  - 欧几里得距离: 直线连接\n")
            f.write("  - 切比雪夫距离: 虚线连接\n")
            f.write("- **右侧信息**: 当前状态统计和距离计算\n")
            f.write("- **动态标记**: 🆕 标记当前新放置的变电站\n\n")
            
            f.write("## 📊 对比图说明\n\n")
            f.write("每个对比图包含三种方法的结果：\n\n")
            f.write("1. **DRL算法** (绿色变电站): 深度强化学习优化结果\n")
            f.write("2. **K-means基线** (蓝色变电站): 传统聚类算法结果\n")
            f.write("3. **随机基线** (红色变电站): 随机放置算法结果\n\n")
            
            if failed_results:
                f.write("## ❌ 失败实验\n\n")
                for result in failed_results:
                    config = result['config']
                    f.write(f"- {config['name']} ({result['distance_metric']}): {result.get('error', '未知错误')}\n")
            
            f.write("\n## 🔗 文件索引\n\n")
            f.write("### 🎬 网格布局动画\n")
            for result in successful_results:
                config = result['config']
                f.write(f"- `{config['name']}_{result['distance_metric']}_grid_layout.gif`\n")
            
            f.write("\n### 📊 方法对比图\n")
            for result in successful_results:
                config = result['config']
                f.write(f"- `{config['name']}_{result['distance_metric']}_comparison.png`\n")
            
            f.write("\n---\n\n")
            f.write("*此报告由自动化实验系统生成*\n")
        
        print(f"📋 总结报告已保存: {report_path}")
    
    def run_all_experiments(self):
        """运行所有实验"""
        configs, distance_metrics = self.get_experiment_configs()
        
        print(f"\n🎯 开始网格布局实验")
        print(f"📊 配置数量: {len(configs)}")
        print(f"📏 距离度量: {distance_metrics}")
        print(f"🔢 总实验数: {len(configs) * len(distance_metrics)}")
        
        total_experiments = len(configs) * len(distance_metrics)
        start_time = time.time()
        
        # 运行所有实验
        for i, config in enumerate(configs):
            for j, distance_name in enumerate(distance_metrics):
                experiment_num = i * len(distance_metrics) + j + 1
                print(f"\n🔄 进度: [{experiment_num}/{total_experiments}]")
                
                # 训练DRL模型
                result = self.train_drl_model(config, distance_name, training_steps=3000)
                self.results.append(result)
                
                if result['success']:
                    # 生成网格布局GIF
                    self.generate_grid_layout_gif(result)
                    
                    # 创建静态对比图片
                    self.create_static_comparison_image(result)
        
        # 生成总结报告
        self.generate_summary_report()
        
        # 计算总耗时
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print(f"\n🎉 所有实验完成!")
        print(f"⏱️ 总耗时: {hours}小时{minutes}分钟")
        print(f"📁 结果保存在: {self.output_dir}")
        print(f"📋 查看报告: {os.path.join(self.output_dir, 'GRID_LAYOUT_EXPERIMENT_SUMMARY.md')}")

def main():
    """主函数"""
    print("🚀 电力网格布局优化 - 网格可视化实验")
    print("=" * 60)
    
    # 创建实验运行器
    runner = FixedExperimentRunner("grid_layout_experiments")
    
    # 运行所有实验
    runner.run_all_experiments()
    
    print("\n" + "=" * 60)
    print("🎊 网格布局实验完成！")
    print("🎬 查看生成的GIF文件了解算法决策过程")
    print("📊 查看对比图片了解性能差异")
    print("=" * 60)

if __name__ == "__main__":
    main()