"""
一键运行完整实验的主脚本
"""
import os
import time
from experiment_runner import main as run_experiments
from training_visualizer import main as generate_training_videos

def main():
    """运行完整的实验流程"""
    print("=" * 80)
    print("电力网格优化 - 完整实验套件")
    print("=" * 80)
    
    start_time = time.time()
    
    # 步骤1: 运行主要实验
    print("\n第1步: 运行主要实验（6组配置 × 3种距离度量）...")
    print("-" * 50)
    run_experiments()
    
    # 步骤2: 生成N=5, D=50的详细训练可视化
    print("\n第2步: 生成N=5, D=50的详细训练可视化...")
    print("-" * 50)
    generate_training_videos()
    
    # 计算总时间
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 80)
    print("实验完成总结")
    print("=" * 80)
    print(f"总耗时: {minutes}分{seconds}秒")
    print("\n生成的文件:")
    print("📊 实验结果:")
    print("  - comprehensive_experiment_results/results_table.png")
    print("  - comprehensive_experiment_results/performance_charts.png")
    print("  - comprehensive_experiment_results/detailed_analysis.txt")
    print("\n🎬 训练可视化:")
    print("  - training_visualization_N5_D50/euclidean/training_process_euclidean.gif")
    print("  - training_visualization_N5_D50/manhattan/training_process_manhattan.gif")
    print("  - training_visualization_N5_D50/chebyshev/training_process_chebyshev.gif")
    print("  - training_visualization_N5_D50/comprehensive_comparison.gif")
    print("\n📸 解决方案图片:")
    print("  - comprehensive_experiment_results/visualizations/*/optimized_solution.png")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()