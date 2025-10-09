"""
快速测试网格布局GIF生成
"""
import os
from fixed_run_all_experiments_with_gif import FixedExperimentRunner

def quick_grid_test():
    """快速测试网格布局可视化"""
    print("🚀 快速网格布局测试")
    
    runner = FixedExperimentRunner("quick_grid_test")
    
    # 只运行一个小例子
    test_configs = [
        {"N": 3, "M": 1, "D": 10, "K": 2, "name": "Quick_Test"}
    ]
    distance_metrics = ['manhattan']
    
    # 临时替换配置
    runner.get_experiment_configs = lambda: (test_configs, distance_metrics)
    
    # 运行实验
    runner.run_all_experiments()
    
    print("✅ 快速测试完成!")
    print("📁 查看 quick_grid_test/ 目录")
    print("🎬 应该有一个网格布局GIF文件")

if __name__ == "__main__":
    quick_grid_test()