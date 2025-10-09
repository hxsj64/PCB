"""
运行深度强化学习实验的主脚本
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_drl_training():
    """运行DRL训练"""
    print("🚀 开始深度强化学习训练...")
    
    # 训练命令
    cmd = [
        sys.executable, "train_drl_power_grid.py",
        "--configs", "Small_Grid", "Medium_Grid", 
        "--distance_metrics", "manhattan", "euclidean",
        "--total_steps", "200000",
        "--use_gpu"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 训练完成!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 训练失败!")
        print(e.stderr)
        return False

def run_visualization():
    """运行可视化"""
    print("📊 开始生成可视化...")
    
    # 查找训练好的模型
    drl_experiments_dir = "drl_experiments"
    if not os.path.exists(drl_experiments_dir):
        print("错误: 未找到DRL实验目录")
        return False
    
    # 查找最新的模型文件
    model_paths = []
    for root, dirs, files in os.walk(drl_experiments_dir):
        for file in files:
            if file == "best_model.pth":
                model_paths.append(os.path.join(root, file))
    
    if not model_paths:
        print("错误: 未找到训练好的模型")
        return False
    
    # 使用最新的模型
    latest_model = max(model_paths, key=os.path.getmtime)
    print(f"使用模型: {latest_model}")
    
    # 运行可视化
    from drl_visualizer import visualize_trained_model
    
    try:
        visualize_trained_model(latest_model)
        print("✅ 可视化完成!")
        return True
    except Exception as e:
        print(f"❌ 可视化失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行深度强化学习电力网格优化实验")
    parser.add_argument("--mode", choices=["train", "visualize", "both"], 
                       default="both", help="运行模式")
    
    args = parser.parse_args()
    
    success = True
    
    if args.mode in ["train", "both"]:
        success &= run_drl_training()
    
    if args.mode in ["visualize", "both"] and success:
        success &= run_visualization()
    
    if success:
        print("\n🎉 所有实验完成!")
        print("📁 结果文件:")
        print("  - drl_experiments/ (训练结果)")
        print("  - drl_decision_analysis/ (决策分析)")
        print("  - drl_baseline_comparison/ (基准比较)")
    else:
        print("\n❌ 实验未完全成功，请检查错误信息")

if __name__ == "__main__":
    main()