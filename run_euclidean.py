"""
快速运行欧式距离实验的脚本
"""
import os
from test_distance_metrics import test_single_distance_metric

def main():
    # 配置参数
    config = {"N": 5, "M": 1, "D": 25, "K": 2}
    
    print("Running Euclidean Distance Experiment...")
    print(f"Configuration: N={config['N']}, M={config['M']}, D={config['D']}, K={config['K']}")
    
    # 运行欧式距离实验
    distance = test_single_distance_metric('euclidean', config)
    
    print(f"\nExperiment completed!")
    print(f"Euclidean distance result: {distance:.2f}")
    print("Check 'results_euclidean_distance' folder for visualizations.")

if __name__ == "__main__":
    main()