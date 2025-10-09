"""
快速运行切比雪夫距离实验的脚本
"""
import os
from test_distance_metrics import test_single_distance_metric

def main():
    # 配置参数
    config = {"N": 5, "M": 1, "D": 25, "K": 2}
    
    print("Running Chebyshev Distance Experiment...")
    print(f"Configuration: N={config['N']}, M={config['M']}, D={config['D']}, K={config['K']}")
    
    # 运行切比雪夫距离实验
    distance = test_single_distance_metric('chebyshev', config)
    
    print(f"\nExperiment completed!")
    print(f"Chebyshev distance result: {distance:.2f}")
    print("Check 'results_chebyshev_distance' folder for visualizations.")

if __name__ == "__main__":
    main()
    
    