"""
实验配置定义
"""

# 实验配置 - 6组数据
EXPERIMENT_CONFIGS = [
    # 小规模测试
    {"N": 5, "M": 1, "D": 10, "K": 2, "name": "Small_Grid_Few_Generators"},
    {"N": 5, "M": 2, "D": 25, "K": 2, "name": "Medium_Grid_Few_Generators"},
    {"N": 5, "M": 2, "D": 50, "K": 2, "name": "Large_Grid_Few_Generators"},
    
    # 大规模测试 (跳过N=10, D=10因为空间限制)
    {"N": 10, "M": 3, "D": 25, "K": 2, "name": "Medium_Grid_Many_Generators"},
    {"N": 10, "M": 4, "D": 50, "K": 2, "name": "Large_Grid_Many_Generators"},
    
    # 额外配置
    {"N": 8, "M": 2, "D": 30, "K": 2, "name": "Mixed_Scale_Test"}
]

# 每个配置运行的次数（用于计算平均值）
RUNS_PER_CONFIG = 5

# 距离度量类型
DISTANCE_METRICS_LIST = ['euclidean', 'manhattan', 'chebyshev']