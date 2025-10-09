import numpy as np

class DistanceMetric:
    """距离度量基类"""
    
    @staticmethod
    def calculate(p1, p2):
        """计算两点间距离"""
        raise NotImplementedError
    
    @staticmethod
    def name():
        """返回距离名称"""
        raise NotImplementedError

class EuclideanDistance(DistanceMetric):
    """欧式距离"""
    
    @staticmethod
    def calculate(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    @staticmethod
    def name():
        return "Euclidean"

class ManhattanDistance(DistanceMetric):
    """曼哈顿距离"""
    
    @staticmethod
    def calculate(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)
    
    @staticmethod
    def name():
        return "Manhattan"

class ChebyshevDistance(DistanceMetric):
    """切比雪夫距离"""
    
    @staticmethod
    def calculate(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return max(abs(x1 - x2), abs(y1 - y2))
    
    @staticmethod
    def name():
        return "Chebyshev"

# 可用的距离度量字典
DISTANCE_METRICS = {
    'euclidean': EuclideanDistance,
    'manhattan': ManhattanDistance,
    'chebyshev': ChebyshevDistance
}