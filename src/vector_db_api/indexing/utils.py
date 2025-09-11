import math
from typing import List, Tuple, Optional, Callable

def dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))

def norm(a: List[float]) -> float:
    return math.sqrt(sum(ai ** 2 for ai in a))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot(a, b) / (norm_a * norm_b)

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two vectors"""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def euclidean_similarity(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean similarity (1 / (1 + distance))"""
    distance = euclidean_distance(a, b)
    return 1.0 / (1.0 + distance)

def dot_product_similarity(a: List[float], b: List[float]) -> float:
    """Calculate dot product similarity"""
    return dot(a, b)

def normalize(a: List[float]) -> Optional[List[float]]:
    norm_a = norm(a)
    if norm_a == 0:
        return None
    inv = 1.0 / norm_a
    return [ai * inv for ai in a]

def argmax_idx(xs: List[float]) -> int:
    mi, mv = 0, xs[0]
    for i, x in enumerate(xs):
        if x > mv:
            mi, mv = i, x
    return mi

def get_similarity_function(metric: str) -> Callable[[List[float], List[float]], float]:
    """Get the appropriate similarity function based on metric"""
    similarity_functions = {
        "cosine": cosine_similarity,
        "euclidean": euclidean_similarity,
        "dot_product": dot_product_similarity,
    }
    
    if metric not in similarity_functions:
        raise ValueError(f"Unsupported similarity metric: {metric}. Supported metrics: {list(similarity_functions.keys())}")
    
    return similarity_functions[metric]