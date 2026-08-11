"""
dynamics package

柔性机械臂动力学模块统一导出入口。
"""

from .params import DynamicsParams
from .precompute import PrecomputedTerms, precompute_all
from .mass_matrix import compute_M
from .generalized_force import compute_Q
from .model import FlexibleArmModel

__all__ = [
    "DynamicsParams",
    "PrecomputedTerms",
    "precompute_all",
    "compute_M",
    "compute_Q",
    "FlexibleArmModel",
]

"""
使用示例：
from tracking.dynamics import DynamicsParams, FlexibleArmModel, precompute_all
"""