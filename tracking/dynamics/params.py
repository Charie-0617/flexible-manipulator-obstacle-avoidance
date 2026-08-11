"""
dynamics/params.py
机械臂与柔性杆动力学模型的基础参数定义。
说明
----
1. 本文件只负责“参数定义”，不承担动力学计算逻辑。
2. 建议所有长度单位统一为 m，质量单位统一为 kg，转动惯量单位统一为 kg*m^2。
"""

from dataclasses import dataclass, field

import torch
import numpy as np

@dataclass
class DynamicsParams:
    # 模态截断参数
    n_A: int = 1  # 轴向模态数（与策略网络 A1 一致）
    n_B: int = 1  # 横向模态数（与策略网络 B1 一致）
    n_x_grid: int = 200    # 空间离散点数（模态函数/积分计算时使用）

    # 第一连杆（刚性杆）参数
    L1: float = 2.4          # 第一杆长度
    lc1: float = 1.2         # 第一杆质心到转轴距离
    m1: float = 14.0         # 第一杆质量
    J1: float = 6.72         # 第一杆绕质心的转动惯量
    Jo1: float = field(init=False)    # 第一杆绕关节O的等效转动惯量

    # 第二连杆（柔性杆）参数
    L2: float = 2.4          # 第二杆/柔性杆长度
    m2: float = 1.992        # 第二杆质量
    S: float = 3e-4          # 柔性杆横截面积
    Iz: float = 1.333e-8     # 截面惯性矩[m^4]
    rho: float = 2.7667e3    # 材料密度[kg/m^3]
    E: float = 6.8952e10     # 弹性模量[Pa]
    Job: float = field(init=False)   # 第二杆刚体基准转动惯量

    # 柔性铰参数
    K1: float = 1330.0       # 柔性铰1扭转刚度
    K2: float = 1330.0       # 柔性铰2扭转刚度

    # 电机转子参数
    Jm1: float = 0.7         # 电机1转子转动惯量
    Jm2: float = 0.7         # 电机2转子转动惯量

    # 横向模态阻尼参数（已删除，不再使用）
    c_B: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    # device设备设置
    device: str = "cuda"
    dtype: str = "float64"



    def __post_init__(self) -> None:
        """计算依赖基础参数的派生量，并初始化 torch 设备与数据类型。"""
        self.Jo1 = self.J1 + self.m1 * self.lc1**2
        self.Job = self.rho * self.S * self.L2**3 / 3

        self.c_B = np.asarray(self.c_B, dtype=float).reshape(-1)

        if self.c_B.size == 1 and self.n_B > 1:
            self.c_B = np.full(self.n_B, self.c_B.item(), dtype=float)

        if self.c_B.size != self.n_B:
            raise ValueError(
                f"c_B 的长度应为 n_B={self.n_B}，当前为 {self.c_B.size}。"
            )

        if np.any(self.c_B < 0):
            raise ValueError("c_B 中的阻尼系数不能为负。")

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.torch_device = torch.device(self.device)

        if self.dtype == "float64":
            self.torch_dtype = torch.float64
        elif self.dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError("dtype 只支持 'float64' 或 'float32'。")


    @property
    def Cb(self) -> torch.Tensor:
        """横向模态阻尼矩阵 C_b（torch 张量）。"""
        c_B_tensor = torch.as_tensor(
            self.c_B,
            device=self.torch_device,
            dtype=self.torch_dtype,
        )
        return torch.diag(c_B_tensor)