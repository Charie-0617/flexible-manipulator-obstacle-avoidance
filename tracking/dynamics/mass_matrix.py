"""
dynamics/mass_matrix.py

广义质量阵模块：
1. 解析广义坐标 q
2. 计算质量阵各分块
3. 组装完整广义质量阵 M(q)
4. 对 M 做基本检查与对称性处理

当前广义坐标顺序约定为：
    xi = [theta1, theta2, q1, q2, A..., B...]
其中：
    theta1 : 第一关节角
    theta2 : 第二关节角
    q1     : 第一驱动侧/柔性铰广义坐标
    q2     : 第二驱动侧/柔性铰广义坐标
    A      : 轴向模态坐标向量
    B      : 横向模态坐标向量
"""

import torch

from .params import DynamicsParams
from .precompute import PrecomputedTerms

# 基础工具函数
def _zero_scalar(params: DynamicsParams) -> torch.Tensor:
    """创建 0 维零标量张量。"""
    return torch.zeros((), device=params.torch_device, dtype=params.torch_dtype)

def get_dof(params: DynamicsParams) -> int:
    """返回系统总自由度数。"""
    return 4 + params.n_A + params.n_B

def split_xi(
    xi: torch.Tensor,
    params: DynamicsParams
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按 [theta1, theta2, q1, q2, A..., B...] 的顺序拆分广义坐标。"""
    xi = torch.as_tensor(
        xi,
        device=params.torch_device,
        dtype=params.torch_dtype
    ).reshape(-1)

    n_xi = get_dof(params)

    if xi.numel() != n_xi:
        raise ValueError(f"xi 的长度应为 {n_xi}，当前为 {xi.numel()}。")

    theta1 = xi[0]
    theta2 = xi[1]
    q1 = xi[2]
    q2 = xi[3]
    A = xi[4:4 + params.n_A]
    B = xi[4 + params.n_A:4 + params.n_A + params.n_B]

    return theta1, theta2, q1, q2, A, B


def check_mass_matrix(M: torch.Tensor, atol: float = 1e-9) -> None:
    """检查质量矩阵是否为有限、方阵且近似对称。"""
    if M.ndim != 2:
        raise ValueError("质量矩阵 M 必须是二维张量。")

    if M.shape[0] != M.shape[1]:
        raise ValueError(f"质量矩阵 M 必须是方阵，当前 shape = {tuple(M.shape)}。")

    if not torch.all(torch.isfinite(M)):
        raise ValueError("质量矩阵 M 含有 nan 或 inf。")

    if not torch.allclose(M, M.T, atol=atol, rtol=1e-7):
        raise ValueError("质量矩阵 M 不是近似对称矩阵。")


def _mass_density(params: DynamicsParams) -> float:
    """返回分布质量系数 rho * S。"""
    return params.rho * params.S

def _trapz(values: torch.Tensor, x_grid: torch.Tensor) -> torch.Tensor:
    """对一维离散函数做梯形积分并返回标量张量。"""
    dx = x_grid[1:] - x_grid[:-1]
    return torch.sum(0.5 * (values[:-1] + values[1:]) * dx)

def _scalarize(x: torch.Tensor) -> torch.Tensor:
    """将 0 维张量或单元素张量转为标量张量。"""
    return x.reshape(-1)[0]

def _phi_xA_at_grid(A: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 Phi_x(x) A。"""
    if A.numel() == 0:
        return torch.zeros_like(const.x_grid)
    return A @ const.Phi_A

def _phi_yB_at_grid(B: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 Phi_y(x) B，返回形状 (n_x_grid,)。"""
    if B.numel() == 0:
        return torch.zeros_like(const.x_grid)
    return B @ const.Phi_B

def _HB_at_grid(B: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 H(x)B，返回形状 (n_B, n_x_grid)。"""
    return torch.einsum("ijn,j->in", const.Hx, B)

def _BtHB_at_grid(B: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 B^T H(x) B，返回形状 (n_x_grid,)。"""
    return torch.einsum("i,ijn,j->n", B, const.Hx, B)

def enforce_symmetry(M: torch.Tensor) -> torch.Tensor:
    """对数值误差导致的轻微非对称矩阵进行对称化。"""
    return 0.5 * (M + M.T)


# M 的各分块元素
def compute_M11(
    theta2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M11。"""
    mu = _mass_density(params)

    phi_xA = _phi_xA_at_grid(A, const)
    BtHB = _BtHB_at_grid(B, const)

    term_rigid_1 = torch.tensor(
        params.m1 * (params.lc1 ** 2) + params.J1,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )
    term_rigid_2 = torch.tensor(
        mu * (params.L1 ** 2) * params.L2,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )
    term_job = torch.as_tensor(
        params.Job,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )

    term_AA = _scalarize(A @ const.mx @ A) if A.numel() > 0 else _zero_scalar(params)
    term_BB = _scalarize(B @ const.my @ B) if B.numel() > 0 else _zero_scalar(params)

    term_H2 = 0.25 * mu * _trapz(BtHB ** 2, const.x_grid)
    term_xA = 2.0 * _scalarize(const.sx @ A) if A.numel() > 0 else _zero_scalar(params)
    term_xHB = -mu * _trapz(const.x_grid * BtHB, const.x_grid)
    term_AHB = -mu * _trapz(phi_xA * BtHB, const.x_grid)
    term_sin = -2.0 * torch.sin(theta2) * _scalarize(const.ry @ B) if B.numel() > 0 else _zero_scalar(params)
    term_cos = 2.0 * torch.cos(theta2) * mu * _trapz(
        params.L1 * (const.x_grid + phi_xA - 0.5 * BtHB),
        const.x_grid
    )

    return (
        term_rigid_1
        + term_rigid_2
        + term_job
        + term_AA
        + term_BB
        + term_H2
        + term_xA
        + term_xHB
        + term_AHB
        + term_sin
        + term_cos
    )


def compute_M12(
    theta2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M12 = M21。"""
    mu = _mass_density(params)

    phi_xA = _phi_xA_at_grid(A, const)
    BtHB = _BtHB_at_grid(B, const)

    term_BB = _scalarize(B @ const.my @ B) if B.numel() > 0 else _zero_scalar(params)
    term_job = torch.as_tensor(
        params.Job,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )
    term_AA = _scalarize(A @ const.mx @ A) if A.numel() > 0 else _zero_scalar(params)

    term_H2 = 0.25 * mu * _trapz(BtHB ** 2, const.x_grid)
    term_xA = 2.0 * _scalarize(const.sx @ A) if A.numel() > 0 else _zero_scalar(params)
    term_xHB = -mu * _trapz(const.x_grid * BtHB, const.x_grid)
    term_AHB = -mu * _trapz(phi_xA * BtHB, const.x_grid)
    term_sin = -torch.sin(theta2) * _scalarize(const.ry @ B) if B.numel() > 0 else _zero_scalar(params)
    term_cos = torch.cos(theta2) * mu * _trapz(
        params.L1 * (const.x_grid + phi_xA - 0.5 * BtHB),
        const.x_grid
    )

    return (
        term_BB
        + term_job
        + term_AA
        + term_H2
        + term_xA
        + term_xHB
        + term_AHB
        + term_sin
        + term_cos
    )


def compute_M22(
    A: torch.Tensor,
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M22。"""
    mu = _mass_density(params)

    phi_xA = _phi_xA_at_grid(A, const)
    BtHB = _BtHB_at_grid(B, const)

    term_BB = _scalarize(B @ const.my @ B) if B.numel() > 0 else _zero_scalar(params)
    term_job = torch.as_tensor(
        params.Job,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )
    term_AA = _scalarize(A @ const.mx @ A) if A.numel() > 0 else _zero_scalar(params)

    term_H2 = 0.25 * mu * _trapz(BtHB ** 2, const.x_grid)
    term_xA = 2.0 * _scalarize(const.sx @ A) if A.numel() > 0 else _zero_scalar(params)
    term_xHB = -mu * _trapz(const.x_grid * BtHB, const.x_grid)
    term_AHB = -mu * _trapz(phi_xA * BtHB, const.x_grid)

    return (
        term_BB
        + term_job
        + term_AA
        + term_H2
        + term_xA
        + term_xHB
        + term_AHB
    )


def compute_M25(
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M25 = M52^T。"""
    mu = _mass_density(params)

    phi_yB = _phi_yB_at_grid(B, const)
    n_A = params.n_A
    M25 = torch.zeros((1, n_A), device=params.torch_device, dtype=params.torch_dtype)

    for i in range(n_A):
        phi_xi = const.Phi_A[i, :]
        M25[0, i] = -mu * _trapz(phi_xi * phi_yB, const.x_grid)

    return M25


def compute_M33(params: DynamicsParams) -> torch.Tensor:
    """计算 M33。"""
    return torch.tensor(
        params.Jm1,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )


def compute_M44(params: DynamicsParams) -> torch.Tensor:
    """计算 M44。"""
    return torch.tensor(
        params.Jm2,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )


def compute_M15(
    theta2: torch.Tensor,
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M15 = M51^T。"""
    if params.n_A == 0:
        return torch.zeros((1, 0), device=params.torch_device, dtype=params.torch_dtype)

    term1 = torch.sin(theta2) * const.rx
    term2 = const.mxy @ B if B.numel() > 0 else torch.zeros(
        params.n_A,
        device=params.torch_device,
        dtype=params.torch_dtype
    )

    return (term1 - term2).reshape(1, -1)


def compute_M16(
    theta2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M16 = M61^T。"""
    mu = _mass_density(params)

    phi_xA = _phi_xA_at_grid(A, const)
    phi_yB = _phi_yB_at_grid(B, const)
    HB = _HB_at_grid(B, const)
    BtHB = _BtHB_at_grid(B, const)

    n_B = params.n_B
    M16 = torch.zeros((1, n_B), device=params.torch_device, dtype=params.torch_dtype)

    for j in range(n_B):
        phi_yj = const.Phi_B[j, :]
        HBj = HB[j, :]

        term1 = -params.L1 * torch.sin(theta2) * mu * _trapz(HBj, const.x_grid)
        term2 = mu * _trapz(phi_yB * HBj, const.x_grid)
        term3 = torch.cos(theta2) * const.ry[j]
        term4 = const.sy[j]
        term5 = mu * _trapz(phi_xA * phi_yj, const.x_grid)
        term6 = -0.5 * mu * _trapz(BtHB * phi_yj, const.x_grid)

        M16[0, j] = term1 + term2 + term3 + term4 + term5 + term6

    return M16


def compute_M26(
    A: torch.Tensor,
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M26 = M62^T。"""
    mu = _mass_density(params)

    phi_xA = _phi_xA_at_grid(A, const)
    phi_yB = _phi_yB_at_grid(B, const)
    HB = _HB_at_grid(B, const)
    BtHB = _BtHB_at_grid(B, const)

    n_B = params.n_B
    M26 = torch.zeros((1, n_B), device=params.torch_device, dtype=params.torch_dtype)

    for j in range(n_B):
        phi_yj = const.Phi_B[j, :]
        HBj = HB[j, :]

        term1 = mu * _trapz(phi_yB * HBj, const.x_grid)
        term2 = const.sy[j]
        term3 = mu * _trapz(phi_xA * phi_yj, const.x_grid)
        term4 = -0.5 * mu * _trapz(BtHB * phi_yj, const.x_grid)

        M26[0, j] = term1 + term2 + term3 + term4

    return M26


def compute_M55(
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M55。"""
    return const.mx.clone()


def compute_M56(
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M56 = M65^T。"""
    mu = _mass_density(params)

    HB = _HB_at_grid(B, const)
    n_A = params.n_A
    n_B = params.n_B
    M56 = torch.zeros((n_A, n_B), device=params.torch_device, dtype=params.torch_dtype)

    for i in range(n_A):
        phi_xi = const.Phi_A[i, :]
        for j in range(n_B):
            HBj = HB[j, :]
            M56[i, j] = -mu * _trapz(phi_xi * HBj, const.x_grid)

    return M56


def compute_M66(
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 M66。"""
    mu = _mass_density(params)

    HB = _HB_at_grid(B, const)
    n_B = params.n_B
    M66 = torch.zeros((n_B, n_B), device=params.torch_device, dtype=params.torch_dtype)

    for i in range(n_B):
        for j in range(n_B):
            term_H = mu * _trapz(HB[i, :] * HB[j, :], const.x_grid)
            term_Y = mu * _trapz(const.Phi_B[i, :] * const.Phi_B[j, :], const.x_grid)
            M66[i, j] = term_H + term_Y

    return M66


# 总装函数
def assemble_mass_matrix(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """按分块结构组装完整广义质量阵。"""
    n_A = params.n_A
    n_B = params.n_B
    n_xi = get_dof(params)

    M = torch.zeros((n_xi, n_xi), device=params.torch_device, dtype=params.torch_dtype)

    M[0, 0] = compute_M11(theta2, A, B, params, const)
    M[0, 1] = compute_M12(theta2, A, B, params, const)
    M[1, 0] = M[0, 1]
    M[1, 1] = compute_M22(A, B, params, const)

    M[2, 2] = compute_M33(params)
    M[3, 3] = compute_M44(params)

    if n_A > 0:
        M15 = compute_M15(theta2, B, params, const)
        M[0, 4:4 + n_A] = M15.reshape(-1)
        M[4:4 + n_A, 0] = M15.T.reshape(-1)

    if n_A > 0:
        M25 = compute_M25(B, params, const)
        M[1, 4:4 + n_A] = M25.reshape(-1)
        M[4:4 + n_A, 1] = M25.T.reshape(-1)

    if n_B > 0:
        M16 = compute_M16(theta2, A, B, params, const)
        M[0, 4 + n_A:4 + n_A + n_B] = M16.reshape(-1)
        M[4 + n_A:4 + n_A + n_B, 0] = M16.T.reshape(-1)

    if n_B > 0:
        M26 = compute_M26(A, B, params, const)
        M[1, 4 + n_A:4 + n_A + n_B] = M26.reshape(-1)
        M[4 + n_A:4 + n_A + n_B, 1] = M26.T.reshape(-1)

    if n_A > 0:
        M55 = compute_M55(params, const)
        M[4:4 + n_A, 4:4 + n_A] = M55

    if n_A > 0 and n_B > 0:
        M56 = compute_M56(B, params, const)
        M[4:4 + n_A, 4 + n_A:4 + n_A + n_B] = M56
        M[4 + n_A:4 + n_A + n_B, 4:4 + n_A] = M56.T

    if n_B > 0:
        M66 = compute_M66(B, params, const)
        M[4 + n_A:4 + n_A + n_B, 4 + n_A:4 + n_A + n_B] = M66

    return M


def compute_M(
    xi: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms,
    enforce_symmetry_flag: bool = True
) -> torch.Tensor:
    """计算完整广义质量阵 M(xi)。"""
    theta1, theta2, q1, q2, A, B = split_xi(xi, params)

    M = assemble_mass_matrix(theta1, theta2, q1, q2, A, B, params, const)

    if enforce_symmetry_flag:
        M = enforce_symmetry(M)

    check_mass_matrix(M)
    return M


if __name__ == "__main__":
    try:
        from .precompute import precompute_all
    except ImportError:
        from precompute import precompute_all

    params = DynamicsParams()
    const = precompute_all(params)

    xi = torch.zeros(get_dof(params), device=params.torch_device, dtype=params.torch_dtype)
    M = compute_M(xi, params, const)

    print("=== mass matrix test ===")
    print(f"dof       : {get_dof(params)}")
    print(f"xi shape  : {tuple(xi.shape)}")
    print(f"M shape   : {tuple(M.shape)}")
    print(f"M finite  : {torch.all(torch.isfinite(M)).item()}")
    print(f"M sym     : {torch.allclose(M, M.T, atol=1e-9)}")
    print("M =\n", M.detach().cpu())