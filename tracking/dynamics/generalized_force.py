"""
dynamics/generalized_force.py

广义力模块：
1. 解析广义坐标 xi 与广义速度 dxi
2. 按分块形式计算各 Q_i
3. 组装完整广义力向量 Q(xi, dxi)
4. 叠加驱动力矩输入 tau

当前广义坐标顺序约定为：
    xi  = [theta1, theta2, q1, q2, A..., B...]
    dxi = [dtheta1, dtheta2, dq1, dq2, dA..., dB...]
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


def split_dxi(
    dxi: torch.Tensor,
    params: DynamicsParams
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按 [dtheta1, dtheta2, dq1, dq2, dA..., dB...] 的顺序拆分广义速度。"""
    dxi = torch.as_tensor(
        dxi,
        device=params.torch_device,
        dtype=params.torch_dtype
    ).reshape(-1)

    n_xi = get_dof(params)

    if dxi.numel() != n_xi:
        raise ValueError(f"dxi 的长度应为 {n_xi}，当前为 {dxi.numel()}。")

    dtheta1 = dxi[0]
    dtheta2 = dxi[1]
    dq1 = dxi[2]
    dq2 = dxi[3]
    dA = dxi[4:4 + params.n_A]
    dB = dxi[4 + params.n_A:4 + params.n_A + params.n_B]

    return dtheta1, dtheta2, dq1, dq2, dA, dB


def _mass_density(params):
    return torch.tensor(
        params.rho * params.S,
        device=params.torch_device,
        dtype=params.torch_dtype
    )

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
    """计算离散网格上的 Phi_y(x) B。"""
    if B.numel() == 0:
        return torch.zeros_like(const.x_grid)
    return B @ const.Phi_B

def _HB_at_grid(B: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 H(x)B，返回形状 (n_B, n_x_grid)。"""
    return torch.einsum("ijn,j->in", const.Hx, B)

def _BtHB_at_grid(B: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 B^T H(x) B，返回形状 (n_x_grid,)。"""
    return torch.einsum("i,ijn,j->n", B, const.Hx, B)

def _BtdotHBdot_at_grid(dB: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 Bdot^T H(x) Bdot，返回形状 (n_x_grid,)。"""
    return torch.einsum("i,ijn,j->n", dB, const.Hx, dB)

def _BtHBdot_at_grid(B: torch.Tensor, dB: torch.Tensor, const: PrecomputedTerms) -> torch.Tensor:
    """计算离散网格上的 B^T H(x) Bdot，返回形状 (n_x_grid,)。"""
    return torch.einsum("i,ijn,j->n", B, const.Hx, dB)


# 输入与阻尼
def compute_input_vector(
    tau: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """将驱动力矩 tau 装配到广义力向量对应位置。"""
    tau = torch.as_tensor(
        tau,
        device=params.torch_device,
        dtype=params.torch_dtype
    ).reshape(-1)

    if tau.numel() != 2:
        raise ValueError(f"tau 的长度应为 2，当前为 {tau.numel()}。")

    u = torch.zeros(get_dof(params), device=params.torch_device, dtype=params.torch_dtype)
    u[2] = tau[0]
    u[3] = tau[1]
    return u

def compute_modal_damping_force(
    dB: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算横向模态阻尼广义力 C_b dB。"""
    return params.Cb @ dB


# Q 的各分块元素
def compute_Q1(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dtheta1: torch.Tensor,
    dtheta2: torch.Tensor,
    dq1: torch.Tensor,
    dq2: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 Q_(theta1)。"""
    mu = _mass_density(params)
    omega = dtheta1 + dtheta2

    phi_xA = _phi_xA_at_grid(A, const)
    phi_yB = _phi_yB_at_grid(B, const)
    BtHB = _BtHB_at_grid(B, const)
    BtHBdot = _BtHBdot_at_grid(B, dB, const)
    BtdotHBdot = _BtdotHBdot_at_grid(dB, const)

    term1 = -2.0 * omega * _scalarize(const.sx @ dA)
    term2 = 2.0 * omega * mu * _trapz(const.x_grid * BtHBdot, const.x_grid)
    term3 = -2.0 * omega * _scalarize(A @ const.mx @ dA)
    term4 = 2.0 * omega * mu * _trapz(phi_xA * BtHBdot, const.x_grid)

    phi_xdA = _phi_xA_at_grid(dA, const)
    term5 = omega * mu * _trapz(BtHB * phi_xdA, const.x_grid)
    term6 = -omega * mu * _trapz(BtHB * BtHBdot, const.x_grid)
    term7 = -2.0 * omega * _scalarize(B @ const.my @ dB)

    term8 = -2.0 * omega * torch.cos(theta2) * _scalarize(const.rx @ dA)
    term9 = 2.0 * omega * torch.cos(theta2) * mu * params.L1 * _trapz(BtHBdot, const.x_grid)
    term10 = 2.0 * omega * torch.sin(theta2) * _scalarize(const.ry @ dB)

    term11 = dtheta2 * (2.0 * dtheta1 + dtheta2) * torch.cos(theta2) * _scalarize(const.ry @ B)
    term12 = dtheta2 * (2.0 * dtheta1 + dtheta2) * torch.sin(theta2) * const.jr
    term13 = dtheta2 * (2.0 * dtheta1 + dtheta2) * torch.sin(theta2) * _scalarize(const.rx @ A)
    term14 = -0.5 * dtheta2 * (2.0 * dtheta1 + dtheta2) * torch.sin(theta2) * _scalarize(B @ const.rh @ B)

    term15 = params.L1 * torch.sin(theta2) * mu * _trapz(BtdotHBdot, const.x_grid)
    term16 = -mu * _trapz(phi_yB * BtdotHBdot, const.x_grid)
    term17 = params.K1 * (q1 - theta1)

    return (
        term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
        + term9 + term10 + term11 + term12 + term13 + term14 + term15
        + term16 + term17
    )


def compute_Q2(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dtheta1: torch.Tensor,
    dtheta2: torch.Tensor,
    dq1: torch.Tensor,
    dq2: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 Q_(theta2)。"""
    mu = _mass_density(params)
    omega = dtheta1 + dtheta2

    phi_xdA = _phi_xA_at_grid(dA, const)
    phi_ydB = _phi_yB_at_grid(dB, const)
    phi_xA = _phi_xA_at_grid(A, const)

    BtHB = _BtHB_at_grid(B, const)
    BtHBdot = _BtHBdot_at_grid(B, dB, const)

    term1 = -2.0 * omega * torch.cos(theta2) * params.L1 * mu * _trapz(
        phi_xdA - BtHBdot,
        const.x_grid
    )

    term2 = 2.0 * omega * torch.sin(theta2) * params.L1 * mu * _trapz(
        phi_ydB,
        const.x_grid
    )

    term3 = dtheta1 * (dtheta1 + 2.0 * dtheta2) * torch.cos(theta2) * _scalarize(const.ry @ B)

    term4 = dtheta1 * (dtheta1 + 2.0 * dtheta2) * torch.sin(theta2) * params.L1 * mu * _trapz(
        const.x_grid + phi_xA - 0.5 * BtHB,
        const.x_grid
    )

    term5 = params.K2 * (q2 - theta2)

    return term1 + term2 + term3 + term4 + term5


def compute_Q3(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dtheta1: torch.Tensor,
    dtheta2: torch.Tensor,
    dq1: torch.Tensor,
    dq2: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 Q3 = -K1(q1 - theta1)。"""
    return -params.K1 * (q1 - theta1)

def compute_Q4(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dtheta1: torch.Tensor,
    dtheta2: torch.Tensor,
    dq1: torch.Tensor,
    dq2: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 Q4 = -K2(q2 - theta2)。"""
    return -params.K2 * (q2 - theta2)

def compute_Q5(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dtheta1: torch.Tensor,
    dtheta2: torch.Tensor,
    dq1: torch.Tensor,
    dq2: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 Q5。对应轴向模态广义力块。"""
    mu = _mass_density(params)
    n_A = params.n_A

    Q5 = torch.zeros(n_A, device=params.torch_device, dtype=params.torch_dtype)

    omega = dtheta1 + dtheta2
    BtHB = _BtHB_at_grid(B, const)
    BtdotHBdot = _BtdotHBdot_at_grid(dB, const)

    for i in range(n_A):
        phi_xi = const.Phi_A[i, :]

        term1 = (omega ** 2) * const.sx[i]
        term2 = (omega ** 2) * (const.mx[i, :] @ A)
        term3 = -0.5 * (omega ** 2) * mu * _trapz(phi_xi * BtHB, const.x_grid)
        term4 = 2.0 * omega * (const.mxy[i, :] @ dB)
        term5 = mu * _trapz(phi_xi * BtdotHBdot, const.x_grid)
        term6 = (dtheta1 ** 2) * torch.cos(theta2) * const.rx[i]
        term7 = -(const.kx[i, :] @ A)

        Q5[i] = term1 + term2 + term3 + term4 + term5 + term6 + term7

    return Q5


def compute_Q6(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dtheta1: torch.Tensor,
    dtheta2: torch.Tensor,
    dq1: torch.Tensor,
    dq2: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算 Q6。对应横向模态广义力块。"""
    mu = _mass_density(params)
    n_B = params.n_B

    Q6 = torch.zeros(n_B, device=params.torch_device, dtype=params.torch_dtype)

    omega = dtheta1 + dtheta2
    phi_xA = _phi_xA_at_grid(A, const)
    HB = _HB_at_grid(B, const)
    BtHB = _BtHB_at_grid(B, const)
    BtHBdot = _BtHBdot_at_grid(B, dB, const)
    BtdotHBdot = _BtdotHBdot_at_grid(dB, const)

    for j in range(n_B):
        phi_yj = const.Phi_B[j, :]
        HBj = HB[j, :]

        term1 = -(omega ** 2) * (const.my[j, :] @ B)
        term2 = -(omega ** 2) * mu * _trapz(const.x_grid * HBj, const.x_grid)
        term3 = -(omega ** 2) * mu * _trapz(phi_xA * HBj, const.x_grid)
        term4 = 0.5 * (omega ** 2) * mu * _trapz(BtHB * HBj, const.x_grid)
        term5 = -2.0 * omega * (const.mxy[:, j] @ dA)
        term6 = 2.0 * omega * mu * _trapz(BtHBdot * phi_yj, const.x_grid)
        term7 = -mu * _trapz(BtdotHBdot * HBj, const.x_grid)
        term8 = dtheta1 * (dtheta1 + 2.0 * dtheta2) * torch.sin(theta2) * const.ry[j]
        term9 = -dtheta1 * (dtheta1 + 2.0 * dtheta2) * params.L1 * torch.cos(theta2) * mu * _trapz(HBj, const.x_grid)
        term10 = -(const.ky[j, :] @ B)

        Q6[j] = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10

    # 模态阻尼已删除（c_B = 0，不需要衰减）
    # Q6 -= compute_modal_damping_force(dB, params)

    return Q6
# 总装函数
def assemble_generalized_force(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dtheta1: torch.Tensor,
    dtheta2: torch.Tensor,
    dq1: torch.Tensor,
    dq2: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    tau: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """按分块形式组装完整广义力向量 Q。"""
    n_xi = get_dof(params)
    Q = torch.zeros(n_xi, device=params.torch_device, dtype=params.torch_dtype)

    Q[0] = compute_Q1(theta1, theta2, q1, q2, A, B, dtheta1, dtheta2, dq1, dq2, dA, dB, params, const)
    Q[1] = compute_Q2(theta1, theta2, q1, q2, A, B, dtheta1, dtheta2, dq1, dq2, dA, dB, params, const)
    Q[2] = compute_Q3(theta1, theta2, q1, q2, A, B, dtheta1, dtheta2, dq1, dq2, dA, dB, params, const)
    Q[3] = compute_Q4(theta1, theta2, q1, q2, A, B, dtheta1, dtheta2, dq1, dq2, dA, dB, params, const)

    if params.n_A > 0:
        Q[4:4 + params.n_A] = compute_Q5(
            theta1, theta2, q1, q2, A, B,
            dtheta1, dtheta2, dq1, dq2, dA, dB,
            params, const
        )

    if params.n_B > 0:
        Q[4 + params.n_A:4 + params.n_A + params.n_B] = compute_Q6(
            theta1, theta2, q1, q2, A, B,
            dtheta1, dtheta2, dq1, dq2, dA, dB,
            params, const
        )

    Q += compute_input_vector(tau, params)
    return Q


def compute_Q(
    xi: torch.Tensor,
    dxi: torch.Tensor,
    tau: torch.Tensor,
    params: DynamicsParams,
    const: PrecomputedTerms
) -> torch.Tensor:
    """计算完整广义力向量 Q(xi, dxi)。"""
    theta1, theta2, q1, q2, A, B = split_xi(xi, params)
    dtheta1, dtheta2, dq1, dq2, dA, dB = split_dxi(dxi, params)

    Q = assemble_generalized_force(
        theta1, theta2, q1, q2, A, B,
        dtheta1, dtheta2, dq1, dq2, dA, dB,
        tau, params, const
    )

    if Q.shape != (get_dof(params),):
        raise ValueError(f"Q 的形状应为 ({get_dof(params)},)，当前为 {tuple(Q.shape)}。")

    return Q


if __name__ == "__main__":
    try:
        from .precompute import precompute_all
    except ImportError:
        from precompute import precompute_all

    params = DynamicsParams()
    const = precompute_all(params)

    xi = torch.zeros(get_dof(params), device=params.torch_device, dtype=params.torch_dtype)
    dxi = torch.zeros(get_dof(params), device=params.torch_device, dtype=params.torch_dtype)
    tau = torch.zeros(2, device=params.torch_device, dtype=params.torch_dtype)

    Q = compute_Q(xi, dxi, tau, params, const)

    print("=== generalized force test ===")
    print(f"dof         : {get_dof(params)}")
    print(f"xi shape    : {tuple(xi.shape)}")
    print(f"dxi shape   : {tuple(dxi.shape)}")
    print(f"Q shape     : {tuple(Q.shape)}")
    print(f"Q finite    : {torch.all(torch.isfinite(Q)).item()}")
    print(f"Q near zero : {torch.allclose(Q, torch.zeros_like(Q), atol=1e-9)}")
    print("Q =\n", Q.detach().cpu())