"""
dynamics/precompute.py

预计算模块：
1. 生成柔性杆空间离散网格
2. 计算轴向/横向模态函数及其导数
3. 计算耦合形函数 H(x)
4. 计算 M、Q 中反复使用的常系数矩阵
5. 将所有预计算结果打包，供后续动力学模块直接调用

说明
----
1. 本模块只负责“与系统状态 q, dq, tau 无关”的预计算内容。
2. 本模块中的结果在参数不变时只需计算一次。
3. 若模态阶数、材料参数、几何参数变化，需要重新预计算。
"""

from dataclasses import dataclass
import math
import torch

from .params import DynamicsParams


@dataclass
class PrecomputedTerms:
    # 空间离散网格
    x_grid: torch.Tensor       # 柔性杆轴向离散坐标
    dx: torch.Tensor           # 网格步长

    # 轴向模态函数
    Phi_A: torch.Tensor        # 轴向模态函数值
    dPhi_A: torch.Tensor       # 轴向模态函数一阶导数
    ddPhi_A: torch.Tensor      # 轴向模态函数二阶导数

    # 横向模态函数
    Phi_B: torch.Tensor        # 横向模态函数值
    dPhi_B: torch.Tensor       # 横向模态函数一阶导数
    ddPhi_B: torch.Tensor      # 横向模态函数二阶导数

    # 耦合形函数
    Hx: torch.Tensor           # 非线性耦合形函数 H(x)，形状为 (n_B, n_B, n_x_grid)

    # 预计算常系数项（供 M / Q 直接调用）
    sx: torch.Tensor              # Sx：第二杆轴向模态的一次惯性矩耦合项
    sy: torch.Tensor              # Sy：第二杆横向模态的一次惯性矩耦合项
    mx: torch.Tensor              # Mx：第二杆轴向模态质量项
    my: torch.Tensor              # My：第二杆横向模态质量项
    mxy: torch.Tensor             # Mxy：第二杆轴向/横向模态耦合质量项
    ch: torch.Tensor              # C：由 H(x) 引入的非线性惯性耦合项
    kx: torch.Tensor              # KA：第二杆轴向模态刚度项
    ky: torch.Tensor              # KB：第二杆横向弯曲模态刚度项

    rx: torch.Tensor              # rx：第一杆长度 L1 与第二杆轴向模态 Phi_x 的惯性耦合项
    ry: torch.Tensor              # ry：第一杆长度 L1 与第二杆横向模态 Phi_y 的惯性耦合项
    jr: torch.Tensor              # jr：第一杆长度 L1 与第二杆刚体分布质量的惯性耦合项
    rh: torch.Tensor              # rh：第一杆长度 L1 与非线性耦合函数 H(x) 的惯性耦合项


# 基础工具函数
def _zeros(shape, params: DynamicsParams) -> torch.Tensor:
    """按统一 device / dtype 创建零张量。"""
    return torch.zeros(shape, device=params.torch_device, dtype=params.torch_dtype)

def _trapz_integral(values: torch.Tensor, x_grid: torch.Tensor) -> torch.Tensor:
    """对离散函数在空间网格上做梯形积分，沿最后一维积分。"""
    dx = x_grid[1:] - x_grid[:-1]
    return torch.sum(0.5 * (values[..., :-1] + values[..., 1:]) * dx, dim=-1)


# 空间离散网格
def build_x_grid(params: DynamicsParams) -> tuple[torch.Tensor, torch.Tensor]:
    """生成柔性杆区间 [0, L2] 上的空间离散网格。"""
    if params.n_x_grid < 2:
        raise ValueError("n_x_grid 必须大于等于 2。")

    x_grid = torch.linspace(
        0.0,
        params.L2,
        params.n_x_grid,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )
    dx = x_grid[1] - x_grid[0]

    return x_grid, dx


# 轴向模态函数
def compute_axial_modes(
    x_grid: torch.Tensor,
    params: DynamicsParams
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按文档中的轴向模态公式计算轴向模态函数及其一、二阶空间导数。"""
    n_A = params.n_A
    L = params.L2

    Phi_A = _zeros((n_A, x_grid.numel()), params)
    dPhi_A = _zeros((n_A, x_grid.numel()), params)
    ddPhi_A = _zeros((n_A, x_grid.numel()), params)

    for i in range(n_A):
        mode_id = i + 1
        alpha_i = (2 * mode_id - 1) * math.pi / (2 * L)

        Phi_A[i, :] = torch.sin(alpha_i * x_grid)
        dPhi_A[i, :] = alpha_i * torch.cos(alpha_i * x_grid)
        ddPhi_A[i, :] = -(alpha_i ** 2) * torch.sin(alpha_i * x_grid)

    return Phi_A, dPhi_A, ddPhi_A


# 横向模态函数
def compute_transverse_modes(
    x_grid: torch.Tensor,
    params: DynamicsParams
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算横向模态函数及其一、二阶空间导数。"""
    n_B = params.n_B
    L2 = params.L2

    Phi_B = _zeros((n_B, x_grid.numel()), params)
    dPhi_B = _zeros((n_B, x_grid.numel()), params)
    ddPhi_B = _zeros((n_B, x_grid.numel()), params)

    betaL_list = [1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839]

    if n_B > len(betaL_list):
        raise ValueError(f"当前仅内置前 {len(betaL_list)} 阶悬臂梁横向模态参数，请补充更多 betaL。")

    for i in range(n_B):
        betaL = betaL_list[i]
        beta = betaL / L2

        gamma = -(
            math.cos(betaL) + math.cosh(betaL)
        ) / (
            math.sin(betaL) + math.sinh(betaL)
        )

        sin_term = torch.sin(beta * x_grid)
        cos_term = torch.cos(beta * x_grid)
        sinh_term = torch.sinh(beta * x_grid)
        cosh_term = torch.cosh(beta * x_grid)

        Phi_B[i, :] = (
            cos_term - cosh_term
            + gamma * (sin_term - sinh_term)
        )

        dPhi_B[i, :] = (
            -beta * sin_term
            - beta * sinh_term
            + gamma * beta * (cos_term - cosh_term)
        )

        ddPhi_B[i, :] = (
            -beta**2 * cos_term
            - beta**2 * cosh_term
            + gamma * beta**2 * (-sin_term - sinh_term)
        )

    return Phi_B, dPhi_B, ddPhi_B


# 耦合形函数
def compute_Hx(
    x_grid: torch.Tensor,
    Phi_B: torch.Tensor,
    dPhi_B: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算多模态耦合形函数 H(x)。"""
    n_B = params.n_B
    n_x = x_grid.numel()

    Hx = _zeros((n_B, n_B, n_x), params)
    dx = x_grid[1:] - x_grid[:-1]

    for i in range(n_B):
        for j in range(n_B):
            integrand = dPhi_B[i, :] * dPhi_B[j, :]
            Hx[i, j, 0] = 0
            Hx[i, j, 1:] = torch.cumsum(
                0.5 * (integrand[:-1] + integrand[1:]) * dx,
                dim=0
            )

    return Hx


# 常系数项计算
def compute_sx(
    x_grid: torch.Tensor,
    Phi_A: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 Sx：第二杆轴向模态的一次惯性矩耦合项。"""
    return params.rho * params.S * _trapz_integral(x_grid * Phi_A, x_grid)


def compute_sy(
    x_grid: torch.Tensor,
    Phi_B: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 Sy：第二杆横向模态的一次惯性矩耦合项。"""
    return params.rho * params.S * _trapz_integral(x_grid * Phi_B, x_grid)


def compute_mx(
    x_grid: torch.Tensor,
    Phi_A: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 Mx：第二杆轴向模态质量项。"""
    n_A = Phi_A.shape[0]
    mx = _zeros((n_A, n_A), params)

    for i in range(n_A):
        for j in range(n_A):
            mx[i, j] = params.rho * params.S * _trapz_integral(
                Phi_A[i, :] * Phi_A[j, :], x_grid
            )

    return mx


def compute_my(
    x_grid: torch.Tensor,
    Phi_B: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 My：第二杆横向模态质量项。"""
    n_B = Phi_B.shape[0]
    my = _zeros((n_B, n_B), params)

    for i in range(n_B):
        for j in range(n_B):
            my[i, j] = params.rho * params.S * _trapz_integral(
                Phi_B[i, :] * Phi_B[j, :], x_grid
            )

    return my


def compute_mxy(
    x_grid: torch.Tensor,
    Phi_A: torch.Tensor,
    Phi_B: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 Mxy：第二杆轴向/横向模态耦合质量项。"""
    n_A = Phi_A.shape[0]
    n_B = Phi_B.shape[0]
    mxy = _zeros((n_A, n_B), params)

    for i in range(n_A):
        for j in range(n_B):
            mxy[i, j] = params.rho * params.S * _trapz_integral(
                Phi_A[i, :] * Phi_B[j, :], x_grid
            )

    return mxy


def compute_ch(
    x_grid: torch.Tensor,
    Hx: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 C：由 H(x) 引入的非线性惯性耦合项。"""
    return params.rho * params.S * _trapz_integral(x_grid * Hx, x_grid)


def compute_kx(
    x_grid: torch.Tensor,
    dPhi_A: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 Kx：第二杆轴向模态刚度项。"""
    n_A = dPhi_A.shape[0]
    kx = _zeros((n_A, n_A), params)

    for i in range(n_A):
        for j in range(n_A):
            kx[i, j] = params.E * params.S * _trapz_integral(
                dPhi_A[i, :] * dPhi_A[j, :], x_grid
            )

    return kx


def compute_ky(
    x_grid: torch.Tensor,
    ddPhi_B: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 Ky：第二杆横向弯曲模态刚度项。"""
    n_B = ddPhi_B.shape[0]
    ky = _zeros((n_B, n_B), params)

    for i in range(n_B):
        for j in range(n_B):
            ky[i, j] = params.E * params.Iz * _trapz_integral(
                ddPhi_B[i, :] * ddPhi_B[j, :], x_grid
            )

    return ky


# 其余惯性耦合项
def compute_rx(
    x_grid: torch.Tensor,
    Phi_A: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 rx：第一杆长度 L1 与第二杆轴向模态 Phi_x 的惯性耦合项。"""
    return params.L1 * params.rho * params.S * _trapz_integral(Phi_A, x_grid)


def compute_ry(
    x_grid: torch.Tensor,
    Phi_B: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 ry：第一杆长度 L1 与第二杆横向模态 Phi_y 的惯性耦合项。"""
    return params.L1 * params.rho * params.S * _trapz_integral(Phi_B, x_grid)


def compute_jr(params: DynamicsParams) -> torch.Tensor:
    """计算 jr：第一杆长度 L1 与第二杆刚体分布质量的惯性耦合项。"""
    return torch.tensor(
        params.L1 * params.rho * params.S * (params.L2 ** 2) / 2.0,
        device=params.torch_device,
        dtype=params.torch_dtype,
    )


def compute_rh(
    x_grid: torch.Tensor,
    Hx: torch.Tensor,
    params: DynamicsParams
) -> torch.Tensor:
    """计算 rh：第一杆长度 L1 与非线性耦合函数 H(x) 的惯性耦合项。"""
    return params.L1 * params.rho * params.S * _trapz_integral(Hx, x_grid)



# 总装入口
def precompute_all(params: DynamicsParams) -> PrecomputedTerms:
    """执行全部预计算，并将结果打包返回。"""
    x_grid, dx = build_x_grid(params)

    Phi_A, dPhi_A, ddPhi_A = compute_axial_modes(x_grid, params)
    Phi_B, dPhi_B, ddPhi_B = compute_transverse_modes(x_grid, params)

    Hx = compute_Hx(x_grid, Phi_B, dPhi_B, params)

    sx = compute_sx(x_grid, Phi_A, params)
    sy = compute_sy(x_grid, Phi_B, params)
    mx = compute_mx(x_grid, Phi_A, params)
    my = compute_my(x_grid, Phi_B, params)
    mxy = compute_mxy(x_grid, Phi_A, Phi_B, params)
    ch = compute_ch(x_grid, Hx, params)
    kx = compute_kx(x_grid, dPhi_A, params)
    ky = compute_ky(x_grid, ddPhi_B, params)

    rx = compute_rx(x_grid, Phi_A, params)
    ry = compute_ry(x_grid, Phi_B, params)
    jr = compute_jr(params)
    rh = compute_rh(x_grid, Hx, params)

    return PrecomputedTerms(
        x_grid=x_grid,
        dx=dx,
        Phi_A=Phi_A,
        dPhi_A=dPhi_A,
        ddPhi_A=ddPhi_A,
        Phi_B=Phi_B,
        dPhi_B=dPhi_B,
        ddPhi_B=ddPhi_B,
        Hx=Hx,
        sx=sx,
        sy=sy,
        mx=mx,
        my=my,
        mxy=mxy,
        ch=ch,
        kx=kx,
        ky=ky,
        rx=rx,
        ry=ry,
        jr=jr,
        rh=rh,
    )


if __name__ == "__main__":
    params = DynamicsParams()
    const = precompute_all(params)

    def check_finite(name: str, value):
        if torch.all(torch.isfinite(value)):
            print(f"[OK] {name} is finite.")
        else:
            print(f"[ERR] {name} contains nan/inf.")

    print("=== precompute test ===")
    print(f"x_grid shape : {tuple(const.x_grid.shape)}")
    print(f"dx           : {const.dx.item():.6e}")

    print(f"Phi_A shape  : {tuple(const.Phi_A.shape)}")
    print(f"dPhi_A shape : {tuple(const.dPhi_A.shape)}")
    print(f"ddPhi_A shape: {tuple(const.ddPhi_A.shape)}")

    print(f"Phi_B shape  : {tuple(const.Phi_B.shape)}")
    print(f"dPhi_B shape : {tuple(const.dPhi_B.shape)}")
    print(f"ddPhi_B shape: {tuple(const.ddPhi_B.shape)}")

    print(f"Hx shape     : {tuple(const.Hx.shape)}")

    check_finite("Phi_A", const.Phi_A)
    check_finite("dPhi_A", const.dPhi_A)
    check_finite("ddPhi_A", const.ddPhi_A)
    check_finite("Phi_B", const.Phi_B)
    check_finite("dPhi_B", const.dPhi_B)
    check_finite("ddPhi_B", const.ddPhi_B)
    check_finite("Hx", const.Hx)

    check_finite("sx", const.sx)
    check_finite("sy", const.sy)
    check_finite("mx", const.mx)
    check_finite("my", const.my)
    check_finite("mxy", const.mxy)
    check_finite("ch", const.ch)
    check_finite("kx", const.kx)
    check_finite("ky", const.ky)
    check_finite("rx", const.rx)
    check_finite("ry", const.ry)
    check_finite("jr", const.jr)
    check_finite("rh", const.rh)

    print("\n=== preview ===")
    print("sx =", const.sx.detach().cpu())
    print("sy =", const.sy.detach().cpu())
    print("mx =\n", const.mx.detach().cpu())
    print("my =\n", const.my.detach().cpu())
    print("mxy =\n", const.mxy.detach().cpu())
    print("ch =", const.ch.detach().cpu())
    print("kx =\n", const.kx.detach().cpu())
    print("ky =\n", const.ky.detach().cpu())
    print("rx =", const.rx.detach().cpu())
    print("ry =", const.ry.detach().cpu())
    print("jr =", const.jr.item())
    print("rh =", const.rh.detach().cpu())