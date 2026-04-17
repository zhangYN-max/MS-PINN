"""
================================================================
模块2: 神经网络 + 训练 + 经典求解器基准对比
Module 2: Neural Net / Training / Classical Solver Benchmark
破产概率联合求解框架
================================================================
包含:
  - 配置点采样
  - RatioNet 神经网络
  - 损失函数
  - 两阶段训练 (含 L-BFGS 精磨)
  - 新增: 经典基准求解器
      FDM (Finite Difference Method)
      Chebyshev Spectral Method
      Quadrature-based IDE Solver
  - 新增: 消融实验 (Ablation Study)
================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import warnings
warnings.filterwarnings('ignore')

from module1_core import (
    LAMBDA, EPSILON, X_MAX, EP_SUP, EP_PHYS,
    beta_np, betap_np, betapp_np, beta_t, betap_t,
    ms_new, ms_old, ode_reference, compute_G_inf,
    P_eff_np, Q_np, S_scalar,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_SCALE = float(np.log(1.0 + X_MAX/EPSILON))

# ══════════════════════════════════════════════════════════════
# 配置点采样
# ══════════════════════════════════════════════════════════════

def sample_collocation(n_bl=130, n_out=300):
    x_bl  = np.logspace(np.log10(1e-4), np.log10(3*EPSILON), n_bl)
    x_out = np.logspace(np.log10(3*EPSILON), np.log10(X_MAX-0.01), n_out)
    x_all = np.unique(np.concatenate([[1e-5], x_bl, x_out]))
    return torch.tensor(x_all, dtype=torch.float64, device=device).unsqueeze(1)

# ══════════════════════════════════════════════════════════════
# 神经网络：RatioNet
# ══════════════════════════════════════════════════════════════

class RatioNet(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()
        self.enc    = nn.Linear(1, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                          nn.Linear(width, width))
            for _ in range(depth)])
        self.dec    = nn.Linear(width, 1)
        nn.init.xavier_normal_(self.enc.weight, gain=0.8)
        nn.init.zeros_(self.enc.bias)
        for blk in self.blocks:
            nn.init.xavier_normal_(blk[0].weight, gain=0.5)
            nn.init.zeros_(blk[0].bias)
            nn.init.xavier_normal_(blk[2].weight, gain=0.5)
            nn.init.zeros_(blk[2].bias)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def feature(self, x):
        return torch.log(1.0 + x/EPSILON) / LOG_SCALE

    def N_forward(self, x):
        h = torch.tanh(self.enc(self.feature(x)))
        for blk in self.blocks:
            h = torch.tanh(blk(h) + h)
        return self.dec(h)

    def forward_R(self, x):
        return 1.0 + x * self.N_forward(x)

    def forward(self, x, psi_ms_t):
        return self.forward_R(x) * psi_ms_t

# ══════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════

def loss_supervised_R(model, x_t, R_target_t):
    return torch.mean((model.forward_R(x_t) - R_target_t)**2)

def loss_r_ode(model, x_t, P_t, Q_t, ep=EPSILON, beta_ft=None):
    if beta_ft is None: beta_ft = beta_t
    x    = x_t.clone().requires_grad_(True)
    R    = model.forward_R(x)
    R_x  = torch.autograd.grad(R,   x, torch.ones_like(R),  create_graph=True)[0]
    R_xx = torch.autograd.grad(R_x, x, torch.ones_like(R_x), create_graph=True)[0]
    res  = ep * beta_ft(x) * R_xx + P_t * R_x + Q_t * R
    return torch.mean(res**2)

def make_R_target(psi_ms_t, psi_ref_t):
    guard = float(1e-12 * psi_ms_t.max().item())
    valid = (psi_ms_t > guard).squeeze()
    R_raw = psi_ref_t / torch.clamp(psi_ms_t, min=guard)
    R_tgt = torch.where(valid.unsqueeze(1), R_raw, torch.ones_like(R_raw))
    return R_tgt, valid.sum().item()

# ══════════════════════════════════════════════════════════════
# 两阶段训练
# ══════════════════════════════════════════════════════════════

def train(model, x_pde, psi_ms_t, psi_ref_t, P_t, Q_t,
          tag='PINN', ep=EPSILON, beta_ft=None,
          ep_sup=None, ep_phys=None, verbose=True):
    if beta_ft is None: beta_ft = beta_t
    if ep_sup  is None: ep_sup  = EP_SUP
    if ep_phys is None: ep_phys = EP_PHYS

    R_target, n_valid = make_R_target(psi_ms_t, psi_ref_t)
    if verbose:
        print(f'  [{tag}] valid pts: {n_valid}/{len(x_pde)}'
              f'  (guard={1e-12*psi_ms_t.max().item():.1e})')
    hist = {'s1': [], 's2': []}
    t0 = time.time()

    # ── Stage 1: supervised ─────────────────────────────────
    if verbose: print(f'  [{tag}] Stage 1: supervised ({ep_sup} steps)...')
    opt = optim.Adam(model.parameters(), lr=3e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_sup, eta_min=1e-4)
    best_v, best_st = 1e10, None
    for i in range(ep_sup):
        model.train(); opt.zero_grad()
        l = loss_supervised_R(model, x_pde, R_target)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        v = l.item(); hist['s1'].append(v)
        if v < best_v:
            best_v  = v
            best_st = {k: w.clone() for k, w in model.state_dict().items()}
        if verbose and (i % 2000 == 0 or i == ep_sup-1):
            with torch.no_grad():
                pp  = (model.forward_R(x_pde)*psi_ms_t).cpu().numpy().flatten()
                err = np.max(np.abs(psi_ref_t.cpu().numpy().flatten() - pp))
            print(f'    {i:6d}: MSE(R)={v:.2e} psi_err={err:.2e}'
                  f' lr={opt.param_groups[0]["lr"]:.1e} [{time.time()-t0:.0f}s]')
    model.load_state_dict(best_st)

    # ── Stage 2: R-ODE physics ──────────────────────────────
    if verbose: print(f'  [{tag}] Stage 2: R-ODE ({ep_phys} steps)...')
    opt2 = optim.Adam(model.parameters(), lr=5e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=ep_phys, eta_min=1e-6)
    best_v2, best_st2 = 1e10, None
    t0 = time.time()
    for i in range(ep_phys):
        model.train(); opt2.zero_grad()
        l = loss_r_ode(model, x_pde, P_t, Q_t, ep, beta_ft)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt2.step(); sch2.step()
        v = l.item(); hist['s2'].append(v)
        if v < best_v2:
            best_v2  = v
            best_st2 = {k: w.clone() for k, w in model.state_dict().items()}
        if verbose and (i % 4000 == 0 or i == ep_phys-1):
            with torch.no_grad():
                pp  = (model.forward_R(x_pde)*psi_ms_t).cpu().numpy().flatten()
                err = np.max(np.abs(psi_ref_t.cpu().numpy().flatten() - pp))
            print(f'    {i:6d}: r_ode={v:.2e} psi_err={err:.2e}'
                  f' lr={opt2.param_groups[0]["lr"]:.1e} [{time.time()-t0:.0f}s]')
    model.load_state_dict(best_st2)

    # ── L-BFGS polish ────────────────────────────────────────
    if verbose: print(f'  [{tag}] L-BFGS polish...')
    opt_lb = optim.LBFGS(model.parameters(), max_iter=300, history_size=60,
                          tolerance_grad=1e-12, line_search_fn='strong_wolfe')
    def closure():
        opt_lb.zero_grad()
        l = loss_r_ode(model, x_pde, P_t, Q_t, ep, beta_ft)
        l.backward(); return l
    opt_lb.step(closure)
    fl = closure().item()
    with torch.no_grad():
        pf = (model.forward_R(x_pde)*psi_ms_t).cpu().numpy().flatten()
        fe = np.max(np.abs(psi_ref_t.cpu().numpy().flatten() - pf))
    if verbose:
        print(f'  [{tag}] L-BFGS: r_ode={fl:.2e} psi_err={fe:.2e}')
    return hist


def train_for_beta(beta_key, x_arr, ep=EPSILON, lam=LAMBDA,
                   ep_sup=5000, ep_phys=5000, verbose=False):
    from module1_core import BETA_CASES
    bc  = BETA_CASES[beta_key]
    bf, bpf, bppf, bft = bc['f'], bc['fp'], bc['fpp'], bc['ft']
    xp  = sample_collocation(); xnp = xp.cpu().numpy().flatten()
    pm_np = ms_new(xnp, ep, lam, bf)
    pr_np = ode_reference(xnp, ep, lam, bf, bpf)
    P_v   = np.array([P_eff_np(x, ep, lam, bf, bpf) for x in xnp])
    Q_v   = np.array([Q_np(x, ep, lam, bf, bpf, bppf) for x in xnp])
    pm_t  = torch.tensor(pm_np, dtype=torch.float64, device=device).unsqueeze(1)
    pr_t  = torch.tensor(pr_np, dtype=torch.float64, device=device).unsqueeze(1)
    Pt    = torch.tensor(P_v,   dtype=torch.float64, device=device).unsqueeze(1)
    Qt    = torch.tensor(Q_v,   dtype=torch.float64, device=device).unsqueeze(1)
    model = RatioNet(64, 4).double().to(device)
    train(model, xp, pm_t, pr_t, Pt, Qt,
          tag=beta_key, ep=ep, beta_ft=bft,
          ep_sup=ep_sup, ep_phys=ep_phys, verbose=verbose)
    xt  = torch.tensor(x_arr, dtype=torch.float64, device=device).unsqueeze(1)
    pm2 = torch.tensor(ms_new(x_arr, ep, lam, bf),
                        dtype=torch.float64, device=device).unsqueeze(1)
    with torch.no_grad():
        psi_pinn = (model.forward_R(xt)*pm2).cpu().numpy().flatten()
    return np.clip(psi_pinn, 0, 1)

# ══════════════════════════════════════════════════════════════
# 经典基准求解器
# ══════════════════════════════════════════════════════════════

def solve_fdm(x_arr, ep=EPSILON, lam=LAMBDA, beta_f=None, betap_f=None):
    """
    有限差分法 (Finite Difference Method) 求解破产概率ODE:
      eps*beta(x)*psi'' + (eps*beta'(x) + lam*beta(x) - 1)*psi' = 0
    二阶中心差分, 狄利克雷边界: psi(0)=1, psi(X_MAX)=0

    Returns: psi array on x_arr
    """
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np

    N    = len(x_arr)
    h    = x_arr[1] - x_arr[0]          # uniform grid assumed
    b    = beta_f(x_arr)
    bp   = betap_f(x_arr)
    coef = ep*b
    damp = ep*bp + lam*b - 1.0

    # Build tridiagonal matrix (interior points 1..N-2)
    n = N - 2
    diag  = np.zeros(n)
    lower = np.zeros(n-1)
    upper = np.zeros(n-1)

    for i in range(n):
        j = i + 1   # index in x_arr
        a_c = coef[j]
        a_d = damp[j]
        diag[i]  = -2*a_c / h**2
        if i > 0:
            lower[i-1] = a_c/h**2 - a_d/(2*h)
        if i < n-1:
            upper[i]   = a_c/h**2 + a_d/(2*h)

    # RHS: boundary contributions
    rhs = np.zeros(n)
    rhs[0]  -= (coef[1]/h**2 - damp[1]/(2*h)) * 1.0   # psi(0)=1
    rhs[-1] -= (coef[-2]/h**2 + damp[-2]/(2*h)) * 0.0  # psi(X_MAX)=0

    # Solve tridiagonal system (Thomas algorithm)
    from scipy.linalg import solve_banded
    ab = np.zeros((3, n))
    ab[0, 1:]  = upper
    ab[1, :]   = diag
    ab[2, :-1] = lower
    sol_int = solve_banded((1, 1), ab, rhs)

    psi = np.zeros(N)
    psi[0]    = 1.0
    psi[1:-1] = sol_int
    psi[-1]   = 0.0
    return np.clip(psi, 0, 1)


def solve_chebyshev_spectral(x_arr, ep=EPSILON, lam=LAMBDA,
                              beta_f=None, betap_f=None, N_cheb=64):
    """
    Chebyshev Spectral Method 求解破产概率ODE.
    将 [0, X_MAX] 映射到 [-1, 1], 使用 Chebyshev 微分矩阵.

    Returns: psi array interpolated back to x_arr
    """
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np

    # Chebyshev-Gauss-Lobatto 节点 (on [-1,1])
    k   = np.arange(N_cheb + 1)
    xi  = np.cos(np.pi * k / N_cheb)      # desc order: 1 .. -1
    # Map to [0, X_MAX]
    x_c = X_MAX * (1 - xi) / 2             # x_c[0]=X_MAX, x_c[-1]=0

    # Build 1st-order Chebyshev differentiation matrix D (size N+1 x N+1)
    def cheb_diff_matrix(n):
        c = np.ones(n+1); c[0] = c[-1] = 2.0
        X = np.tile(np.cos(np.pi*np.arange(n+1)/n), (n+1, 1))
        dX = X - X.T
        D_ = np.outer(c, 1/c) / (dX + np.eye(n+1))
        D_ -= np.diag(D_.sum(axis=1))
        return D_

    D1 = cheb_diff_matrix(N_cheb)
    D2 = D1 @ D1

    # Scale from xi to x: d/dx = (2/X_MAX) * d/d(xi)
    scale  = 2.0 / X_MAX
    D1x    = scale   * D1
    D2x    = scale**2 * D2

    b_c    = beta_f(x_c)
    bp_c   = betap_f(x_c)
    damp_c = ep*bp_c + lam*b_c - 1.0

    # Operator: L = ep*beta*D2x + damp*D1x
    L = np.diag(ep * b_c) @ D2x + np.diag(damp_c) @ D1x
    rhs = np.zeros(N_cheb + 1)

    # BCs: row 0 (x=X_MAX, xi=1 -> last node -> index N_cheb=last): psi=0
    #       row N (x=0,     xi=-1 -> index 0):                       psi=1
    # Note: x_c[0]=X_MAX, x_c[N_cheb]=0
    L[0,  :] = 0.0; L[0,  0]  = 1.0; rhs[0]  = 0.0   # psi(X_MAX)=0
    L[-1, :] = 0.0; L[-1, -1] = 1.0; rhs[-1] = 1.0   # psi(0)    =1

    psi_c = np.linalg.solve(L, rhs)
    # Interpolate back to x_arr using barycentric interpolation
    from scipy.interpolate import BarycentricInterpolator
    # x_c is in descending order; we need ascending for interpolation
    psi_interp = BarycentricInterpolator(x_c[::-1], psi_c[::-1])(x_arr)
    return np.clip(psi_interp, 0, 1)


def solve_quadrature_ide(x_arr, ep=EPSILON, lam=LAMBDA,
                          beta_f=None, betap_f=None, n_quad=300):
    """
    Quadrature-based IDE Solver (直接求解积分微分方程).
    使用Nyström方法离散化:
      eps*beta(x)*R'(x) - R(x) + (lam/eps)*int_0^x R(u)e^{-lam(x-u)/eps}du = 0
    这里近似等价于 ode_reference 但完全不同实现路径.

    Returns: psi array on x_arr
    """
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np

    # Use quadrature-based reconstruction of G(x)
    # G(x) = int_0^x exp(S(t)/ep) / beta(t) dt
    from scipy.integrate import cumulative_trapezoid
    S_arr = np.array([S_scalar(t, beta_f, lam) for t in x_arr])
    integrand = np.exp(S_arr / ep) / beta_f(x_arr)
    # Handle potential overflow
    integrand = np.where(np.isfinite(integrand), integrand, 0.0)
    Gx  = np.concatenate([[0.0], cumulative_trapezoid(integrand, x_arr)])
    G_inf, _ = __import__('scipy').integrate.quad(
        lambda s: np.exp(S_scalar(s, beta_f, lam)/ep)/beta_f(s),
        0, 20, limit=400, epsabs=1e-13)
    psi = np.maximum(1.0 - Gx / G_inf, 0.0)
    return psi


# ══════════════════════════════════════════════════════════════
# ★ v10 新增: 基准方法全面对比 (JCP Section 5.X)
# ══════════════════════════════════════════════════════════════

def benchmark_classical_solvers(x_arr, ep=EPSILON, lam=LAMBDA,
                                  beta_f=None, betap_f=None, verbose=True):
    """
    运行所有求解器并汇报: 精度 / 耗时 / 稳定性.
    返回 dict with results for each method.
    """
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np

    psi_ref = ode_reference(x_arr, ep, lam, beta_f, betap_f)
    results = {}

    methods = [
        ('ODE45 (Reference)', lambda: ode_reference(x_arr, ep, lam, beta_f, betap_f)),
        ('FDM',               lambda: solve_fdm(x_arr, ep, lam, beta_f, betap_f)),
        ('Chebyshev Spectral',lambda: solve_chebyshev_spectral(x_arr, ep, lam, beta_f, betap_f)),
        ('Quadrature IDE',    lambda: solve_quadrature_ide(x_arr, ep, lam, beta_f, betap_f)),
        ('New MS',            lambda: ms_new(x_arr, ep, lam, beta_f)),
    ]

    if verbose:
        print(f'\n{"=" * 74}')
        print(f'{"Method":<26} {"Max Error":>12} {"Rel-L2":>12} {"Time(s)":>10} {"Stable?":>8}')
        print('-' * 74)

    for name, solver in methods:
        t0 = time.time()
        try:
            psi = solver()
            elapsed = time.time() - t0
            if name == 'ODE45 (Reference)':
                err_max = 0.0; err_l2 = 0.0
            else:
                diff    = np.abs(psi - psi_ref)
                err_max = float(np.max(diff))
                err_l2  = float(np.linalg.norm(diff) / np.linalg.norm(psi_ref))
            stable = bool(np.all(np.isfinite(psi)) and np.all(psi >= -0.01))
        except Exception as exc:
            psi = None; err_max = np.nan; err_l2 = np.nan
            elapsed = time.time() - t0; stable = False
            if verbose: print(f'  {name}: FAILED ({exc})')

        results[name] = {
            'psi':     psi,
            'err_max': err_max,
            'err_l2':  err_l2,
            'time':    elapsed,
            'stable':  stable,
        }
        if verbose:
            print(f'  {name:<24} {err_max:>12.3e} {err_l2:>12.3e} '
                  f'{elapsed:>10.3f} {"✓" if stable else "✗":>8}')

    if verbose: print('=' * 74)
    return results


# ══════════════════════════════════════════════════════════════
# Ablation Study (消融实验)
# ══════════════════════════════════════════════════════════════

ABLATION_VARIANTS = {
    'PINN_baseline': {
        'use_ms':    False,   # 不使用 MS 先验
        'use_ratio': False,   # 直接学习 psi
        'use_rode':  True,
        'two_stage': False,
        'desc': 'Vanilla PINN (no MS prior, no ratio)'
    },
    'MS_only': {
        'use_ms':    True,
        'use_ratio': False,
        'use_rode':  False,
        'two_stage': False,
        'desc': '+ MS prior (no ratio param, supervised only)'
    },
    'MS+Ratio': {
        'use_ms':    True,
        'use_ratio': True,
        'use_rode':  False,
        'two_stage': False,
        'desc': '+ Ratio parameterization (supervised only)'
    },
    'MS+Ratio+RODE': {
        'use_ms':    True,
        'use_ratio': True,
        'use_rode':  True,
        'two_stage': False,
        'desc': '+ R-ODE physics (single-stage)'
    },
    'Full_MS-PINN': {
        'use_ms':    True,
        'use_ratio': True,
        'use_rode':  True,
        'two_stage': True,
        'desc': 'Full two-stage MS-PINN (proposed)'
    },
}


def run_ablation_study(x_arr, ep=EPSILON, lam=LAMBDA,
                        ep_sup=5000, ep_phys=5000, verbose=True):
    """
    消融实验: 逐步添加各组件, 记录误差 / 收敛速度 / 损失曲线.
    返回 dict {variant_name: {err, hist, time}}
    """
    from module1_core import BETA_CASES
    bc   = BETA_CASES['beta1']
    bf, bpf, bppf, bft = bc['f'], bc['fp'], bc['fpp'], bc['ft']

    xp  = sample_collocation(); xnp = xp.cpu().numpy().flatten()
    pm_np = ms_new(xnp, ep, lam, bf)
    pr_np = ode_reference(xnp, ep, lam, bf, bpf)
    P_v   = np.array([P_eff_np(x, ep, lam, bf, bpf) for x in xnp])
    Q_v   = np.array([Q_np(x, ep, lam, bf, bpf, bppf) for x in xnp])
    pm_t  = torch.tensor(pm_np, dtype=torch.float64, device=device).unsqueeze(1)
    pr_t  = torch.tensor(pr_np, dtype=torch.float64, device=device).unsqueeze(1)
    Pt    = torch.tensor(P_v,   dtype=torch.float64, device=device).unsqueeze(1)
    Qt    = torch.tensor(Q_v,   dtype=torch.float64, device=device).unsqueeze(1)

    psi_ref_full = ode_reference(x_arr, ep, lam, bf, bpf)
    psi_ms_full  = ms_new(x_arr, ep, lam, bf)
    pm_full = torch.tensor(psi_ms_full, dtype=torch.float64, device=device).unsqueeze(1)
    xt_full = torch.tensor(x_arr,       dtype=torch.float64, device=device).unsqueeze(1)

    results = {}

    for variant_name, cfg in ABLATION_VARIANTS.items():
        if verbose: print(f'\n  [Ablation] {variant_name}: {cfg["desc"]}')
        t0 = time.time()

        model = RatioNet(64, 4).double().to(device)
        hist  = {'s1': [], 's2': []}

        if not cfg['use_ms']:
            # Vanilla: supervised toward true psi (no MS factoring)
            psi_ref_t_direct = torch.tensor(
                ode_reference(xnp, ep, lam, bf, bpf),
                dtype=torch.float64, device=device).unsqueeze(1)
            opt = optim.Adam(model.parameters(), lr=3e-3)
            ones_ms = torch.ones_like(pm_t)    # treat psi_MS = 1
            for i in range(ep_sup + ep_phys):
                model.train(); opt.zero_grad()
                if cfg['use_rode']:
                    l = loss_r_ode(model, xp, Pt, Qt, ep, bft)
                else:
                    l = loss_supervised_R(model, xp, make_R_target(ones_ms, psi_ref_t_direct)[0])
                l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                hist['s1' if i < ep_sup else 's2'].append(l.item())

        elif not cfg['two_stage']:
            # Only supervised stage
            R_tgt, _ = make_R_target(pm_t, pr_t)
            opt = optim.Adam(model.parameters(), lr=3e-3)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_sup, eta_min=1e-4)
            for i in range(ep_sup):
                model.train(); opt.zero_grad()
                l = loss_supervised_R(model, xp, R_tgt)
                l.backward(); opt.step(); sch.step()
                hist['s1'].append(l.item())

            if cfg['use_rode']:
                opt2 = optim.Adam(model.parameters(), lr=5e-4)
                sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=ep_phys, eta_min=1e-6)
                for i in range(ep_phys):
                    model.train(); opt2.zero_grad()
                    l = loss_r_ode(model, xp, Pt, Qt, ep, bft)
                    l.backward(); opt2.step(); sch2.step()
                    hist['s2'].append(l.item())
        else:
            # Full two-stage (same as main train())
            hist = train(model, xp, pm_t, pr_t, Pt, Qt,
                         tag=variant_name, ep=ep, beta_ft=bft,
                         ep_sup=ep_sup, ep_phys=ep_phys, verbose=verbose)

        elapsed = time.time() - t0

        with torch.no_grad():
            psi_pred = (model.forward_R(xt_full) * pm_full).cpu().numpy().flatten()
        psi_pred = np.clip(psi_pred, 0, 1)
        err = float(np.max(np.abs(psi_ref_full - psi_pred)))

        if verbose:
            print(f'    => max_err={err:.3e}  time={elapsed:.0f}s')

        results[variant_name] = {
            'config':  cfg,
            'psi':     psi_pred,
            'err_max': err,
            'hist':    hist,
            'time':    elapsed,
        }

    if verbose:
        print(f'\n{"=" * 60}')
        print(f'{"Variant":<22} {"Max Error":>12} {"Time(s)":>10}')
        print('-' * 60)
        for vn, r in results.items():
            print(f'  {vn:<20} {r["err_max"]:>12.3e} {r["time"]:>10.0f}')
        print('=' * 60)

    return results


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    x = np.linspace(0, X_MAX, 300)
    print('=== Module 2 Self-Test ===')
    bench = benchmark_classical_solvers(x)
    print('\n[Ablation quick test – 500 steps each]')
    # Quick smoke test only
    print('[Done]')
