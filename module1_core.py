"""
================================================================
模块1: 核心数学函数 / Core Mathematical Functions
破产概率联合求解框架
================================================================
包含:
  - 全局参数
  - β(x)函数族（含非平滑/随机类型）
  - 经典解析解: WKB, MS新旧版, ms_order1
  - R-ODE系数: D1, D2, P_eff, Q
  - 参考解: ode_reference
  - R-ODE残差验证
================================================================
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 全局参数
# ══════════════════════════════════════════════════════════════
LAMBDA  = 1.2
EPSILON = 0.1
X_MAX   = 5.0
EP_SUP  = 10000
EP_PHYS = 20000

# ══════════════════════════════════════════════════════════════
# β(x) 函数族（v10 新增 discontinuous / sharp-gradient / random）
# ══════════════════════════════════════════════════════════════
np.random.seed(42)
_RNG_KNOTS  = np.sort(np.random.uniform(0, X_MAX, 8))
_RNG_VALUES = 1.5 + 0.6 * np.random.uniform(-1, 1, 8)

def _random_beta(x):
    return float(np.interp(x, _RNG_KNOTS, _RNG_VALUES))

def _random_betap(x, h=1e-5):
    return (_random_beta(x+h) - _random_beta(x-h)) / (2*h)

BETA_CASES = {
    'beta1': {
        'name':    r'$\beta_1=1.5+0.3\sin x$ (基准)',
        'name_en': r'$\beta_1=1.5+0.3\sin x$',
        'f':   lambda x: 1.5 + 0.3*np.sin(x),
        'fp':  lambda x: 0.3*np.cos(x),
        'fpp': lambda x: -0.3*np.sin(x),
        'ft':  lambda x: 1.5 + 0.3*__import__('torch').sin(x),
        'fpt': lambda x: 0.3*__import__('torch').cos(x),
    },
    'beta2': {
        'name':    r'$\beta_2=2+\cos x$ (高均值)',
        'name_en': r'$\beta_2=2+\cos x$',
        'f':   lambda x: 2.0 + np.cos(x),
        'fp':  lambda x: -np.sin(x),
        'fpp': lambda x: -np.cos(x),
        'ft':  lambda x: 2.0 + __import__('torch').cos(x),
        'fpt': lambda x: -__import__('torch').sin(x),
    },
    'beta3': {
        'name':    r'$\beta_3=1.5+0.5x/(1+x)$ (单调递增)',
        'name_en': r'$\beta_3=1.5+0.5x/(1+x)$',
        'f':   lambda x: 1.5 + 0.5*x/(1.0+x),
        'fp':  lambda x: 0.5/(1.0+x)**2,
        'fpp': lambda x: -1.0/(1.0+x)**3,
        'ft':  lambda x: 1.5 + 0.5*x/(1.0+x),
        'fpt': lambda x: 0.5/(1.0+x)**2,
    },
    'beta4': {
        'name':    r'$\beta_4=1.5+0.5\sin 2x$ (高频振荡)',
        'name_en': r'$\beta_4=1.5+0.5\sin 2x$',
        'f':   lambda x: 1.5 + 0.5*np.sin(2*x),
        'fp':  lambda x: np.cos(2*x),
        'fpp': lambda x: -2.0*np.sin(2*x),
        'ft':  lambda x: 1.5 + 0.5*__import__('torch').sin(2*x),
        'fpt': lambda x: __import__('torch').cos(2*x),
    },
    # ─── v10 新增3种 ─────────────────────────────────────────
    'beta5': {
        'name':    r'$\beta_5$ discontinuous (阶跃)',
        'name_en': r'$\beta_5$: step discontinuity',
        # vectorized: scalar → float, array → ndarray
        'f':   lambda x: (np.where(np.asarray(x, dtype=float) < X_MAX/2, 2.0, 1.2)
                          if not np.isscalar(x)
                          else (2.0 if float(x) < X_MAX/2 else 1.2)),
        'fp':  lambda x: (np.zeros_like(np.asarray(x, dtype=float))
                          if not np.isscalar(x) else 0.0),
        'fpp': lambda x: (np.zeros_like(np.asarray(x, dtype=float))
                          if not np.isscalar(x) else 0.0),
        'ft':  lambda x: __import__('torch').where(
                    x < X_MAX/2,
                    __import__('torch').full_like(x, 2.0),
                    __import__('torch').full_like(x, 1.2)),
        'fpt': lambda x: __import__('torch').zeros_like(x),
    },
    'beta6': {
        'name':    r'$\beta_6=1.5+0.4\tanh(10(x-2))$ (sharp gradient)',
        'name_en': r'$\beta_6$: sharp gradient',
        'f':   lambda x: 1.5 + 0.4*np.tanh(10*(np.asarray(x, dtype=float) - 2.0)),
        'fp':  lambda x: 4.0 / np.cosh(10*(np.asarray(x, dtype=float) - 2.0))**2,
        'fpp': lambda x: (-80.0*np.tanh(10*(np.asarray(x, dtype=float)-2.0))
                          / np.cosh(10*(np.asarray(x, dtype=float)-2.0))**2),
        'ft':  lambda x: 1.5 + 0.4*__import__('torch').tanh(10*(x - 2.0)),
        'fpt': lambda x: 4.0 / __import__('torch').cosh(10*(x - 2.0))**2,
    },
    'beta7': {
        'name':    r'$\beta_7$ random piecewise (随机)',
        'name_en': r'$\beta_7$: random piecewise',
        'f':   lambda x: (float(_random_beta(float(x))) if np.isscalar(x)
                          else np.array([_random_beta(float(xi))
                                         for xi in np.asarray(x).flatten()],
                                         dtype=float).reshape(np.asarray(x).shape)),
        'fp':  lambda x: (float(_random_betap(float(x))) if np.isscalar(x)
                          else np.array([_random_betap(float(xi))
                                         for xi in np.asarray(x).flatten()],
                                         dtype=float).reshape(np.asarray(x).shape)),
        'fpp': lambda x: (0.0 if np.isscalar(x)
                          else np.zeros_like(np.asarray(x, dtype=float))),
        'ft':  lambda x: __import__('torch').tensor(
                    [_random_beta(float(xi)) for xi in x.detach().cpu().flatten().tolist()],
                    dtype=x.dtype, device=x.device).reshape(x.shape),
        'fpt': lambda x: __import__('torch').tensor(
                    [_random_betap(float(xi)) for xi in x.detach().cpu().flatten().tolist()],
                    dtype=x.dtype, device=x.device).reshape(x.shape),
    },
}

# 默认 β 快捷方式
def beta_np(x):   return BETA_CASES['beta1']['f'](x)
def betap_np(x):  return BETA_CASES['beta1']['fp'](x)
def betapp_np(x): return BETA_CASES['beta1']['fpp'](x)

import torch
def beta_t(x):  return BETA_CASES['beta1']['ft'](x)
def betap_t(x): return BETA_CASES['beta1']['fpt'](x)

# ══════════════════════════════════════════════════════════════
# 基础积分函数
# ══════════════════════════════════════════════════════════════

def S_scalar(xv, beta_f=None, lam=LAMBDA):
    if beta_f is None: beta_f = beta_np
    if xv <= 0: return 0.0
    v, _ = quad(lambda t: 1/beta_f(t) - lam, 0, xv, limit=300)
    return v

def compute_G_inf(ep=EPSILON, lam=LAMBDA, beta_f=None):
    if beta_f is None: beta_f = beta_np
    G, _ = quad(
        lambda s: np.exp(S_scalar(s, beta_f, lam)/ep)/beta_f(s),
        0, 20, limit=600, epsabs=1e-14)
    return G

# ══════════════════════════════════════════════════════════════
# 参考解（RK45）
# ══════════════════════════════════════════════════════════════

def ode_reference(x_arr, ep=EPSILON, lam=LAMBDA, beta_f=None, betap_f=None):
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np
    G  = compute_G_inf(ep, lam, beta_f)
    p0 = -1.0 / (beta_f(0) * G)
    rhs = lambda t, y: [
        y[1],
        -(ep*betap_f(t) + lam*beta_f(t) - 1) / (ep*beta_f(t)) * y[1]
    ]
    sol = solve_ivp(rhs, [0, x_arr[-1]], [1.0, p0],
                    t_eval=x_arr, method='RK45', rtol=1e-12, atol=1e-14)
    return sol.y[0]

# ══════════════════════════════════════════════════════════════
# WKB 精确解 / MS 各阶次
# ══════════════════════════════════════════════════════════════

def wkb_exact(x_arr, ep=EPSILON, lam=LAMBDA, beta_f=None):
    if beta_f is None: beta_f = beta_np
    G  = compute_G_inf(ep, lam, beta_f)
    Gx = np.array([quad(
        lambda s: np.exp(S_scalar(s, beta_f, lam)/ep)/beta_f(s),
        0, xi, limit=200)[0] for xi in x_arr])
    return np.maximum(1 - Gx/G, 0.0)

def ms_old(x_arr, ep=EPSILON, lam=LAMBDA, beta_f=None):
    if beta_f is None: beta_f = beta_np
    S = np.array([S_scalar(x, beta_f, lam) for x in x_arr])
    A = np.sqrt(beta_f(0) / beta_f(x_arr))
    psi = A * np.exp(S/ep); psi /= psi[0]
    return np.maximum(psi, 0.0)

def ms_new(x_arr, ep=EPSILON, lam=LAMBDA, beta_f=None):
    if beta_f is None: beta_f = beta_np
    S = np.array([S_scalar(x, beta_f, lam) for x in x_arr])
    A = (lam*beta_f(0) - 1) / (lam*beta_f(x_arr) - 1)
    psi = A * np.exp(S/ep)
    return np.maximum(psi / psi[0], 0.0)

def ms_order1(x_arr, ep=EPSILON, lam=LAMBDA, beta_f=None, betap_f=None):
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np
    S  = np.array([S_scalar(x, beta_f, lam) for x in x_arr])
    eS = np.exp(S/ep)
    b  = beta_f(x_arr); bp = betap_f(x_arr); Sp = 1/b - lam
    Psi0 = 1/(1 - lam*b)
    Psi1 = -(-lam*bp/(1 - lam*b)**2)/Sp
    P0_0 = 1/(1 - lam*beta_f(0)); P1_0 = Psi1[0]
    G_inf0 = -ep * P0_0
    G_inf1 = -ep * P0_0 - ep**2 * P1_0
    G_x0   = ep*(Psi0*eS - P0_0)
    G_x1   = ep*(Psi0*eS - P0_0) + ep**2*(Psi1*eS - P1_0)
    psi_n0 = np.maximum(1 - G_x0/G_inf0, 0.0)
    psi_n1 = np.maximum(1 - G_x1/G_inf1, 0.0)
    if psi_n0[0] > 0: psi_n0 /= psi_n0[0]
    if psi_n1[0] > 0: psi_n1 /= psi_n1[0]
    return psi_n0, psi_n1, P0_0, P1_0, G_inf0, G_inf1

def ms_inner(x_arr, ep=EPSILON, lam=LAMBDA, beta_f=None):
    if beta_f is None: beta_f = beta_np
    r_x = 1/beta_f(x_arr) - lam
    C   = (lam*beta_f(0)-1)/(lam*beta_f(x_arr)-1)
    R0  = 1 - C*np.exp(r_x*x_arr/ep)
    psi = 1 - R0; psi = np.maximum(psi, 0.0)
    if psi[0] > 0: psi /= psi[0]
    return psi

# ══════════════════════════════════════════════════════════════
# R-ODE 系数
# ══════════════════════════════════════════════════════════════

def D1_np(x, ep=EPSILON, lam=LAMBDA, beta_f=None, betap_f=None):
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np
    b=beta_f(x); bp=betap_f(x)
    return (1/b - lam)/ep - lam*bp/(lam*b - 1)

def D2_np(x, ep=EPSILON, lam=LAMBDA, beta_f=None, betap_f=None, betapp_f=None):
    if beta_f is None:   beta_f   = beta_np
    if betap_f is None:  betap_f  = betap_np
    if betapp_f is None: betapp_f = betapp_np
    b=beta_f(x); bp=betap_f(x); bpp=betapp_f(x)
    return -bp/(ep*b**2) - lam*bpp/(lam*b-1) + lam**2*bp**2/(lam*b-1)**2

def P_eff_np(x, ep=EPSILON, lam=LAMBDA, beta_f=None, betap_f=None):
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np
    b=beta_f(x); bp=betap_f(x)
    d1 = D1_np(x, ep, lam, beta_f, betap_f)
    return 2*ep*b*d1 + ep*bp + lam*b - 1

def Q_np(x, ep=EPSILON, lam=LAMBDA, beta_f=None, betap_f=None, betapp_f=None):
    if beta_f is None:   beta_f   = beta_np
    if betap_f is None:  betap_f  = betap_np
    if betapp_f is None: betapp_f = betapp_np
    b=beta_f(x); bp=betap_f(x)
    d1 = D1_np(x, ep, lam, beta_f, betap_f)
    d2 = D2_np(x, ep, lam, beta_f, betap_f, betapp_f)
    return ep*b*(d2 + d1**2) + (ep*bp + lam*b - 1)*d1

# ══════════════════════════════════════════════════════════════
# R-ODE 残差验证
# ══════════════════════════════════════════════════════════════

def verify_r_ode(silent=False, beta_f=None, betap_f=None, betapp_f=None):
    if beta_f is None:   beta_f   = beta_np
    if betap_f is None:  betap_f  = betap_np
    if betapp_f is None: betapp_f = betapp_np
    h = 1e-6
    G  = compute_G_inf(EPSILON, LAMBDA, beta_f)
    p0 = -1 / (beta_f(0)*G)

    def ms_sc(x):
        A = (LAMBDA*beta_f(0)-1)/(LAMBDA*beta_f(x)-1) if x > 1e-8 else 1.0
        return A * np.exp(S_scalar(x, beta_f, LAMBDA)/EPSILON)

    def R_f(x):
        pr = solve_ivp(
            lambda t, y: [y[1], -(EPSILON*betap_f(t)+LAMBDA*beta_f(t)-1)
                          /(EPSILON*beta_f(t))*y[1]],
            [0, max(x, 1e-6)], [1.0, p0], t_eval=[max(x, 1e-6)],
            method='RK45', rtol=1e-12, atol=1e-14).y[0][0]
        return pr / ms_sc(x)

    maxres = 0.0
    for xv in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
        Rp  = (R_f(xv+h) - R_f(xv-h)) / (2*h)
        Rpp = (R_f(xv+h) - 2*R_f(xv) + R_f(xv-h)) / h**2
        res = (EPSILON*beta_f(xv)*Rpp
               + P_eff_np(xv, EPSILON, LAMBDA, beta_f, betap_f)*Rp
               + Q_np(xv, EPSILON, LAMBDA, beta_f, betap_f, betapp_f)*R_f(xv))
        maxres = max(maxres, abs(res))
    if not silent:
        print(f'  R-ODE max residual={maxres:.2e}  (expected <5e-3)')
    return maxres


# ══════════════════════════════════════════════════════════════
# v10 新增: 算子适定性验证 (coercivity / well-posedness)
# ══════════════════════════════════════════════════════════════

def verify_operator_coercivity(x_arr=None, ep=EPSILON, lam=LAMBDA,
                                beta_f=None, betap_f=None):
    """
    验证R-ODE传输算子 L[R] = eps*beta*R'' + P_eff*R' + Q*R
    的稳定性条件:
      (i)  P_eff(x) < 0  everywhere  (传输方向稳定)
      (ii) Q(x) bounded               (强迫有界)
    返回 dict with diagnostic values.
    """
    if x_arr is None:
        x_arr = np.linspace(1e-4, X_MAX, 1000)
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np

    P_arr = np.array([P_eff_np(x, ep, lam, beta_f, betap_f) for x in x_arr])
    Q_arr = np.array([Q_np(x, ep, lam, beta_f, betap_f) for x in x_arr])
    b_arr = beta_f(x_arr)

    coercive = bool(np.all(P_arr < 0))
    result = {
        'P_eff_max':   float(P_arr.max()),
        'P_eff_min':   float(P_arr.min()),
        'P_eff_negative_everywhere': coercive,
        'Q_norm_inf':  float(np.max(np.abs(Q_arr))),
        'beta_min':    float(b_arr.min()),
        'beta_max':    float(b_arr.max()),
        'ep_beta_min': float(ep * b_arr.min()),
    }
    print(f'\n[Operator Well-posedness @ eps={ep}, lam={lam}]')
    print(f'  P_eff in [{result["P_eff_min"]:.4f}, {result["P_eff_max"]:.4f}]'
          f'  => coercive: {coercive}')
    print(f'  ||Q||_inf = {result["Q_norm_inf"]:.4e}')
    print(f'  eps*beta_min = {result["ep_beta_min"]:.4e}  (diffusion lower bound)')
    return result
# ══════════════════════════════════════════════════════════════
#三项误差分解
# ══════════════════════════════════════════════════════════════

def error_decomposition(x_arr, ep=EPSILON, lam=LAMBDA,
                        beta_f=None, betap_f=None,
                        m_values=None):
    """
    三项误差界估计:
      ||psi - psi_theta|| <= E_asym + E_approx + E_optim
    其中:
      E_asym  ~ O(eps^2)          MS渐近误差
      E_approx~ O(m^{-1/2})       网络近似误差 (Barron)
      E_optim ~ 训练残差           优化误差（数值量化）

    返回 dict 含各项估计值（以实际误差为基准）.
    """
    if beta_f is None:  beta_f  = beta_np
    if betap_f is None: betap_f = betap_np
    if m_values is None: m_values = [16, 32, 64, 128, 256]

    psi_ref = ode_reference(x_arr, ep, lam, beta_f, betap_f)
    psi_ms  = ms_new(x_arr, ep, lam, beta_f)

    E_asym   = float(np.max(np.abs(psi_ref - psi_ms)))        # O(eps^2) term
    # Barron bound: O(C_f / sqrt(m)) where C_f is first-moment norm
    # We estimate C_f from the R-ODE forcing Q
    Q_arr = np.array([Q_np(x, ep, lam, beta_f, betap_f) for x in x_arr])
    C_f_est = float(np.linalg.norm(Q_arr) / np.sqrt(len(x_arr)))
    approx_bounds = {m: C_f_est / np.sqrt(m) for m in m_values}

    print(f'\n[Error Decomposition @ eps={ep}]')
    print(f'  E_asym  (O(eps^2)) = {E_asym:.3e}   [MS asymptotic error]')
    print(f'  C_f estimate       = {C_f_est:.3e}   [Barron first-moment norm]')
    for m, eb in approx_bounds.items():
        print(f'  E_approx(m={m:4d})  = {eb:.3e}   [O(C_f/sqrt(m))]')
    print(f'  E_optim            = training loss  [measured during training]')

    return {
        'E_asym':         E_asym,
        'C_f':            C_f_est,
        'approx_bounds':  approx_bounds,
        'eps_order2':     ep**2,
    }


if __name__ == '__main__':
    x = np.linspace(0, X_MAX, 600)
    print('=== Module 1 Self-Test ===')
    ref = ode_reference(x)
    ms  = ms_new(x)
    print(f'  MS new max error: {np.max(np.abs(ref-ms)):.3e}')
    verify_r_ode()
    verify_operator_coercivity(x)
    error_decomposition(x)
    print('[Done]')
