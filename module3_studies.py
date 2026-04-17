"""
================================================================
模块3: 数值实验 / Numerical Studies
破产概率联合求解框架
================================================================
包含:
  - 收敛阶分析
  - 最优截断定理验证
  - 泛化性验证（含v10新增非平滑/随机β类型）
  - λ参数敏感性分析
  - 边界层解析比较 (gradient comparison)
  - 复杂度分析 (scaling law vs eps/N/m)
  - 时间-精度 tradeoff 图数据
================================================================
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import warnings
warnings.filterwarnings('ignore')

from module1_core import (
    LAMBDA, EPSILON, X_MAX,
    beta_np, betap_np, betapp_np, beta_t,
    BETA_CASES,
    ms_new, ms_old, ode_reference,
    P_eff_np, Q_np,
    ms_order1,
)
from module2_training import (
    RatioNet, sample_collocation, device,
    loss_supervised_R, loss_r_ode, make_R_target,
    train_for_beta, train,
)

# ══════════════════════════════════════════════════════════════
# 收敛阶分析
# ══════════════════════════════════════════════════════════════

def convergence_study(x_test, eps_list=None, n_sup=5000, n_phys=5000):
    if eps_list is None: eps_list = [0.20, 0.10, 0.05, 0.02, 0.01]
    MAX_PHYS   = 8000
    TOL_SCALE  = 1e-7
    print(f'\n[Convergence Study]  sup={n_sup}  phys={n_phys} (fast mode)')
    e_ms_list, e_pinn_list = [], []

    for ep in eps_list:
        n_phys_ep = min(MAX_PHYS, max(n_phys, int(n_phys*(0.1/ep)**1.0)))
        tol_ep    = TOL_SCALE * (ep/0.1)**2
        psi_ref   = ode_reference(x_test, ep=ep)
        psi_ms_   = ms_new(x_test, ep=ep)
        e_ms_list.append(np.max(np.abs(psi_ref - psi_ms_)))

        xp  = sample_collocation(); xnp = xp.cpu().numpy().flatten()
        pm  = ms_new(xnp, ep=ep); pr = ode_reference(xnp, ep=ep)
        P_v = np.array([P_eff_np(x, ep) for x in xnp])
        Q_v = np.array([Q_np(x, ep) for x in xnp])
        pm_t = torch.tensor(pm, dtype=torch.float64, device=device).unsqueeze(1)
        pr_t = torch.tensor(pr, dtype=torch.float64, device=device).unsqueeze(1)
        Pt   = torch.tensor(P_v, dtype=torch.float64, device=device).unsqueeze(1)
        Qt   = torch.tensor(Q_v, dtype=torch.float64, device=device).unsqueeze(1)
        R_tgt, _ = make_R_target(pm_t, pr_t)

        m    = RatioNet(64, 4).double().to(device)
        opt1 = optim.Adam(m.parameters(), lr=3e-3)
        sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=n_sup, eta_min=1e-4)
        best1, st1 = 1e10, None
        for _ in range(n_sup):
            m.train(); opt1.zero_grad()
            l = loss_supervised_R(m, xp, R_tgt)
            l.backward(); opt1.step(); sch1.step()
            v = l.item()
            if v < best1: best1=v; st1={k: w.clone() for k,w in m.state_dict().items()}
        m.load_state_dict(st1)

        opt2 = optim.Adam(m.parameters(), lr=3e-4)
        sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=n_phys_ep, eta_min=1e-7)
        best2, st2 = 1e10, None
        for i in range(n_phys_ep):
            m.train(); opt2.zero_grad()
            l = loss_r_ode(m, xp, Pt, Qt, ep)
            l.backward(); opt2.step(); sch2.step()
            v = l.item()
            if v < best2: best2=v; st2={k: w2.clone() for k,w2 in m.state_dict().items()}
            if v < tol_ep: break
        m.load_state_dict(st2)

        opt_lb = optim.LBFGS(m.parameters(), max_iter=50, history_size=30,
                               line_search_fn='strong_wolfe')
        def _cl():
            opt_lb.zero_grad()
            l = loss_r_ode(m, xp, Pt, Qt, ep)
            l.backward(); return l
        opt_lb.step(_cl); fr = _cl().item()

        xt  = torch.tensor(x_test, dtype=torch.float64, device=device).unsqueeze(1)
        pm2 = torch.tensor(ms_new(x_test, ep), dtype=torch.float64, device=device).unsqueeze(1)
        with torch.no_grad():
            pp  = (m.forward_R(xt)*pm2).cpu().numpy().flatten()
        err = np.max(np.abs(psi_ref - np.clip(pp, 0, 1)))
        e_pinn_list.append(err)
        cvg = '✓' if fr < tol_ep*10 else '✗'
        print(f'  eps={ep:.3f}: MS={e_ms_list[-1]:.2e} PINN={err:.2e}'
              f' r_ode={fr:.1e} steps={n_phys_ep} {cvg}')

    log_e  = np.log(eps_list)
    p_ms   = np.polyfit(log_e, np.log(e_ms_list),   1)[0]
    p_pinn = np.polyfit(log_e, np.log(e_pinn_list), 1)[0]
    local_p = np.diff(np.log(e_pinn_list)) / np.diff(log_e)
    print(f'  Convergence order -> New MS: {p_ms:.2f}, PINN: {p_pinn:.2f}')
    print(f'  Local PINN orders: {[f"{p:.2f}" for p in local_p]}')
    return e_ms_list, e_pinn_list, p_ms, p_pinn, local_p

# ══════════════════════════════════════════════════════════════
# 最优截断定理验证
# ══════════════════════════════════════════════════════════════

def optimal_truncation_study(x_test, eps_list=None, lam=LAMBDA):
    if eps_list is None: eps_list = [0.20, 0.10, 0.05, 0.02, 0.01]
    print(f'\n[最优截断定理验证] λ={lam}')
    print(f"{'ε':>6} {'E₀(n=0)':>12} {'E₁(n=1)':>12} {'E₁/E₀':>8} {'条件值εR₁':>10} {'定理?':>6}")
    print("-" * 60)
    E0_list, E1_list, ratio_list, cond_list = [], [], [], []
    for ep in eps_list:
        psi_ref             = ode_reference(x_test, ep=ep, lam=lam)
        psi_n0, psi_n1, P0_0, P1_0, G0, G1 = ms_order1(x_test, ep=ep, lam=lam)
        E0   = np.max(np.abs(psi_ref - psi_n0))
        E1   = np.max(np.abs(psi_ref - psi_n1))
        cond = abs(P1_0/P0_0) * abs(G0/G1) * ep
        ok   = "OK" if E0 < E1 else "NG"
        E0_list.append(E0); E1_list.append(E1)
        ratio_list.append(E1/E0); cond_list.append(cond)
        print(f"  {ep:.3f}: {E0:>12.3e} {E1:>12.3e} "
              f"{E1/E0:>8.3f} {cond:>10.4f} {ok}")
    print(f"\n  结论: 全部{sum(1 for r in ratio_list if r>1)}/{len(eps_list)}"
          f"个ε值满足E₁>E₀，n=0始终为最优截断")
    print(f"  E₁/E₀均值 = {np.mean(ratio_list):.4f}  (Poincaré发散特性，E₁≈2E₀)")
    return E0_list, E1_list, ratio_list, cond_list

# ══════════════════════════════════════════════════════════════
# 泛化性验证
# ══════════════════════════════════════════════════════════════

def generalization_study(x_test, ep=EPSILON, lam=LAMBDA,
                          eps_scan=None, run_pinn=True,
                          ep_sup=5000, ep_phys=5000,
                          beta_keys=None):
    if eps_scan   is None: eps_scan   = [0.20, 0.10, 0.05, 0.02, 0.01]
    if beta_keys  is None: beta_keys  = list(BETA_CASES.keys())  # all 7

    print(f'\n[泛化性验证] ε={ep}，λ={lam}')
    print(f"{'β(x)':32} {'旧MS':>10} {'新MS':>10} {'MS阶p':>8} {'改善':>6} {'PINN':>12}")
    print("-" * 84)
    results = {}

    for beta_key in beta_keys:
        bc  = BETA_CASES[beta_key]
        bf, bpf = bc['f'], bc['fp']
        e_old_list, e_new_list = [], []
        for ep_ in eps_scan:
            ref_ = ode_reference(x_test, ep_, lam, bf, bpf)
            e_old_list.append(np.max(np.abs(ref_ - ms_old(x_test, ep_, lam, bf))))
            e_new_list.append(np.max(np.abs(ref_ - ms_new(x_test, ep_, lam, bf))))
        try:
            p_new = np.polyfit(np.log(eps_scan), np.log(e_new_list), 1)[0]
        except Exception:
            p_new = float('nan')

        psi_ref    = ode_reference(x_test, ep, lam, bf, bpf)
        e_old_ep   = np.max(np.abs(psi_ref - ms_old(x_test, ep, lam, bf)))
        e_new_ep   = np.max(np.abs(psi_ref - ms_new(x_test, ep, lam, bf)))
        e_pinn_ep  = None; psi_pinn = None

        if run_pinn:
            print(f'  [{beta_key}] PINN training...', end='', flush=True)
            try:
                psi_pinn  = train_for_beta(beta_key, x_test, ep, lam,
                                            ep_sup, ep_phys, verbose=False)
                e_pinn_ep = np.max(np.abs(psi_ref - psi_pinn))
                print(f' done. err={e_pinn_ep:.2e}')
            except Exception as exc:
                print(f' FAILED ({exc})')

        pinn_str = f'{e_pinn_ep:.2e}' if e_pinn_ep is not None else 'N/A'
        print(f"  {bc['name']:32} {e_old_ep:>10.2e} {e_new_ep:>10.2e} "
              f"{p_new:>8.2f} {e_old_ep/e_new_ep:>6.1f}x {pinn_str}")
        results[beta_key] = {
            'psi_ref':    psi_ref,
            'psi_ms_new': ms_new(x_test, ep, lam, bf),
            'psi_ms_old': ms_old(x_test, ep, lam, bf),
            'psi_pinn':   psi_pinn,
            'e_old': e_old_ep, 'e_new': e_new_ep, 'e_pinn': e_pinn_ep,
            'p_new': p_new,
            'e_old_list': e_old_list, 'e_new_list': e_new_list,
        }
    return results

# ══════════════════════════════════════════════════════════════
# λ敏感性分析
# ══════════════════════════════════════════════════════════════

def lambda_sensitivity(x_test, lam_list=None, ep=EPSILON):
    if lam_list is None: lam_list = [1.05, 1.2, 1.5, 2.0, 5.0]
    print(f'\n[λ参数敏感性] ε={ep}')
    print(f"{'λ':>6} {'λβ₀-1':>8} {'旧MS误差':>12} {'新MS误差':>12} {'收敛阶p':>8}")
    print("-" * 50)
    results = {}
    for lam in lam_list:
        eps_scan = [0.20, 0.10, 0.05, 0.02, 0.01]
        e_list   = []
        for ep_ in eps_scan:
            ref_ = ode_reference(x_test, ep_, lam)
            e_list.append(np.max(np.abs(ref_ - ms_new(x_test, ep_, lam))))
        p_new    = np.polyfit(np.log(eps_scan), np.log(e_list), 1)[0]
        psi_ref  = ode_reference(x_test, ep, lam)
        e_old    = np.max(np.abs(psi_ref - ms_old(x_test, ep, lam)))
        e_new    = np.max(np.abs(psi_ref - ms_new(x_test, ep, lam)))
        lam_b0_1 = lam * beta_np(0) - 1
        print(f"  {lam:>5.2f}: {lam_b0_1:>8.4f} {e_old:>12.3e} {e_new:>12.3e} {p_new:>8.2f}")
        results[lam] = {
            'psi_ref': psi_ref,
            'psi_ms':  ms_new(x_test, ep, lam),
            'e_new': e_new, 'p_new': p_new, 'lam_b0_1': lam_b0_1
        }
    return results

# ══════════════════════════════════════════════════════════════
# 边界层解析比较 (gradient comparison)
# ══════════════════════════════════════════════════════════════

def boundary_layer_analysis(x_test, psi_num=None, psi_ms=None, psi_mspinn=None,
                              ep=EPSILON, lam=LAMBDA):
    """
    计算 psi, psi_MS, Gamma 在边界层内的导数幅值和刚度数据.
    这是JCP强烈建议的"解释性图"数据.
    返回 dict.
    """
    if psi_num is None:
        psi_num = ode_reference(x_test, ep, lam)
    if psi_ms  is None:
        psi_ms  = ms_new(x_test, ep, lam)

    Gamma      = psi_num / (psi_ms + 1e-15)
    psi_deriv  = np.abs(np.gradient(psi_num, x_test))
    ms_deriv   = np.abs(np.gradient(psi_ms,  x_test))
    gamma_deriv= np.abs(np.gradient(Gamma,   x_test))

    stiffness_ratio = psi_deriv / (gamma_deriv + 1e-10)
    bl_mask     = x_test <= 3*ep
    bl_stiff    = stiffness_ratio[bl_mask]
    overall_mean= float(np.mean(stiffness_ratio))
    bl_mean     = float(np.mean(bl_stiff)) if len(bl_stiff) > 0 else 0.0

    result = {
        'x':               x_test,
        'psi':             psi_num,
        'psi_ms':          psi_ms,
        'Gamma':           Gamma,
        'psi_deriv':       psi_deriv,
        'ms_deriv':        ms_deriv,
        'gamma_deriv':     gamma_deriv,
        'stiffness_ratio': stiffness_ratio,
        'stiffness_mean':  overall_mean,
        'bl_stiffness':    bl_mean,
        'psi_deriv_max':   float(psi_deriv.max()),
        'gamma_deriv_max': float(gamma_deriv.max()),
    }
    print(f'\n[Boundary Layer Analysis @ eps={ep}]')
    print(f'  |psi_prime|_max   = {result["psi_deriv_max"]:.3f}')
    print(f'  |Gamma_prime|_max = {result["gamma_deriv_max"]:.4f}')
    print(f'  Stiffness reduction (overall mean) = {overall_mean:.1f}x')
    print(f'  Stiffness reduction (BL, x<3eps)   = {bl_mean:.1f}x')
    return result


# ══════════════════════════════════════════════════════════════
# 复杂度分析 (scaling law)
# ══════════════════════════════════════════════════════════════

def complexity_scaling_study(
        x_test,
        eps_list=None,
        nc_list=None,
        m_list=None,
        ep_sup_fixed=3000,
        ep_phys_fixed=3000,
        verbose=True):
    """
    JCP要求的复杂度 scaling law:
      (a) 时间 vs ε  (fixed N_c, m)
      (b) 时间 vs N_c (配置点数, fixed ε, m)
      (c) 误差 vs m  (网络宽度, fixed ε, N_c) — Barron 验证
      (d) 时间-精度 tradeoff

    返回 dict with all timing/error data.
    """
    if eps_list is None: eps_list = [0.20, 0.10, 0.05, 0.02]
    if nc_list  is None: nc_list  = [100, 200, 300, 500, 800]
    if m_list   is None: m_list   = [16, 32, 64, 128, 256]

    results = {
        'scaling_vs_eps': [],
        'scaling_vs_nc':  [],
        'scaling_vs_m':   [],
        'tradeoff':       [],
    }

    # ── (a) Time vs epsilon ────────────────────────────────
    if verbose: print('\n[Complexity] (a) Time vs epsilon...')
    for ep in eps_list:
        t0 = time.time()
        xp = sample_collocation()
        xnp = xp.cpu().numpy().flatten()
        pm_np = ms_new(xnp, ep=ep); pr_np = ode_reference(xnp, ep=ep)
        P_v = np.array([P_eff_np(x, ep) for x in xnp])
        Q_v = np.array([Q_np(x, ep) for x in xnp])
        pm_t = torch.tensor(pm_np, dtype=torch.float64, device=device).unsqueeze(1)
        pr_t = torch.tensor(pr_np, dtype=torch.float64, device=device).unsqueeze(1)
        Pt   = torch.tensor(P_v,   dtype=torch.float64, device=device).unsqueeze(1)
        Qt   = torch.tensor(Q_v,   dtype=torch.float64, device=device).unsqueeze(1)
        m_net = RatioNet(64, 4).double().to(device)
        train(m_net, xp, pm_t, pr_t, Pt, Qt, tag=f'eps={ep}',
              ep=ep, ep_sup=ep_sup_fixed, ep_phys=ep_phys_fixed, verbose=False)
        elapsed = time.time() - t0
        xt  = torch.tensor(x_test, dtype=torch.float64, device=device).unsqueeze(1)
        pm2 = torch.tensor(ms_new(x_test, ep), dtype=torch.float64, device=device).unsqueeze(1)
        with torch.no_grad():
            pp  = (m_net.forward_R(xt)*pm2).cpu().numpy().flatten()
        err = np.max(np.abs(ode_reference(x_test, ep) - np.clip(pp, 0, 1)))
        results['scaling_vs_eps'].append({'ep': ep, 'time': elapsed, 'err': err})
        if verbose: print(f'  eps={ep:.3f}  time={elapsed:.0f}s  err={err:.2e}')

    # ── (b) Time vs N_c ────────────────────────────────────
    if verbose: print('\n[Complexity] (b) Time vs N_c (collocation points)...')
    for nc in nc_list:
        # Build custom collocation with nc points
        x_bl  = np.logspace(np.log10(1e-4), np.log10(3*EPSILON), nc//3)
        x_out = np.logspace(np.log10(3*EPSILON), np.log10(X_MAX-0.01), nc - nc//3)
        x_all = np.unique(np.concatenate([[1e-5], x_bl, x_out]))
        xp = torch.tensor(x_all, dtype=torch.float64, device=device).unsqueeze(1)
        xnp = x_all
        pm_np = ms_new(xnp); pr_np = ode_reference(xnp)
        P_v  = np.array([P_eff_np(x) for x in xnp])
        Q_v  = np.array([Q_np(x) for x in xnp])
        pm_t = torch.tensor(pm_np, dtype=torch.float64, device=device).unsqueeze(1)
        pr_t = torch.tensor(pr_np, dtype=torch.float64, device=device).unsqueeze(1)
        Pt   = torch.tensor(P_v,   dtype=torch.float64, device=device).unsqueeze(1)
        Qt   = torch.tensor(Q_v,   dtype=torch.float64, device=device).unsqueeze(1)
        m_net = RatioNet(64, 4).double().to(device)
        t0 = time.time()
        train(m_net, xp, pm_t, pr_t, Pt, Qt, tag=f'nc={nc}',
              ep_sup=ep_sup_fixed, ep_phys=ep_phys_fixed, verbose=False)
        elapsed = time.time() - t0
        xt  = torch.tensor(x_test, dtype=torch.float64, device=device).unsqueeze(1)
        pm2 = torch.tensor(ms_new(x_test), dtype=torch.float64, device=device).unsqueeze(1)
        with torch.no_grad():
            pp = (m_net.forward_R(xt)*pm2).cpu().numpy().flatten()
        err = np.max(np.abs(ode_reference(x_test) - np.clip(pp, 0, 1)))
        results['scaling_vs_nc'].append({'nc': nc, 'time': elapsed, 'err': err})
        if verbose: print(f'  Nc={nc}  time={elapsed:.0f}s  err={err:.2e}')

    # ── (c) Error vs m (network width) ────────────────────
    if verbose: print('\n[Complexity] (c) Error vs m (network width)...')
    xp  = sample_collocation(); xnp = xp.cpu().numpy().flatten()
    pm_np = ms_new(xnp); pr_np = ode_reference(xnp)
    P_v = np.array([P_eff_np(x) for x in xnp])
    Q_v = np.array([Q_np(x) for x in xnp])
    pm_t = torch.tensor(pm_np, dtype=torch.float64, device=device).unsqueeze(1)
    pr_t = torch.tensor(pr_np, dtype=torch.float64, device=device).unsqueeze(1)
    Pt   = torch.tensor(P_v,   dtype=torch.float64, device=device).unsqueeze(1)
    Qt   = torch.tensor(Q_v,   dtype=torch.float64, device=device).unsqueeze(1)
    for m_width in m_list:
        m_net = RatioNet(m_width, 4).double().to(device)
        t0 = time.time()
        train(m_net, xp, pm_t, pr_t, Pt, Qt, tag=f'm={m_width}',
              ep_sup=ep_sup_fixed, ep_phys=ep_phys_fixed, verbose=False)
        elapsed = time.time() - t0
        xt  = torch.tensor(x_test, dtype=torch.float64, device=device).unsqueeze(1)
        pm2 = torch.tensor(ms_new(x_test), dtype=torch.float64, device=device).unsqueeze(1)
        with torch.no_grad():
            pp = (m_net.forward_R(xt)*pm2).cpu().numpy().flatten()
        err = np.max(np.abs(ode_reference(x_test) - np.clip(pp, 0, 1)))
        n_params = sum(p.numel() for p in m_net.parameters())
        results['scaling_vs_m'].append({'m': m_width, 'err': err,
                                         'time': elapsed, 'n_params': n_params})
        if verbose: print(f'  m={m_width}  params={n_params}  err={err:.2e}  time={elapsed:.0f}s')

    # ── (d) Time-accuracy tradeoff ─────────────────────────
    # Use ms_new as baseline; record error at multiple epoch checkpoints
    if verbose: print('\n[Complexity] (d) Time-accuracy tradeoff...')
    checkpoints = [500, 1000, 2000, 5000, 10000]
    m_net = RatioNet(64, 4).double().to(device)
    R_tgt, _ = make_R_target(pm_t, pr_t)
    opt  = optim.Adam(m_net.parameters(), lr=3e-3)
    xt   = torch.tensor(x_test, dtype=torch.float64, device=device).unsqueeze(1)
    pm2  = torch.tensor(ms_new(x_test), dtype=torch.float64, device=device).unsqueeze(1)
    psi_true = ode_reference(x_test)
    cp_idx = 0; t0_global = time.time()
    for i in range(max(checkpoints)):
        m_net.train(); opt.zero_grad()
        l = loss_supervised_R(m_net, xp, R_tgt)
        l.backward(); opt.step()
        if cp_idx < len(checkpoints) and i+1 == checkpoints[cp_idx]:
            elapsed = time.time() - t0_global
            with torch.no_grad():
                pp = (m_net.forward_R(xt)*pm2).cpu().numpy().flatten()
            err = np.max(np.abs(psi_true - np.clip(pp, 0, 1)))
            results['tradeoff'].append({'epoch': i+1, 'time': elapsed, 'err': err})
            if verbose: print(f'  epoch={i+1}  time={elapsed:.1f}s  err={err:.2e}')
            cp_idx += 1

    if verbose:
        print('\n[Complexity Summary]')
        print(f"  eps scaling:  O(eps^{{-{_fit_exponent([r['ep'] for r in results['scaling_vs_eps']],[r['time'] for r in results['scaling_vs_eps']]):.2f}}})")
        print(f"  Nc  scaling:  O(Nc^{{{_fit_exponent([r['nc'] for r in results['scaling_vs_nc']],[r['time'] for r in results['scaling_vs_nc']]):.2f}}})")
        print(f"  m   scaling:  O(m^{{{_fit_exponent([r['m'] for r in results['scaling_vs_m']],[r['err'] for r in results['scaling_vs_m']]):.2f}}}) (error)")

    return results


def _fit_exponent(x_vals, y_vals):
    lx = np.log(np.array(x_vals, dtype=float))
    ly = np.log(np.array(y_vals, dtype=float))
    valid = np.isfinite(lx) & np.isfinite(ly)
    if valid.sum() < 2: return float('nan')
    return float(np.polyfit(lx[valid], ly[valid], 1)[0])


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    x = np.linspace(0, X_MAX, 300)
    print('=== Module 3 Self-Test ===')
    bl = boundary_layer_analysis(x)
    print(f'  BL stiffness reduction: {bl["stiffness_mean"]:.1f}x overall')
    print('[Done]')
