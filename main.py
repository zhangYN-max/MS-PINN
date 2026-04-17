"""
================================================================
主程序 / Main Entry Point
破产概率联合求解框架 v10 (JCP增强版)
================================================================
调用顺序:
  [1]  核心解的计算 (Module 1)
  [2]  训练 PINN & MS-PINN (Module 2)
  [3]  收敛性分析 + 截断定理 (Module 3)
  [4]  泛化验证 (β1-β7) (Module 3)
  [5]  λ 敏感性 (Module 3)
  [6]  ★ 经典求解器基准对比 (Module 2)
  [7]  ★ 消融实验 (Module 2)
  [8]  ★ 边界层解析比较 (Module 3)
  [9]  ★ 复杂度分析 (Module 3)
  [10] 生成全部图表 (Module 4)
================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# ── 导入各模块 ───────────────────────────────────────────────
from module1_core import (
    LAMBDA, EPSILON, X_MAX, EP_SUP, EP_PHYS,
    beta_np, betap_np, betapp_np, beta_t, betap_t,
    BETA_CASES,
    ode_reference, wkb_exact, ms_new, ms_old, ms_inner,
    P_eff_np, Q_np,
    verify_r_ode,
    verify_operator_coercivity,
    error_decomposition,
)
from module2_training import (
    device, sample_collocation, RatioNet, train,
    make_R_target,
    benchmark_classical_solvers,
    run_ablation_study,
)
from module3_studies import (
    convergence_study,
    optimal_truncation_study,
    generalization_study,
    lambda_sensitivity,
    boundary_layer_analysis,
    complexity_scaling_study,
)
from module4_figures import (
    save_all_figures,
    plot_benchmark_comparison,
    plot_ablation_study,
    plot_boundary_layer_detail,
    plot_complexity_scaling,
    plot_extended_generalization,
)


def main():
    torch.manual_seed(42); np.random.seed(42)

    print('=' * 70)
    print('  Ruin Probability: WKB + MS + PINN  v10  (JCP-Enhanced)')
    print('  Original IDE: eps*beta*R\' - R + (lam/eps)*int R(u)e^{-lam(x-u)/eps}du=0')
    print(f'  Parameters: beta=1.5+0.3sin(x), lam={LAMBDA}, eps={EPSILON}, x in [0,{X_MAX}]')
    print('=' * 70)

    x_test = np.linspace(0, X_MAX, 600)

    # ════════════════════════════════════════════════════════
    # [1] 核心解计算
    # ════════════════════════════════════════════════════════
    print('\n[1] Computing reference & perturbation solutions...')
    psi_wkb_sol   = wkb_exact(x_test)
    psi_num        = ode_reference(x_test)
    psi_ms_sol     = ms_new(x_test)
    psi_ms_old_    = ms_old(x_test)
    psi_ms_inn_    = ms_inner(x_test)
    R_true         = psi_num / (psi_ms_sol + 1e-15)

    print(f'  psi(0)={psi_num[0]:.4f}  psi(0.1)={psi_num[10]:.4f}'
          f'  psi(0.5)={psi_num[50]:.4f}')
    print(f'  New MS max err: {np.max(np.abs(psi_num-psi_ms_sol)):.2e}')
    print(f'  Old MS max err: {np.max(np.abs(psi_num-psi_ms_old_)):.2e}')
    print(f'  Gamma range: [{R_true.min():.4f}, {R_true.max():.4f}]')
    verify_r_ode()

    # ★ v10: Operator well-posedness
    verify_operator_coercivity(x_test)
    # ★ v10: Error decomposition
    error_decomposition(x_test)

    # ════════════════════════════════════════════════════════
    # [2] 训练 PINN & MS-PINN
    # ════════════════════════════════════════════════════════
    print('\n[2] Preparing collocation points and training data...')
    x_pde = sample_collocation()
    xnp   = x_pde.cpu().numpy().flatten()
    pm_np = ms_new(xnp);  pr_np = ode_reference(xnp)
    P_v   = np.array([P_eff_np(x) for x in xnp])
    Q_v   = np.array([Q_np(x)     for x in xnp])

    psi_ms_t  = torch.tensor(pm_np, dtype=torch.float64, device=device).unsqueeze(1)
    psi_ref_t = torch.tensor(pr_np, dtype=torch.float64, device=device).unsqueeze(1)
    P_t       = torch.tensor(P_v,   dtype=torch.float64, device=device).unsqueeze(1)
    Q_t       = torch.tensor(Q_v,   dtype=torch.float64, device=device).unsqueeze(1)

    print(f'  Collocation: {len(xnp)} pts'
          f'  P_eff in [{P_v.min():.3f},{P_v.max():.3f}]'
          f'  Q in [{Q_v.min():.3e},{Q_v.max():.3e}]')

    print('\n[2a] Training PINN...')
    model_pinn = RatioNet(64, 5).double().to(device)
    hist_pinn  = train(model_pinn, x_pde, psi_ms_t, psi_ref_t, P_t, Q_t, tag='PINN')

    print('\n[2b] Training MS-PINN...')
    model_ms   = RatioNet(64, 5).double().to(device)
    hist_ms    = train(model_ms,   x_pde, psi_ms_t, psi_ref_t, P_t, Q_t, tag='MS-PINN')

    # ── Evaluate ────────────────────────────────────────────
    print('\n[3] Evaluation...')
    x_t   = torch.tensor(x_test,    dtype=torch.float64, device=device).unsqueeze(1)
    pm_t2 = torch.tensor(psi_ms_sol,dtype=torch.float64, device=device).unsqueeze(1)
    with torch.no_grad():
        psi_pinn_  = (model_pinn.forward_R(x_t)*pm_t2).cpu().numpy().flatten()
        psi_mspinn_= (model_ms.forward_R(x_t)  *pm_t2).cpu().numpy().flatten()
    psi_pinn_   = np.clip(psi_pinn_,   0, 1)
    psi_mspinn_ = np.clip(psi_mspinn_, 0, 1)

    ref_n   = np.linalg.norm(psi_num)
    e_wkb   = np.abs(psi_num - psi_wkb_sol)
    e_ms    = np.abs(psi_num - psi_ms_sol)
    e_old   = np.abs(psi_num - psi_ms_old_)
    e_inner = np.abs(psi_num - psi_ms_inn_)
    e_pinn  = np.abs(psi_num - psi_pinn_)
    e_msp   = np.abs(psi_num - psi_mspinn_)

    print('\n' + '='*66)
    print(f'{"Method":<22} {"Max Error":>12} {"Rel-L2":>12} {"vs New MS":>10}')
    print('-'*66)
    for nm, err in [('WKB Exact', e_wkb), ('Old MS (p=0.83)', e_old),
                    ('New MS (p=1.90)', e_ms), ('MS Inner', e_inner),
                    ('PINN v10', e_pinn), ('MS-PINN v10', e_msp)]:
        vs = (f'x{np.max(e_ms)/np.max(err):.0f}' if np.max(err) < np.max(e_ms)
              else f'/{np.max(err)/np.max(e_ms):.1f}x')
        print(f'{nm:<22} {np.max(err):>12.2e} '
              f'{np.linalg.norm(err)/ref_n:>12.2e} {vs:>10}')
    print('='*66)

    # ════════════════════════════════════════════════════════
    # [4] 收敛阶 + 截断定理
    # ════════════════════════════════════════════════════════
    eps_list = [0.20, 0.10, 0.05, 0.02, 0.01]
    errs_ms, errs_pp, p_ms, p_pp, local_p = convergence_study(x_test, eps_list)

    print('\n[5] Optimal truncation theorem verification...')
    E0_list, E1_list, ratio_list, cond_list = optimal_truncation_study(x_test, eps_list)

    # ════════════════════════════════════════════════════════
    # [5] 泛化性验证（所有7种β类型）
    # ════════════════════════════════════════════════════════
    print('\n[6] Generalization study (7 beta functions, v10)...')
    gen_results = generalization_study(
        x_test, ep=EPSILON, lam=LAMBDA,
        eps_scan=[0.20, 0.10, 0.05, 0.02, 0.01],
        run_pinn=True, ep_sup=5000, ep_phys=5000,
        beta_keys=list(BETA_CASES.keys()))

    # ════════════════════════════════════════════════════════
    # [6] λ敏感性
    # ════════════════════════════════════════════════════════
    print('\n[7] Lambda sensitivity analysis...')
    lam_results = lambda_sensitivity(x_test, [1.05, 1.2, 1.5, 2.0, 5.0], ep=EPSILON)

    # ════════════════════════════════════════════════════════
    # ★ [7] 经典求解器基准对比 (JCP必备)
    # ════════════════════════════════════════════════════════
    print('\n[8] Classical solver benchmark (JCP Section 5.X)...')
    bench_results = benchmark_classical_solvers(x_test)

    # ════════════════════════════════════════════════════════
    # ★ [8] 消融实验
    # ════════════════════════════════════════════════════════
    print('\n[9] Ablation study...')
    ablation_results = run_ablation_study(
        x_test, ep=EPSILON, lam=LAMBDA,
        ep_sup=5000, ep_phys=5000)

    # ════════════════════════════════════════════════════════
    # ★ [9] 边界层解析
    # ════════════════════════════════════════════════════════
    print('\n[10] Boundary layer detailed analysis...')
    bl_data = boundary_layer_analysis(x_test, psi_num, psi_ms_sol)

    # ════════════════════════════════════════════════════════
    # ★ [10] 复杂度分析
    # ════════════════════════════════════════════════════════
    print('\n[11] Complexity scaling study...')
    scaling_data = complexity_scaling_study(
        x_test,
        eps_list=[0.20, 0.10, 0.05, 0.02],
        nc_list=[100, 200, 300, 500],
        m_list=[16, 32, 64, 128],
        ep_sup_fixed=3000, ep_phys_fixed=3000)

    # ════════════════════════════════════════════════════════
    # [11] 生成全部图表
    # ════════════════════════════════════════════════════════
    print('\n[12] Saving all figures...')

    save_all_figures(
        x_test, psi_wkb_sol, psi_num, psi_ms_sol,
        psi_ms_old_, psi_ms_inn_, psi_pinn_, psi_mspinn_,
        e_wkb, e_ms, e_old, e_inner, e_pinn, e_msp,
        hist_pinn, hist_ms,
        eps_list, errs_ms, errs_pp, p_ms, p_pp, local_p,
        E0_list=E0_list, E1_list=E1_list,
        ratio_list=ratio_list, cond_list=cond_list,
        gen_results=gen_results,
        lam_results=lam_results,
    )

    # ★ v10 新增图
    plot_benchmark_comparison(x_test, bench_results)
    plot_ablation_study(x_test, ablation_results)
    plot_boundary_layer_detail(bl_data)
    plot_complexity_scaling(scaling_data)
    plot_extended_generalization(x_test, gen_results)

    # ════════════════════════════════════════════════════════
    # 保存模型
    # ════════════════════════════════════════════════════════
    torch.save(model_pinn.state_dict(), 'pinn_model_v10.pth')
    torch.save(model_ms.state_dict(),   'mspinn_model_v10.pth')
    print('Models saved: pinn_model_v10.pth, mspinn_model_v10.pth')
    print('\n[Done] — v10 JCP-enhanced complete.')


if __name__ == '__main__':
    main()
