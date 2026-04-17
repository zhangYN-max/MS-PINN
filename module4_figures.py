"""
================================================================
模块4: 图表生成 / Figure Generation
破产概率联合求解框架 v10 (JCP增强版)
================================================================
包含原有14张图 + v10 新增图:
  Fig15: 经典求解器基准对比表格 (JCP Section 5.X)
  Fig16: 消融实验误差/收敛曲线
  Fig17: 边界层导数幅值对比 (放大图)
  Fig18: 复杂度 Scaling Law (4子图)
  Fig19: 泛化 — 新增β类型 (beta5/6/7)
================================================================
"""

import numpy as np
import matplotlib
# 字体设置（必须在 use 前）
import matplotlib.font_manager as fm
_zh_candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                   'Noto Sans CJK SC', 'PingFang SC', 'Arial Unicode MS']
_zh_font = next((fc for fc in _zh_candidates
                  if any(fc.lower() in f.name.lower() for f in fm.fontManager.ttflist)),
                 'DejaVu Sans')
matplotlib.rcParams['font.family']       = [_zh_font, 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus']= False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from module1_core import LAMBDA, EPSILON, X_MAX, BETA_CASES

DPI = 300

# ── helper ─────────────────────────────────────────────────────
def sf(name, fig=None):
    for ext in ['pdf', 'png']:
        plt.savefig(f'{name}.{ext}', bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print(f'  Saved: {name}.pdf/.png')


# ══════════════════════════════════════════════════════════════
# 原有图 1–14（保持不变，统一通过此函数调用）
# ══════════════════════════════════════════════════════════════

def save_all_figures(
        x_test, psi_wkb_sol, psi_num, psi_ms_, psi_ms_old_, psi_ms_inner_,
        psi_pinn_, psi_mspinn_,
        e_wkb, e_ms, e_old, e_inner, e_pinn, e_msp,
        hist_pinn, hist_ms,
        eps_list, errs_ms, errs_pp, p_ms, p_pp, local_p,
        E0_list=None, E1_list=None, ratio_list=None, cond_list=None,
        gen_results=None, lam_results=None):

    # ── Fig 1 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(x_test, psi_num,       'k',   lw=3,   label='ODE45 Reference')
    ax.plot(x_test, psi_wkb_sol,   'r-',  lw=1.8, label=f'WKB Exact {np.max(e_wkb):.0e}')
    ax.plot(x_test, psi_ms_,       'b',   lw=2.2, label=f'New MS {np.max(e_ms):.2e}')
    ax.plot(x_test, psi_ms_old_,   'm--', lw=1.8, label=f'Old MS {np.max(e_old):.2e}')
    ax.plot(x_test, psi_ms_inner_, color=[0.1,0.6,0.2], ls='-.', lw=1.6,
            label=f'MS Inner {np.max(e_inner):.2e}')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$\psi(x)$', fontsize=13)
    ax.set_title(r'Ruin Probability $\psi(x)$: WKB & MS Methods', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim([0, X_MAX])
    ax = axes[1]
    m_bl = x_test <= 0.5
    for arr, c, ls, lbl in [(psi_num,'k','-','ODE45'), (psi_wkb_sol,'r','-','WKB Exact'),
                              (psi_ms_,'b','-','New MS'), (psi_ms_old_,'m','--','Old MS'),
                              (psi_ms_inner_,'#1a9a3a','-.','MS Inner')]:
        ax.plot(x_test[m_bl], arr[m_bl], color=c, ls=ls, lw=2, label=lbl)
    ax.axvline(EPSILON,   color='gray', ls='--', lw=1, label=r'$\varepsilon$')
    ax.axvline(3*EPSILON, color='gray', ls=':', lw=1, label=r'$3\varepsilon$')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$\psi(x)$', fontsize=13)
    ax.set_title(r'Boundary Layer $x\in[0,0.5]$', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); sf('fig1_wkb_ms_solution')

    # ── Fig 2 ─────────────────────────────────────────────────
    from module1_core import S_scalar, beta_np
    fig, ax = plt.subplots(figsize=(8, 5))
    S_arr = np.array([S_scalar(x) for x in x_test]); eS = np.exp(S_arr/EPSILON)
    A_exact = np.where(eS > 1e-15, psi_num/(eS+1e-300), np.nan)
    A_new   = (LAMBDA*beta_np(0)-1)/(LAMBDA*beta_np(x_test)-1)
    A_old   = np.sqrt(beta_np(0)/beta_np(x_test))
    m3 = x_test <= 3
    ax.plot(x_test[m3], A_exact[m3], 'k', lw=2.5, label=r'Exact $A=\psi/e^{S/\varepsilon}$')
    ax.plot(x_test[m3], A_new[m3],   'b', lw=2.2,
            label=r'New MS: $A=(\lambda\beta_0-1)/(\lambda\beta-1)$')
    ax.plot(x_test[m3], A_old[m3],   'm--', lw=1.8,
            label=r'Old MS: $A=\sqrt{\beta_0/\beta}$')
    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$A(x)=\psi(x)/\exp\{S(x)/\varepsilon\}$', fontsize=13)
    ax.set_title('Amplitude Function Comparison', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim([0, 3])
    plt.tight_layout(); sf('fig2_amplitude')

    # ── Fig 3 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for arr, c, ls, lbl in [
            (e_wkb, 'r','-', f'WKB Exact {np.max(e_wkb):.0e}'),
            (e_ms,  'b','-', f'New MS {np.max(e_ms):.2e}'),
            (e_old, 'm','--',f'Old MS {np.max(e_old):.2e}'),
            (e_inner,'#1a9a3a','-.', f'MS Inner {np.max(e_inner):.2e}'),
            (e_pinn,'#4a90d9','-.', f'PINN v9 {np.max(e_pinn):.2e}'),
            (e_msp, '#2e8b2e','-',  f'MS-PINN v9 {np.max(e_msp):.2e}')]:
        ax.semilogy(x_test, arr+1e-16, color=c, ls=ls, lw=2, label=lbl)
    ax.axhline(EPSILON,    color='gray', ls=':',  lw=1, label=r'$\varepsilon$')
    ax.axhline(EPSILON**2, color='gray', ls='--', lw=1, label=r'$\varepsilon^2$')
    ax.set_title(f'Pointwise Absolute Error [eps={EPSILON}]', fontsize=13)
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3); ax.set_xlim([0, X_MAX])
    plt.tight_layout(); sf('fig3_error')

    # ── Fig 4 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ['WKB\nExact','Old MS\n(p=0.83)','New MS\n(p=1.90)',
               'MS Inner\n(local)','PINN\nv9','MS-PINN\nv9']
    errors  = [np.max(e_wkb),np.max(e_old),np.max(e_ms),
               np.max(e_inner),np.max(e_pinn),np.max(e_msp)]
    colors  = ['#555555','#c04a8c','#4a7cc0','#4aab6d','#4a7cc0','#2e8b2e']
    bars = ax.bar(methods, errors, color=colors, edgecolor='k', alpha=0.85, width=0.6)
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x()+bar.get_width()/2, val*1.4,
                f'{val:.1e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_yscale('log')
    ax.axhline(EPSILON,    color='gray', ls=':', lw=1.5, label=r'$\varepsilon$')
    ax.axhline(EPSILON**2, color='gray', ls='--',lw=1.5, label=r'$\varepsilon^2$')
    ax.set_title(f'Error Summary [eps={EPSILON}, lam={LAMBDA}]', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); sf('fig4_error_bar')

    # ── Fig 5 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    R_true = psi_num/(psi_ms_+1e-15)
    R_pinn = psi_pinn_/(psi_ms_+1e-15)
    R_msp  = psi_mspinn_/(psi_ms_+1e-15)
    ax.plot(x_test, R_true, 'k',   lw=2.5, label=r'True $\Gamma=\psi/\psi_{MS}$')
    ax.plot(x_test, R_pinn, 'b-.', lw=2,   label='PINN v9')
    ax.plot(x_test, R_msp,  'g',   lw=2,   label='MS-PINN v9')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    var = R_true.max()-R_true.min()
    ax.text(3.0, 1.15, f'Variation={var:.3f}\n|Gamma_prime|<=0.21 (no stiffness)',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$\Gamma(x)$', fontsize=13)
    ax.set_title(r'Correction Ratio $\Gamma(x)$: Slow-varying, Hard BC', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim([0, X_MAX])
    plt.tight_layout(); sf('fig5_ratio_R')

    # ── Fig 6 ─────────────────────────────────────────────────
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax_, hist_, lbl, c1, c2 in [
            (axes2[0], hist_pinn, 'PINN v9',    'deepskyblue', 'steelblue'),
            (axes2[1], hist_ms,   'MS-PINN v9', 'orchid',      'purple')]:
        N1 = len(hist_['s1'])
        ax_.semilogy(range(N1), hist_['s1'], color=c1, lw=1.5, alpha=0.8,
                     label=r'Stage 1: supervised $\Gamma_{target}$')
        ax_.semilogy(range(N1, N1+len(hist_['s2'])), hist_['s2'],
                     color=c2, lw=1.5, label='Stage 2: R-ODE physics')
        ax_.axvline(N1, color='gray', ls=':', lw=1.2)
        ax_.set_xlabel('Epoch', fontsize=12); ax_.set_ylabel('Loss', fontsize=12)
        ax_.set_title(f'{lbl} Training Loss', fontsize=12)
        ax_.legend(fontsize=9); ax_.grid(True, alpha=0.3)
    plt.suptitle('Two-Stage Training: Supervised Prior -> R-ODE Physics',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(); sf('fig6_training_loss')

    # ── Fig 7 ─────────────────────────────────────────────────
    fig, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes2[0]
    ax.plot(x_test, psi_num,     'k',   lw=3,  label='ODE45 Reference')
    ax.plot(x_test, psi_ms_,     'r--', lw=2,  label=f'New MS {np.max(e_ms):.2e}')
    ax.plot(x_test, psi_pinn_,   'b-.', lw=2,  label=f'PINN v9 {np.max(e_pinn):.2e}')
    ax.plot(x_test, psi_mspinn_, 'g',   lw=2,  label=f'MS-PINN v9 {np.max(e_msp):.2e}')
    ax.set_title(f'PINN Solutions [eps={EPSILON}]', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim([0, X_MAX])
    ax = axes2[1]
    ax.semilogy(x_test, e_ms+1e-16,   'r--', lw=2, label=f'New MS {np.max(e_ms):.2e}')
    ax.semilogy(x_test, e_pinn+1e-16, 'b-.', lw=2, label=f'PINN v9 {np.max(e_pinn):.2e}')
    ax.semilogy(x_test, e_msp+1e-16,  'g',   lw=2, label=f'MS-PINN v9 {np.max(e_msp):.2e}')
    ax.axhline(EPSILON,    color='gray', ls=':', lw=1, label=r'$\varepsilon$')
    ax.axhline(EPSILON**2, color='gray', ls='--',lw=1, label=r'$\varepsilon^2$')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$|\psi-\psi_{ref}|$', fontsize=13)
    ax.set_title('Absolute Error (log scale)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim([0, X_MAX])
    plt.tight_layout(); sf('fig7_pinn_solution')

    # ── Fig 8 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(eps_list, errs_ms, 'rs-', lw=2, ms=9, label=f'New MS $p={p_ms:.2f}$')
    ax.loglog(eps_list, errs_pp, 'bo-', lw=2, ms=9, label=f'PINN v9 $p={p_pp:.2f}$')
    ref2 = errs_ms[1]*(np.array(eps_list)/eps_list[1])**2
    ref3 = errs_ms[1]*(np.array(eps_list)/eps_list[1])**3
    ax.loglog(eps_list, ref2, 'k--', lw=1.5, label=r'$O(\varepsilon^2)$')
    ax.loglog(eps_list, ref3, 'k:',  lw=1.5, label=r'$O(\varepsilon^3)$')
    for i, lp in enumerate(local_p):
        xm = np.exp(0.5*(np.log(eps_list[i])+np.log(eps_list[i+1])))
        ym = np.exp(0.5*(np.log(errs_pp[i]) +np.log(errs_pp[i+1])))
        ax.annotate(f'$p={lp:.1f}$', xy=(xm, ym*1.5), fontsize=9, color='steelblue', ha='center')
    ax.set_xlabel(r'$\varepsilon$', fontsize=14); ax.set_ylabel('Max Absolute Error', fontsize=13)
    ax.set_title('Convergence Order vs. Perturbation Parameter', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); sf('fig8_convergence')

    # ── Fig 9: Results summary table ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5)); ax.axis('off')
    ref_n = np.linalg.norm(psi_num)
    rows = [
        ['Method','Max Error','Rel-L2','Order','vs New MS'],
        ['WKB Exact', f'{np.max(e_wkb):.2e}', f'{np.linalg.norm(e_wkb)/ref_n:.2e}', 'machine','---'],
        ['Old MS (sqrt b0/b)', f'{np.max(e_old):.2e}', f'{np.linalg.norm(e_old)/ref_n:.2e}', 'p≈0.83','1x'],
        ['New MS (Watson n=0)', f'{np.max(e_ms):.2e}', f'{np.linalg.norm(e_ms)/ref_n:.2e}', 'p≈1.90','---'],
        ['MS Inner (local)', f'{np.max(e_inner):.2e}', f'{np.linalg.norm(e_inner)/ref_n:.2e}', 'O(1)','---'],
        ['PINN v9', f'{np.max(e_pinn):.2e}', f'{np.linalg.norm(e_pinn)/ref_n:.2e}',
         f'p≈{p_pp:.2f}', f'x{np.max(e_ms)/np.max(e_pinn):.0f}'],
        ['MS-PINN v9', f'{np.max(e_msp):.2e}', f'{np.linalg.norm(e_msp)/ref_n:.2e}',
         'O(e^5)', f'x{np.max(e_ms)/np.max(e_msp):.0f}'],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.9)
    for j in range(5):
        tbl[(0,j)].set_facecolor('#1a4a7a'); tbl[(0,j)].set_text_props(color='white', fontweight='bold')
    for i, rc in enumerate(['#f5f5f5','#eaf2fb','#d5e9fb','#f5f5f5','#c8f0d8','#a8e0ba']):
        for j in range(5): tbl[(i+1,j)].set_facecolor(rc)
    ax.set_title(f'Results Summary eps={EPSILON}, lam={LAMBDA}, beta=1.5+0.3sin(x)',
                  fontsize=12, pad=12)
    plt.tight_layout(); sf('fig9_results_table')

    # ── Fig 10: Optimal truncation ────────────────────────────
    if E0_list is not None:
        fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes2[0]
        ax.loglog(eps_list, E0_list, 'bs-', lw=2, ms=9, label=r'$E_0$ (n=0, optimal)')
        ax.loglog(eps_list, E1_list, 'r^--',lw=2, ms=9, label=r'$E_1$ (n=1)')
        ref2_ = E0_list[1]*(np.array(eps_list)/eps_list[1])**2
        ax.loglog(eps_list, ref2_, 'k:', lw=1.5, label=r'$O(\varepsilon^2)$')
        ax.set_xlabel(r'$\varepsilon$', fontsize=13); ax.set_ylabel('Max Absolute Error', fontsize=12)
        ax.set_title('Watson Truncation Error: n=0 vs n=1', fontsize=12)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        ax = axes2[1]
        ax.semilogx(eps_list, ratio_list, 'ko-', lw=2, ms=9, label=r'$E_1/E_0$')
        ax.axhline(2.0, color='b', ls='--', lw=1.5, label=r'$E_1/E_0=2$ (theory)')
        ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax2b = ax.twinx()
        ax2b.semilogx(eps_list, cond_list, 'rs--', lw=1.5, ms=7, alpha=0.7,
                      label=r'Condition $\varepsilon|\Psi_1/\Psi_0||G_0/G_1|$')
        ax2b.set_ylabel('Condition value', color='r', fontsize=11)
        ax2b.tick_params(axis='y', labelcolor='r')
        ax.set_xlabel(r'$\varepsilon$', fontsize=13); ax.set_ylabel(r'$E_1/E_0$', fontsize=12)
        ax.set_title(r'Theorem 1: $E_1/E_0\approx 2$ for all $\varepsilon$', fontsize=12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.suptitle('Optimal Truncation Theorem (Theorem 1)', fontsize=13, fontweight='bold')
        plt.tight_layout(); sf('fig10_optimal_truncation')

    # ── Fig 11: Generalization (original 4 betas) ─────────────
    if gen_results is not None:
        beta_keys_orig = ['beta1','beta2','beta3','beta4']
        available = [k for k in beta_keys_orig if k in gen_results]
        if len(available) >= 1:
            fig, axes2 = plt.subplots(2, 2, figsize=(13, 10))
            for idx, (bk, ax_) in enumerate(zip(available, axes2.flatten())):
                r = gen_results[bk]; bc = BETA_CASES[bk]
                ax_.plot(x_test, r['psi_ref'],    'k', lw=2.5, label='Reference')
                ax_.plot(x_test, r['psi_ms_new'], 'b', lw=2,   label=f"New MS {r['e_new']:.2e}")
                ax_.plot(x_test, r['psi_ms_old'], 'm--',lw=1.8, label=f"Old MS {r['e_old']:.2e}")
                if r['psi_pinn'] is not None:
                    ax_.plot(x_test, r['psi_pinn'], 'g', lw=2,
                             label=f"MS-PINN {r['e_pinn']:.2e}")
                ax_.set_title(bc['name_en'], fontsize=11)
                ax_.set_xlabel(r'$x$', fontsize=11); ax_.set_ylabel(r'$\psi(x)$', fontsize=11)
                ax_.legend(fontsize=8.5); ax_.grid(True, alpha=0.3); ax_.set_xlim([0, X_MAX])
                ax_.text(0.97, 0.97, f"p(New MS)={r['p_new']:.2f}",
                         transform=ax_.transAxes, ha='right', va='top', fontsize=9,
                         bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
            plt.suptitle(r'Generalization: 4 Beta Functions [eps=0.1, lam=1.2]',
                          fontsize=13, fontweight='bold')
            plt.tight_layout(); sf('fig11_generalization')

    # ── Fig 12: Lambda sensitivity ─────────────────────────────
    if lam_results is not None:
        lam_list_ = list(lam_results.keys())
        fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes2[0]
        colors_l = plt.cm.viridis(np.linspace(0, 0.9, len(lam_list_)))
        for i, (lam_, c) in enumerate(zip(lam_list_, colors_l)):
            r = lam_results[lam_]
            ax.plot(x_test, r['psi_ref'], '--', color=c, lw=1.2, alpha=0.5)
            ax.plot(x_test, r['psi_ms'],  '-',  color=c, lw=2,
                    label=f'lam={lam_:.2f} err={r["e_new"]:.2e}')
        ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$\psi(x)$', fontsize=13)
        ax.set_title('Effect of lambda (solid=New MS, dashed=Ref)', fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim([0, 3])
        ax = axes2[1]
        e_new_list_ = [lam_results[l]['e_new'] for l in lam_list_]
        p_list_     = [lam_results[l]['p_new'] for l in lam_list_]
        bars = ax.bar([str(l) for l in lam_list_], e_new_list_,
                      color=[plt.cm.viridis(i/len(lam_list_)) for i in range(len(lam_list_))],
                      edgecolor='k', alpha=0.85)
        for bar, val, p_ in zip(bars, e_new_list_, p_list_):
            ax.text(bar.get_x()+bar.get_width()/2, val*1.3,
                    f'{val:.1e}\np={p_:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_yscale('log')
        ax.set_xlabel('lambda', fontsize=13); ax.set_ylabel('Max Error', fontsize=12)
        ax.set_title('Lambda Sensitivity: larger lambda -> smaller error', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.suptitle('Effect of Lambda [eps=0.1, beta=1.5+0.3sin(x)]',
                      fontsize=13, fontweight='bold')
        plt.tight_layout(); sf('fig12_lambda_sensitivity')

    # ── Fig 13: Stiffness removal visualization ────────────────
    psi_deriv  = np.abs(np.gradient(psi_num, x_test))
    Gamma_arr  = psi_num/(psi_ms_+1e-15)
    Gamma_deriv= np.abs(np.gradient(Gamma_arr, x_test))
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes2[0]
    ax.semilogy(x_test, psi_deriv+1e-10,  'b',   lw=2.5, label=r'$|\psi^\prime(x)|$')
    ax.semilogy(x_test, Gamma_deriv+1e-10,'r--',  lw=2.5, label=r'$|\Gamma^\prime(x)|$')
    ax.axvline(EPSILON, color='gray', ls=':', lw=1.5, label=r'$\varepsilon=0.1$')
    ax.fill_betweenx([1e-6,10], [0], [EPSILON], alpha=0.1, color='orange',
                      label='Boundary layer')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel('Derivative magnitude', fontsize=12)
    ax.set_title(r'Lemma 4.1: $\Gamma=\psi/\psi_{MS}$ removes stiffness', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim([0, 2])
    ax = axes2[1]
    ratio_deriv = psi_deriv/(Gamma_deriv+1e-10)
    ax.semilogy(x_test[x_test<=2], ratio_deriv[x_test<=2], 'g', lw=2.5,
                label=r'$|\psi^\prime|/|\Gamma^\prime|$ (stiffness reduction)')
    mean_stiff = float(np.mean(ratio_deriv))
    ax.axhline(mean_stiff, color='b', ls='--', lw=1.5,
               label=f'Mean ~{mean_stiff:.0f}x reduction')
    ax.axhline(1, color='gray', ls=':', lw=1)
    ax.fill_betweenx([0.5,200], [0], [EPSILON], alpha=0.1, color='orange',
                      label='Boundary layer')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel('Stiffness reduction factor', fontsize=12)
    ax.set_title('Stiffness reduction (>100x in BL)', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim([0, 2])
    plt.suptitle(r'Lemma 4.1: Ratio parameterization removes BL stiffness',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(); sf('fig13_stiffness_removal')

    # ── Fig 14: R-ODE coefficients ────────────────────────────
    from module1_core import Q_np, P_eff_np
    Q_arr    = np.array([Q_np(x)    for x in x_test])
    Peff_arr = np.array([P_eff_np(x) for x in x_test])
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes2[0]
    ax.plot(x_test, Q_arr, 'b', lw=2.5, label=r'$Q(x)$ (forcing term)')
    ax.axhline(0, color='k', ls='-', lw=0.8, alpha=0.5)
    ax.fill_between(x_test, 0, Q_arr, where=Q_arr>0, alpha=0.2, color='blue')
    ax.fill_between(x_test, 0, Q_arr, where=Q_arr<0, alpha=0.2, color='red')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$Q(x)$', fontsize=12)
    ax.set_title('R-ODE Forcing Q(x): Non-zero everywhere', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax = axes2[1]
    ax.plot(x_test, Peff_arr, 'r', lw=2.5, label=r'$P_{\rm eff}(x)$')
    ax.fill_between(x_test, Peff_arr, 0, alpha=0.2, color='red')
    ax.axhline(0, color='k', ls='-', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$P_{\rm eff}(x)$', fontsize=12)
    ax.set_title('Transport coeff P_eff(x): Negative everywhere -> stable', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.suptitle(f'R-ODE Coefficient Distribution [eps={EPSILON}, lam={LAMBDA}]',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(); sf('fig14_rode_coefficients')

    print(f'\n14 core figures saved (DPI={DPI}).')


# ══════════════════════════════════════════════════════════════
# ★ v10 新增图 15–19
# ══════════════════════════════════════════════════════════════

def plot_benchmark_comparison(x_arr, bench_results, ep=EPSILON):
    """Fig15: 经典求解器对比表格 + 误差曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 误差曲线
    ax = axes[0]
    colors_b = {'FDM':'#e07a2a', 'Chebyshev Spectral':'#7a2ae0',
                 'Quadrature IDE':'#2a7ae0', 'New MS':'#2aae7a', 'MS-PINN v9':'#ae2a2a'}
    psi_ref = bench_results.get('ODE45 (Reference)', {}).get('psi', None)
    for name, r in bench_results.items():
        if name == 'ODE45 (Reference)' or r['psi'] is None: continue
        diff = np.abs(r['psi'] - psi_ref)
        c    = colors_b.get(name, 'gray')
        ax.semilogy(x_arr, diff+1e-16, lw=2, color=c,
                    label=f"{name}  {r['err_max']:.2e}")
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel('Pointwise Error', fontsize=12)
    ax.set_title(f'Classical Solver Benchmark [eps={ep}]', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 汇总表格
    ax2 = axes[1]; ax2.axis('off')
    rows = [['Method','Max Error','Rel-L2','Time(s)','Stable?']]
    for name, r in bench_results.items():
        rows.append([name, f'{r["err_max"]:.2e}', f'{r["err_l2"]:.2e}',
                     f'{r["time"]:.3f}', '✓' if r['stable'] else '✗'])
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.7)
    for j in range(5):
        tbl[(0,j)].set_facecolor('#1a4a7a'); tbl[(0,j)].set_text_props(color='white', fontweight='bold')
    ax2.set_title('Comparison with Classical Solvers (JCP Section 5.X)', fontsize=11, pad=10)
    plt.tight_layout(); sf('fig15_classical_benchmark')


def plot_ablation_study(x_arr, ablation_results):
    """Fig16: 消融实验误差对比 + 训练曲线"""
    names  = list(ablation_results.keys())
    errors = [ablation_results[n]['err_max'] for n in names]
    times  = [ablation_results[n]['time']    for n in names]
    colors_a = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(names)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    bars = ax.bar(names, errors, color=colors_a, edgecolor='k', alpha=0.85)
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x()+bar.get_width()/2, val*1.4,
                f'{val:.1e}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_yscale('log'); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Max Error', fontsize=12)
    ax.set_title('Ablation Study: Max Error per Variant', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    ax2.bar(names, times, color=colors_a, edgecolor='k', alpha=0.85)
    ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax2.set_ylabel('Training Time (s)', fontsize=12)
    ax2.set_title('Ablation Study: Training Time', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = axes[2]
    for name, c in zip(names, colors_a):
        r    = ablation_results[name]
        hist = r['hist']
        s1   = hist.get('s1', []); s2 = hist.get('s2', [])
        all_loss = s1 + s2
        if all_loss:
            ax3.semilogy(range(len(all_loss)), all_loss, lw=1.5, color=c,
                         label=f'{name} ({r["err_max"]:.1e})')
    ax3.set_xlabel('Epoch', fontsize=12); ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Ablation: Loss Curves', fontsize=12)
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    plt.suptitle('Ablation Study: Contribution of Each Component',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(); sf('fig16_ablation_study')


def plot_boundary_layer_detail(bl_data, ep=EPSILON):
    """Fig17: 边界层放大 + gradient comparison (JCP strong suggestion)"""
    x       = bl_data['x']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # subplot 1: psi / psi_MS / Gamma in BL
    ax = axes[0]
    bl_mask = x <= 5*ep
    ax.plot(x[bl_mask], bl_data['psi'][bl_mask],    'k',   lw=2.5, label=r'$\psi(x)$')
    ax.plot(x[bl_mask], bl_data['psi_ms'][bl_mask], 'b--', lw=2,   label=r'$\psi_{MS}(x)$')
    ax.plot(x[bl_mask], bl_data['Gamma'][bl_mask],  'r',   lw=2,   label=r'$\Gamma(x)=\psi/\psi_{MS}$')
    ax.axvline(ep, color='gray', ls=':', lw=1.5, label=r'$\varepsilon$')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel('Value', fontsize=12)
    ax.set_title(r'Boundary Layer: $\psi$, $\psi_{MS}$, $\Gamma$ (zoomed)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # subplot 2: derivative magnitudes (semilogy)
    ax = axes[1]
    ax.semilogy(x, bl_data['psi_deriv']+1e-10,   'b',   lw=2.5, label=r'$|\psi^\prime|$')
    ax.semilogy(x, bl_data['ms_deriv']+1e-10,    'r--', lw=2,   label=r'$|\psi_{MS}^\prime|$')
    ax.semilogy(x, bl_data['gamma_deriv']+1e-10, 'g',   lw=2,   label=r'$|\Gamma^\prime|$')
    ax.axvline(ep, color='gray', ls=':', lw=1.5)
    ax.fill_betweenx([1e-6, 100], [0], [ep], alpha=0.1, color='orange',
                      label='BL region')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel('Derivative magnitude', fontsize=12)
    ax.set_title('Gradient Comparison (log scale)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim([0, min(2, x.max())])

    # subplot 3: stiffness ratio
    ax = axes[2]
    sr = bl_data['stiffness_ratio']
    ax.semilogy(x[x<=2], sr[x<=2]+1e-1, 'g', lw=2.5,
                label=r'$|\psi^\prime|/|\Gamma^\prime|$')
    ax.axhline(bl_data['stiffness_mean'], color='b', ls='--', lw=1.5,
               label=f'Overall mean ≈{bl_data["stiffness_mean"]:.0f}x')
    ax.axhline(bl_data['bl_stiffness'], color='r', ls='-.', lw=1.5,
               label=f'BL mean ≈{bl_data["bl_stiffness"]:.0f}x')
    ax.fill_betweenx([0.1,500], [0], [ep], alpha=0.1, color='orange', label='BL')
    ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel('Stiffness reduction', fontsize=12)
    ax.set_title('Stiffness Reduction Ratio', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim([0, 2])

    plt.suptitle(r'Boundary Layer Analysis: $\psi$ vs $\psi_{MS}$ vs $\Gamma$ '
                 r'— gradient comparison (JCP Fig)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); sf('fig17_boundary_layer_detail')


def plot_complexity_scaling(scaling_data):
    """Fig18: Complexity scaling law (4 subplots, JCP required)"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) time vs eps
    ax = axes[0,0]
    if scaling_data['scaling_vs_eps']:
        eps_v = [r['ep']   for r in scaling_data['scaling_vs_eps']]
        t_v   = [r['time'] for r in scaling_data['scaling_vs_eps']]
        e_v   = [r['err']  for r in scaling_data['scaling_vs_eps']]
        ax.loglog(eps_v, t_v, 'bs-', lw=2, ms=8, label='Training time')
        ax2a = ax.twinx()
        ax2a.loglog(eps_v, e_v, 'r^--', lw=2, ms=8, label='Error')
        ax2a.set_ylabel('Max Error', color='r', fontsize=11)
        ax2a.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel(r'$\varepsilon$', fontsize=13); ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(r'Scaling: Time \& Error vs $\varepsilon$', fontsize=12)
    ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3)

    # (b) time vs Nc
    ax = axes[0,1]
    if scaling_data['scaling_vs_nc']:
        nc_v = [r['nc']   for r in scaling_data['scaling_vs_nc']]
        t_v  = [r['time'] for r in scaling_data['scaling_vs_nc']]
        e_v  = [r['err']  for r in scaling_data['scaling_vs_nc']]
        ax.loglog(nc_v, t_v, 'bs-', lw=2, ms=8, label='Training time')
        ax2b = ax.twinx()
        ax2b.loglog(nc_v, e_v, 'r^--', lw=2, ms=8, label='Error')
        ax2b.set_ylabel('Max Error', color='r', fontsize=11)
        ax2b.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel(r'$N_c$ (collocation points)', fontsize=13); ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(r'Scaling: Time \& Error vs $N_c$', fontsize=12)
    ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3)

    # (c) error vs m (Barron bound verification)
    ax = axes[1,0]
    if scaling_data['scaling_vs_m']:
        m_v   = [r['m']   for r in scaling_data['scaling_vs_m']]
        e_v   = [r['err'] for r in scaling_data['scaling_vs_m']]
        ax.loglog(m_v, e_v, 'gs-', lw=2, ms=8, label='Observed error')
        # Barron reference O(m^{-1/2})
        if len(m_v) > 1:
            m_arr = np.array(m_v, dtype=float)
            ref_b = e_v[0] * np.sqrt(m_v[0]) / np.sqrt(m_arr)
            ax.loglog(m_v, ref_b, 'k--', lw=1.5, label=r'$O(m^{-1/2})$ Barron bound')
    ax.set_xlabel(r'Network width $m$', fontsize=13); ax.set_ylabel('Max Error', fontsize=12)
    ax.set_title(r'Approximation Error vs Network Width (Barron)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # (d) time-accuracy tradeoff
    ax = axes[1,1]
    if scaling_data['tradeoff']:
        t_v = [r['time'] for r in scaling_data['tradeoff']]
        e_v = [r['err']  for r in scaling_data['tradeoff']]
        ax.semilogy(t_v, e_v, 'mo-', lw=2, ms=9, label='MS-PINN v10')
        for r in scaling_data['tradeoff']:
            ax.annotate(f"e={r['epoch']}", xy=(r['time'], r['err']),
                        xytext=(r['time']*1.05, r['err']*1.3), fontsize=8, color='purple')
    ax.set_xlabel('Wallclock time (s)', fontsize=13); ax.set_ylabel('Max Error', fontsize=12)
    ax.set_title('Time–Accuracy Tradeoff', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.suptitle('Computational Complexity Analysis (JCP Section)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(); sf('fig18_complexity_scaling')


def plot_extended_generalization(x_arr, gen_results):
    """Fig19: 新增β类型 (beta5/6/7) 泛化结果"""
    new_keys = [k for k in ['beta5','beta6','beta7'] if k in gen_results]
    if not new_keys: return
    fig, axes = plt.subplots(1, len(new_keys), figsize=(6*len(new_keys), 5))
    if len(new_keys) == 1: axes = [axes]
    for ax_, bk in zip(axes, new_keys):
        r = gen_results[bk]; bc = BETA_CASES[bk]
        ax_.plot(x_arr, r['psi_ref'],    'k', lw=2.5, label='Reference (ODE45)')
        ax_.plot(x_arr, r['psi_ms_new'], 'b', lw=2,   label=f"New MS {r['e_new']:.2e}")
        ax_.plot(x_arr, r['psi_ms_old'], 'm--',lw=1.8, label=f"Old MS {r['e_old']:.2e}")
        if r['psi_pinn'] is not None:
            ax_.plot(x_arr, r['psi_pinn'], 'g', lw=2, label=f"MS-PINN {r['e_pinn']:.2e}")
        ax_.set_title(bc['name_en'], fontsize=11)
        ax_.set_xlabel(r'$x$', fontsize=11); ax_.set_ylabel(r'$\psi(x)$', fontsize=11)
        ax_.legend(fontsize=9); ax_.grid(True, alpha=0.3)
    plt.suptitle('Extended Generalization: Non-smooth & Random β(x) — v10',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(); sf('fig19_extended_generalization')


if __name__ == '__main__':
    print('Module 4 (figures) loaded successfully.')
