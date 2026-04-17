# 破产概率联合求解框架

## 代码结构

| 文件 | 职责 | 行数 |
|------|------|------|
| `module1_core.py` | 核心数学函数、β族定义、解析解、R-ODE系数 | ~290 |
| `module2_training.py` | 神经网络、训练流程、经典基准求解器、消融实验 | ~380 |
| `module3_studies.py` | 收敛分析、泛化、λ敏感性、边界层解析、复杂度分析| ~310 |
| `module4_figures.py` | 所有图表生成 | ~460 |
| `main.py` | 主程序调度| ~180 |


### ✅ 1. 经典数值方法基准对比 (Fig 15)
位置: `module2_training.py` → `benchmark_classical_solvers()`

新增求解器:
- **FDM** (二阶中心差分，Thomas算法)
- **Chebyshev Spectral** (Chebyshev微分矩阵，重心插值)
- **Quadrature-based IDE solver** (累积梯形积分)

输出: 方法 × {精度, 时间, 稳定性} 对比表格

### ✅ 2. 消融实验 Ablation Study (Fig 16)
位置: `module2_training.py` → `run_ablation_study()`

5个变体:
1. PINN_baseline (无MS先验)
2. MS_only (+ MS先验)
3. MS+Ratio (+ Ratio参数化)
4. MS+Ratio+RODE (+ R-ODE物理)
5. **Full MS-PINN** (完整两阶段，本文方法)

输出: 误差对比 / 训练时间 / 损失曲线

### ✅ 3. 收敛性理论加强
位置: `module1_core.py` → `verify_operator_coercivity()` + `error_decomposition()`

新增:
- **算子适定性验证**: 证明 P_eff(x)<0 everywhere (coercive), ||Q||<∞
- **三项误差分解**:
  - E_asym ~ O(ε²)  [MS渐近误差]
  - E_approx ~ O(m^{-1/2})  [Barron 近似误差, C_f估计]
  - E_optim = 训练残差  [优化误差]

### ✅ 4. 边界层解析比较 (Fig 17)
位置: `module3_studies.py` → `boundary_layer_analysis()`

输出:
- ψ vs ψ_MS vs Γ 曲线（boundary layer放大）
- gradient comparison (log scale)
- Stiffness reduction ratio 图

### ✅ 5. 泛化升级 (Fig 19)
位置: `module1_core.py` BETA_CASES → beta5/6/7
- **beta5**: 不连续阶跃 β(x)
- **beta6**: sharp gradient β (tanh, 10x陡变)
- **beta7**: random piecewise β (随机分段)

### ✅ 6. 复杂度分析 Scaling Law (Fig 18)
位置: `module3_studies.py` → `complexity_scaling_study()`

4子图:
- (a) 时间 vs ε  (fixed N_c, m)
- (b) 时间 vs N_c (配置点数)
- (c) 误差 vs m  (网络宽度, Barron验证)
- (d) 时间-精度 tradeoff 图

## 运行方式

```bash
cd ruin_v10
python main.py
```

依赖:
```
numpy scipy torch matplotlib
```

## 输出文件

- `fig1_wkb_ms_solution.pdf/.png`  ... 原有14张
- `fig15_classical_benchmark`  
- `fig16_ablation_study`
- `fig17_boundary_layer_detail`
- `fig18_complexity_scaling`
- `fig19_extended_generalization`
- `pinn_model_v10.pth`
- `mspinn_model_v10.pth`
