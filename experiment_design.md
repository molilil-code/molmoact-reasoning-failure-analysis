# 实验设计文档：MolmoAct 显式空间推理链的失败自诊断研究

> 项目代号：MolmoAct-Interpretability  
> 版本：v0.4（2026-09-04，协议冻结版）  
> 硬件约束：AutoDL 云 GPU（4090 优先）+ 本地 CPU 分析  
> 目标用途：作为联系具身智能/VLA 方向导师的独立研究项目

## 版本变更记录

### v0.3 → v0.4（协议冻结）

| 优先级 | 变更 |
|---|---|
| 必须 | RQ4 改为真正的 leave-one-task-out：**2 任务训练 → 1 任务测试**（3 折）；single-source transfer 降为可选压力测试 |
| 必须 | **特征预注册**：6 个核心特征在采集完成前冻结，不做数据驱动筛选；§6.1 降级为探索性理解，不参与模型选择（防 feature selection leakage） |
| 必须 | 主指标改为 **episode 级误报率** FAR_episode = P(∃t: r_t ≥ τ \| success episode) ≤ 10%，报告失败拦截率；per-step FPR 不作为部署口径 |
| 强烈建议 | endpoint_drift 改为**相对规划向量漂移**（v_t = p_end − p_start）+ arc-length 重采样比较；处理 1-5 不定长 waypoints |
| 强烈建议 | Lead time 拆为 L_end / L_onset / L_norm 三个指标 |
| 措辞 | related work 区分 SAFE（latent 检测器）/ FPC-VLA（外部 VLM supervisor）/ ReconVLA（action 不确定性）；"复现 SAFE 结论"→"检验该观察是否迁移到 MolmoAct"；阴性结果强度收敛；"零参数量"→"零模型修改" |

**重要**：v0.4 全部修改为分析阶段变更，§3.1 采集 schema 不变，**Day 1-3 执行不受任何影响**。协议冻结后不再做结构性修改，第一批真实数据出来后再调整特征细节。

### v0.2 → v0.3（方法学修正）

定位收紧为"显式推理链 self-diagnostic signals"；RQ 重排；时间泄漏修正（r_t = f(z_{≤t})）；350 eps；无效信号删除；RQ1 因果措辞修正；episode 分组评估。

---

## 1. 项目概述

### 1.1 背景与定位

MolmoAct 的推理结构不是普通的 I,T→a，而是显式生成：

```
Image, Text → Depth tokens → Visual Trace → Action tokens → 执行
```

即 **perception–planning–control 三阶段 Action Reasoning**：深度感知 token（VQ-VAE 离散化的深度图）、视觉推理轨迹（图像上 1-5 个 future waypoints，坐标 ∈ [0,256)）、离散化动作 token（256 bins）。这提供了罕见的机会：不只问"机器人失败了吗"，而是检查**失败前模型显式空间推理结构发生了什么变化**。

**相关工作的差异化定位**：
- SAFE（NeurIPS 2025, [arXiv:2506.09937](https://arxiv.org/abs/2506.09937)）：从 VLA **末层 hidden states** 学习 failure likelihood 检测器；发现 token 级 logits 不确定性对 VLA 几乎无效（ROC-AUC 45-60%）
- FPC-VLA：引入**外部 VLM supervisor**，在关键帧以结构化视觉语言查询判断 action 可行性并给出 correction——不是 hidden-state 检测器
- ReconVLA：对 **action outputs 做 conformal uncertainty**，并在 robot state space 检测 OOD/unsafe state——接近执行前 failure anticipation

报告用差异化表述（英文定稿建议）：

> SAFE learns failure detectors from latent VLA features; FPC-VLA introduces an external VLM supervisor; ReconVLA calibrates uncertainty over action outputs and robot states. In contrast, we investigate whether the VLA's native, human-interpretable spatial reasoning representations—depth and visual traces—can themselves serve as early failure indicators.

**本项目不训练黑盒检测器，也不引入外部 supervisor**：只利用模型原生输出的显式推理链。信号天然可解释、**无需修改模型**，可用无参数阈值或轻量线性检测器实现（注意：使用逻辑回归时存在可学习参数，表述为"零模型修改"而非"零参数量"）。

### 1.2 研究问题（v0.4，协议冻结）

- **RQ1（异常溯源）**：失败 rollout 中，**最早可观测到的异常**通常出现在 Depth → Trace → Action 推理链的哪一层？异常如何向下游传播？
- **RQ2（在线预警）**：哪些显式 reasoning signals 能预测最终执行失败？**在线形式**（r_t = f(z_{≤t})，只用当前及历史推理输出）能否在动作执行前给出预警？
- **RQ3（信号消融）**：结构化推理信号（Depth/Trace）是否在 action confidence 之外提供**额外**预测信息？
- **RQ4（泛化与提前量）**：这些信号能否跨任务泛化（**leave-one-task-out：2 任务训练 → 1 任务测试**）？预警比失败早多少个决策周期（Detection Lead Time）？

### 1.3 核心假设

- **H1（主假设，"犹豫的模型更容易失败"）**：连续 replanning 之间的**相对规划向量漂移**是最强的失败预兆。定义 v_t = p_t^end − p_t^start（当前轨迹起点→终点的规划向量，扣除 gripper 整体图像运动影响），D_t^plan = ‖v_t − v_{t−1}‖₂：模型确定时连续帧的规划向量应平滑变化，犹豫时剧烈摆动。
- H2：深度感知的帧间不一致性（静态区域深度图漂移）提供感知层的补充预警信号。
- H3：动作饱和度（频繁命中 bin 两端）与动作跳变是解码层的预警信号。
- H4：Depth/Trace 信号在 action confidence 之外提供额外预测力（若 H4 不成立：结论严格限于"**在所评测任务与所提出特征下，显式推理表示未提供超出动作级信号的额外失败诊断信息**"——不得推出"推理链对动作预测无用"）。

---

## 2. 实验环境

### 2.1 硬件与系统

| 项 | 配置 |
|---|---|
| GPU（评测） | AutoDL 云实例：**4090 优先**，batch=1 实测峰值显存确认 <21GB 且稳定后，后续才考虑 3090（7B + SigLIP ViT + connector + KV cache + workspace，24GB 余量需实测） |
| 系统 | Ubuntu 22.04（AutoDL 镜像，PyTorch 2.x + CUDA 12.x） |
| 本地机器 | Windows 11 + WSL2：仅用于失败编码、回放器、分析脚本（全部 CPU 工作） |
| 推理框架 | vLLM 0.8.5（主）+ HuggingFace transformers 4.52（备选，用于取 logprobs） |

> 评测不放在本地 WSL2（SAPIEN 渲染坑）。AutoDL 采用无卡关机策略：只在跑评测时按小时租 GPU，人工工作在本地（操作手册见附录 C）。

### 2.2 模型

- 主实验对象（唯一模型）：`allenai/MolmoAct-7B-D-Pretrain-0812`（零样本）
- 动作维度说明：**256-bin 离散化确认；动作维数/步长以 checkpoint 的 norm_stats 为准**（`get_action_dim` 返回 `len(q01)`）。**已实测（2026-09-04 冒烟测试）**：`unnorm_key=fractal20220817_data` 下 **D=7（Δ位置3 + Δ姿态3 + gripper1），单步动作，无 8 步 chunk**——chunking 是 post-training 阶段才引入的，预训练 checkpoint 不适用。SimplerEnv 循环每步直接执行解析出的 7 维动作，**无时间聚合**（LIBERO 的时间聚合仅存在于其微调 checkpoint）
- future work：同一方法应用于 RT-1 微调版 / MolmoAct2 检验普适性

### 2.3 评测环境与任务

SimplerEnv，3 任务 × visual matching，覆盖三类操作结构（支持 LOTO）：

| 任务 | 类型 | episodes |
|---|---|---|
| pick_coke_can | 抓取 | 150（主任务：信号筛选 + 训练） |
| open_drawer | 关节物体操作 | 100（held-out） |
| move_near | 关系性移动（目标参照物） | 100（held-out） |

指标：逐 episode 二值成功率。预计失败样本：pick ≈ 43、drawer ≈ 33、move_near ≈ 30（按论文零样本成功率 ~70% 估算）。

---

## 3. 数据采集方案

### 3.1 插桩推理策略

在 SimplerEnv fork 中插桩 MolmoAct 策略（`simpler_env/policies/molmoact/molmoact_model.py`），每步落盘。参考本仓库 `SteerSimplerEnv/`（`changes.sh`、`maniskill2_evaluator_steer.py`、`molmoact_model_test.py`）。

**每步记录字段**：

```
episode_id, step_idx, task_name, seed
images: [base_camera 原始帧]
generated_text: 完整生成文本
parsed_depth_tokens, parsed_trace, parsed_action
logprobs: 各段（depth/trace/action）逐 token 对数似然   # Day 3 验证 vLLM 是否透传
env_state: ee_pose, gripper_state, target_obj_pose      # GT，仅 oracle 分析用
success: episode 最终成功标志（在线预测的标签）
```

**logprobs 验证（Day 3 必做）**：vLLM 0.8.5 SamplingParams 支持 logprobs，但自定义 MolmoAct 集成是否透传需实测。备选：(a) HF `generate(return_dict_in_generate=True)` 取 `scores`；(b) 丢弃 logprob 信号，仅用结构化信号。

> 本 schema 对 v0.4 全部分析指标（含重采样轨迹比较、L_onset 等）已完备，采集阶段无需任何修改。

### 3.2 回放器（并行开发）

Gradio 回放器：帧 + 深度 overlay + 轨迹 overlay + CoT 文本 + 动作箭头，支持失败标注（含 failure onset 步标注，供 RQ1/RQ4 使用）。数据采集期间并行开发。

---

## 4. 信号定义

信号分三类：**结构化推理信号**（Depth/Trace，本项目核心）、**action 信号**（基线）、**oracle 信号**（仅 RQ1 理解用，不进过滤器）。

**预注册核心特征集**（采集完成前冻结，进入在线模型与泛化检验的唯一特征来源，不做数据驱动筛选）：

```
endpoint_drift, waypoint_inconsistency, depth_temporal,
action_jump, saturation_ratio, action_logprob（若 logprobs 可用）
```

### A. 深度感知信号

| 信号 | 定义 | 类型 |
|---|---|---|
| depth_logprob | 深度段逐 token 对数似然均值 | 推理* |
| depth_entropy | 深度段逐 token 生成熵 | 推理* |
| depth_temporal | 相邻帧解码深度图在静态背景区域的平均绝对差（帧间一致性） | 推理 |
| depth_obj_dispersion | 目标物 mask 内解码深度空间方差 | oracle |
| depth_obj_corr | 解码深度 vs GT 深度在物体 mask 内相关系数 | oracle |

*依赖 logprobs 可用性（§3.1）。

### B. 视觉轨迹信号（H1 的核心信号族）

| 信号 | 定义 | 类型 |
|---|---|---|
| **endpoint_drift**（主信号） | 相对规划向量漂移：v_t = p_t^end − p_t^start，D_t^plan = ‖v_t − v_{t−1}‖₂ | 推理 |
| waypoint_inconsistency | 相邻帧轨迹 arc-length 重采样为 K=5 点、平移使首点归零后逐点平均距离（1/K）Σ‖p̃_{t,i} − p̃_{t−1,i}‖₂ | 推理 |
| trace_spread | 轨迹点空间发散度（协方差行列式，备选） | 推理 |
| trace_path_len | 轨迹路径总长度 | 推理 |
| trace_curvature | 轨迹曲率（路径长度/首尾直线距离比） | 推理 |
| trace_parse_failure | 轨迹段解析失败（生成格式非法、点数 <2 的退化轨迹单独标记） | 推理 |
| trace_endpoint_border | 轨迹终点到图像边界的距离 | 推理 |
| trace_endpoint_dist | 轨迹终点 vs GT 目标物投影质心距离 | oracle |
| trace_obj_overlap | 轨迹最后 k 点落在目标物 mask 内比例 | oracle |

> 注：MolmoAct trace 为 1-5 个**不定长** waypoints（预训练用 line_length=5 生成），所有形状比较特征须先 arc-length 重采样归一化。

### C. 动作信号（基线信号）

| 信号 | 定义 | 类型 |
|---|---|---|
| action_logprob | 动作段逐 token 对数似然均值 | 推理* |
| **saturation_ratio** | 动作 token 频繁落在 bin 两端（0-5 / 250-255）的比例 | 推理 |
| action_jump | 相邻帧动作块（反归一化后）L2 突变 | 推理 |
| dir_flip | 动作方向符号翻转（平移/旋转分量） | 推理 |
| gripper_flip | 相邻帧 gripper 开合状态翻转频率 | 推理 |
| action_chunk_var | ~~单动作块内部方差~~ **已实测无 chunk，不适用**（仅当启用 LIBERO 备份路径时恢复） | — |

### D. 语言信号（降级为探索性，不进入主叙事）

cot_len / 否定词 / 模糊词——生成文本若含稳定自然语言推理则事后计算（零成本），不作为任何结论的依据。

### E. 执行上下文（仅 RQ1 分析）

末端执行器与目标物距离、抓取闭合时机 vs 物体接触时机等 GT 信号，用于区分"策略输出正确但时序/控制器问题"。

---

## 5. 实验一：异常溯源与传播（RQ1）

**表述纪律**：depth→trace→action 串联生成，下游异常可能继承上游，观测数据无法归因根因。因此 RQ1 只研究**最早可观测异常**，不做因果声明。

### 5.1 流程

1. 全量评测（350 eps）后，对全部失败 episode 人工编码：
   - **开放编码**：归纳异常现象 → codebook（异常类型 × 出现层级）
   - **正式编码**：每局标注"最早可观测异常层级"（Depth / Trace / Action / 执行时序）+ **failure onset 步**（供 RQ4 的 L_onset）+ 异常传播路径；test-retest 一致性 ≥ 0.7
2. 统计：各任务的异常首发层级分布 + 传播链图；oracle 信号（§4E）用于确认感知层异常是否真实（解码深度 vs GT 深度对比）

### 5.2 预期产出

- 异常 codebook + 传播链图（主报告图 1）
- 典型案例库（每类 3-5 个可视化案例）
- failure onset 标注（RQ4 预警提前量的参考基准）

---

## 6. 实验二：信号分析与在线预警（RQ2）

### 6.1 阶段一：回顾性信号分析（探索性，不参与模型选择）

- 分析单元：episode；信号聚合：最差值（主）/ 均值 / 末步值
- 方法：单信号 AUROC + 点双列相关，BH 校正
- **纪律**：此阶段仅用于**理解信号性质**（哪些信号与成败相关、强弱排序、跨任务稳定性），报告中明确标注为回顾性关联，不得表述为"执行前预测"（最差值聚合含 episode 后段信息）。**进入在线模型的特征以 §4 预注册集为准**——禁止用此阶段结果增删特征（防 feature selection leakage）。

### 6.2 阶段二：在线预警（核心，无泄漏定义）

```
Observation_t → Depth_t → Trace_t → Action_t
                                    ↓
                          r_t = f(z_{≤t})    ← 只用当前及历史 reasoning outputs
                          （此时 action 尚未执行）
                       ↙            ↘
                   r_t < τ          r_t ≥ τ
                  执行 action_t     预警（abstain）
```

- **标签**：y_t = 该 episode 最终是否失败（episode 常量；预测未来结局是合法目标，泄漏只来自使用未来特征——已排除）
- **模型纪律**：单特征阈值法 + L2 正则化逻辑回归，**不上 Random Forest / XGBoost / 神经网络**——本项目贡献在 reasoning signals，不在分类器
- **防泄漏**：按 episode 划分 train/test（同 episode 的 steps 不可跨划分）；**特征标准化与阈值选择仅在训练 fold 内完成**
- **主指标（episode 级口径）**：
  - FAR_episode = P(∃t: r_t ≥ τ | success episode) ≤ 10% 约束下，报告失败拦截率 FDR
  - **不报告 per-step FPR 作为部署口径**：成功 episode 约 50 步，per-step 10% 意味着 episode 级几乎必然报警
- **评估纪律（防伪相关）**：step 级样本在同 episode 内高度相关，必须同时报告 per-step AUROC + **episode 分组 bootstrap 置信区间** + episode 级聚合指标
- **Prefix evaluation**：分别只看 episode 前 10% / 25% / 50% / 75% 的 rollout 做预测，绘制 **AUROC vs rollout progress 曲线**——回答"需要观察多少比例才能预警"，比混合所有 step 的单一 AUROC 更易解释

---

## 7. 实验三：特征消融阶梯（RQ3）

**零额外算力**（复用已采集数据），回答"结构化推理信号是否比 action confidence 提供额外信息"：

| 条件 | 特征集 | 回答的问题 |
|---|---|---|
| A0 | action_logprob 单独 | token 置信度基线（**检验 SAFE 关于 token 不确定性弱的观察是否迁移到 MolmoAct**——模型、tokenization、benchmark 均不同，不称"复现"） |
| A1 | A0 + action 结构化信号（saturation/jump/flip） | 动作层面的结构化信息增量 |
| A2 | Trace 信号族单独 | 轨迹信号独立能力 |
| A3 | Depth + Trace | 感知+规划组合 |
| A4 | Depth + Trace + Action 全部 | 完整推理链 |

报告每条件 AUROC（episode 级）。结果解释：
- **A3/A4 显著优于 A0/A1** → H4 成立，项目核心贡献成立
- **无增量** → 结论严格限于："**在所评测任务与所提出特征下，显式推理表示未提供超出动作级信号的额外失败诊断信息**"。不宣称推理链对动作预测无用，不宣称"推理只是装饰"

---

## 8. 实验四：跨任务泛化与预警提前量（RQ4）

### 8.1 Leave-one-task-out（3 折，2 任务训练 → 1 任务测试）

| Fold | 训练 | Held-out |
|---|---|---|
| 1 | Pick + Drawer | Move Near |
| 2 | Pick + Move Near | Drawer |
| 3 | Drawer + Move Near | Pick |

- 特征固定为 §4 预注册集；标准化与阈值在训练 fold 内完成；held-out 只测试一次
- 报告：每折 held-out AUROC（episode 级）+ FAR_episode ≤ 10% 约束下的 FDR
- 可选压力测试（时间允许）：single-source transfer（1 任务训练 → 1 任务测试），报告衰减

### 8.2 Detection Lead Time（三个指标）

```
        t_alert          t_onset         T (episode 结束)
          ↑                ↑                ↑
          |---- L_onset ---|
          |------------- L_end -------------|
```

- **L_end** = T − t_alert：距 episode 结束的步数
- **L_onset** = t_onset − t_alert：预警早于人工判定 failure onset 的步数；**L_onset > 0 说明模型内部信号在"人类观察者认为失败已明显发生"之前报警**——本项目比 AUROC 更接近机器人安全研究的指标（依赖 RQ1 编码的 onset 标注）
- **L_norm** = L_onset / T：归一化提前量，跨任务可比（Pick/Drawer/Move Near 的 episode 长度不同）

---

## 9. 统计与效度

- 成功率报告 95% Wilson CI
- 多重比较：BH 校正（§6.1，信号数 ~15）
- 防泄漏：在线模型按 episode 划分；特征只用 z_{≤t}；**标准化与阈值只在训练 fold 内**；**核心特征预注册冻结，held-out 不参与任何筛选**
- 分组评估：step 级指标的置信区间按 episode 分组 bootstrap（§6.2）
- 外部效度声明：单一模型、3 任务、仿真环境；普适性验证（其他 VLA、MolmoAct2）列为 future work

---

## 10. 时间线（6 周）

| 周 | 里程碑 | 并行任务 |
|---|---|---|
| 1 | AutoDL 环境 + 插桩（附录 C） | 回放器开发 |
| 2 | 数据采集（350 eps，无人值守数小时）+ 第一批失败编码 | 信号提取脚本 |
| 3 | 失败编码完成（~100 失败局 × 5-8 分钟 ≈ 9-14h 人力，关键路径） | §6.1 回顾性分析 |
| 4 | RQ2 在线预警（§6.2 全指标，含 prefix evaluation） | RQ3 消融阶梯 |
| 5 | RQ4 LOTO + lead time 三指标 | — |
| 6 | 缓冲：报告 + demo 视频 + GitHub 打磨 | 若顺利，第 5 周末进入交付 |

**关键路径是人工编码**。若时间不足：失败编码的细粒度 codebook 只标注 pick 任务的失败局，drawer/move_near 仅标"最早异常层级 + onset 步"粗类。

---

## 11. 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| SimplerEnv 在 AutoDL 环境崩溃 | 低 | 回退 LIBERO（同插桩方案）；或换实例重装（镜像已存，成本低） |
| vLLM 不暴露 logprobs | 中 | HF scores 方案；或从预注册集删除 action_logprob/depth_logprob（结构化信号足以支撑 H1） |
| 失败模式无法清晰归层 | 中 | 粗粒度三层 + "混合"标签；RQ2-RQ4 不受影响 |
| H4 不成立（消融无增量） | 中 | 按 §7 限定措辞的阴性结果写，故事依然完整 |
| 编码工作量超预期 | 中 | §10 的粗类标注预案 |
| 3090 显存不足 | — | 4090 实测通过后再评估，不预设 |
| L_onset 标注困难（onset 判定主观） | 中 | onset 定义操作化：编码手册规定"首次出现不可挽回行为迹象"的判定标准；不可靠则只报告 L_end/L_norm |

---

## 12. 预期产出物清单

1. **分析报告**（4-6 页，英文，arXiv style）：RQ1 异常传播链 + RQ2 在线预警（episode 级 FAR/FDR + prefix 曲线）+ RQ3 消融阶梯表 + RQ4 LOTO 与 lead time
2. **GitHub 仓库**：插桩评测代码 + 分析脚本 + 回放器
3. **Demo 视频**（1-2 分钟）：回放器 + 实时风险判定 + 预警案例
4. **异常 codebook**（文档）
5. **联系导师邮件素材**：一句话研究问题 + 链接

---

## 附录 A：与仓库文件的对应关系

| 用途 | 文件 |
|---|---|
| 零样本推理 prompt | `olmo/hf_model/molmoact/test_molmoact.py` |
| 深度/轨迹/动作解析 | `olmo/hf_model/molmoact/modeling_molmoact.py`（`parse_depth/parse_trace/parse_action`、`get_action_dim`） |
| 深度 token 解码 | `scripts/reconstruct_from_tokens.py`（VQVAE 解码器） |
| 插桩参考 | `SteerSimplerEnv/maniskill2_evaluator_steer.py`、`SteerSimplerEnv/molmoact_model_test.py` |
| 评测脚本 | SimplerEnv fork 的 `scripts/molmoact_*_visual_matching.sh`（3 个任务） |
| 备选评测 | `experiments/LIBERO/run_libero_eval_vllm.py` |

## 附录 B：待验证清单（动手前先确认）

- [ ] vLLM 0.8.5 自定义 MolmoAct 模型是否支持 `SamplingParams.logprobs`（Day 3 必做）
- [x] Pretrain checkpoint 的实际动作维度与步长 → **已实测（2026-09-04）：D=7 单步动作**（fractal20220817_data，见 §2.2）
- [ ] SimplerEnv 评测的 episode 数与 seed 控制参数
- [ ] 环境是否提供目标物 segmentation mask 与相机内参（oracle 信号依赖）
- [ ] 4090 batch=1 峰值显存实测（决定后续是否换 3090）

## 附录 C：周 1 操作手册（AutoDL 方案）

### C.1 租机与成本策略

- **机型**：4090（24GB）优先；镜像 Ubuntu 22.04 + PyTorch 2.x + CUDA 12.x
- **存储**：数据盘 ≥ 60GB（模型 14GB + 资产 + 350 eps 落盘数据）
- **省钱模式**：跑完评测立即无卡关机，人工工作回本地。预计全程 GPU 租金 150-300 元
- **保险**：环境跑通当天保存自定义镜像
- **国内下载**：`export HF_ENDPOINT=https://hf-mirror.com`

### C.2 分日计划

**Day 1 — 租机与基础依赖**：租 4090 实例 → conda env（Python 3.10）→ `pip install einops torchvision accelerate vllm==0.8.5 transformers==4.52`（版本精确匹配）→ hf-mirror 下载模型（~14GB）→ **预下载 Qwen/Qwen2-7B tokenizer 至 HF 缓存**（`parse_action` 懒加载会隐式下载；本地离线分析也需要，需拷贝缓存）→ `unset OMP_NUM_THREADS`（容器预设值非法会报 libgomp 警告）

**Day 2 — SimplerEnv 安装**：克隆 allenai/SimplerEnv 按 MolmoAct Inference Setup 安装；3 个目标任务是 Google Robot 系（sapien），无需 mujoco；首次运行触发资产下载，PartNet Mobility 许可按文档处理

**Day 3 — 两级冒烟测试 + 两项必做验证**：
1. 模型级：`test_molmoact.py` 喂 2 张图验证生成与 parse 接口
2. 环境级：官方 `molmoact_pick_coke_can_visual_matching.sh` 完整跑通
3. **验证动作维度/步长**（`parse_action` 输出长度）
4. **验证 logprobs 透传**（vLLM SamplingParams），不可用 → 当天切 HF scores 方案
5. 记录 batch=1 峰值显存（nvidia-smi 监控），决定后续机型

**Day 4-5 — 插桩**：参考 `SteerSimplerEnv/` patch 方式落盘 §3.1 字段；10 eps 验证字段齐全

**Day 6 — 小批量采集 + 回放器并行**：三任务各 10 eps；Gradio 回放器骨架

**Day 7 — 缓冲**：补漏 → 保存镜像 → 无卡关机

### C.3 周 1 退出标准

- [ ] 3 任务 × ≥10 eps 全字段落盘
- [ ] logprobs 可用性结论已定（可用 / 已切换备选）
- [ ] 动作维度/步长已实测确认
- [ ] 4090 峰值显存已记录
- [ ] 回放器能完整显示一局
- [ ] 自定义镜像已保存
