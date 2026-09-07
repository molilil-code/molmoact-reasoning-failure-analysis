##Molmoact实验记录

研究问题：reasoning chain 能不能在 Action 错误前暴露问题？

实验发现：
1.在同一张RGB上，HF和VLLM的输出几乎等价，Depth:99/100
HF 和 vLLM 从相同 RGB 开始，但两种 backend 的生成并非严格等价；差异先出现在 Depth，随后出现在 Trace，最后在第 2 个控制步传播到 Action。
action L2为0.00938，经过closed-loop会放大最终导致failure

Small backend-induced discrepancies first emerged in depth tokens, subsequently appeared in trajectory traces, and eventually propagated to action outputs.

2.选择backend:
取一条成功结束的HF reference trajectory，固定这条轨迹中的若干 RGB，做 open-loop HF vs vLLM 多帧静态对比

对比结果：
Depth:
平均 token agreement = 98.11%
最低 = 95%
→ 很接近

Trace:
exact match = 44.4%
平均 point distance = 1.31 px
平均 endpoint distance = 1.76 px
→ 大部分差异其实很小，但不是完全一致

Action:
从逐帧结果看
step 0   L2 = 0.04595
step 2   L2 = 0.03993
step 30  L2 = 0.07716
其他 6 帧 = 0
→ 约 1/3 的采样帧 action 有明显数值差异

速度:
HF ≈ 6.5–9.3 s
vLLM ≈ 3.6–6.3 s
→ 大约快 1.5–1.8×

接着做小规模 matched closed-loop backend study进一步验证：
每个任务各选 1 个 success + 1 个 failure
6 对 initial RGB：6/6 完全一致
outcome agreement：5/6 = 83.3%
vLLM overall speedup：1.68×

唯一的翻转是：

Coke condition_0001

HF   → success, 51 steps
vLLM → failure, 80 steps

其余 5 对的最终 success/failure 都一致，而且成功案例的步数也很接近：

Drawer condition_0004：21 vs 22
MoveNear condition_0000：58 vs 57

但是进一步比较，发现Drawer condition_0000：

HF:
failure
max_qpos = 0.121
progress = 80.5%
→ 接近成功，但没达到 0.15 阈值

vLLM:
failure
max_qpos = 0.000
progress = 0%
→ 基本完全没有完成任务进展

表面上两边都是 failure，但实际上是完全不同类型的失败。

最终选择HF backend

3.episode-level exploratory analysis

它会重点分析：

Depth
- temporal change ratio
- episode mean / p90

Trace
- num_points
- single-point degeneracy
- sustained degeneracy streak
- endpoint drift
- shape drift

Action
- model action jump
- translation jump

Task progress
- Drawer max qpos / progress
- Coke grasp / lift stage

发现：
表现为 reasoning / action 逐渐停滞、变化不足
OpenDrawer：

mean trace endpoint drift
success = 15.36
failure = 8.22

failure - success = -7.14
bootstrap CI = [-12.78, -0.13]

Trace shape drift 也是：

mean:
success = 11.03
failure = 4.85

p90:
success = 17.80
failure = 11.75

而 Depth：

p90 depth change
success = 0.736
failure = 0.516

MoveNear 也有类似现象：

mean depth change

success = 0.3746
failure = 0.3033

failure-success = -0.0713
CI = [-0.1227, -0.0126]

Coke failure:
5 episodes
其中 1 个出现 sustained single-point trace degeneration

Coke success:
1 episode
0 个出现

4.formal实验：100次rollout
pick_coke_can: 45
move_near:     35
open_drawer:   20
| Task          | Episodes | Success | Failure | Success Rate |
| ------------- | -------: | ------: | ------: | -----------: |
| Pick Coke Can |       45 |      18 |      27 |        40.0% |
| Move Near     |       35 |      15 |      20 |        42.9% |
| Open Drawer   |       20 |      12 |       8 |        60.0% |
| **Total**     |  **100** |  **45** |  **55** |    **45.0%** |

5.定义failure onset:显式 Trace 是否能在最终失败前暴露异常？
加入Trace–Depth 联合特征：
mean depth along trace
$$ \text{depth variance along trace} $$ $$ \Delta z_{\text{trace}} $$

以及跨 step：

trace-depth consistency

| Ablation | 输入                     |
| -------- | ---------------------- |
| A0       | Action                 |
| A1       | Action + Depth         |
| A2       | Action + Trace         |
| A3       | Action + Depth + Trace |
| A4       | Depth + Trace          |

| Task       | 最明显的新信号                              | Failure | Success | Failure-Success |
| ---------- | ------------------------------------ | ------: | ------: | --------------: |
| MoveNear   | mean endpoint-depth percentile drift |  0.0505 |  0.0317 |     **+0.0188** |
| OpenDrawer | mean Trace-depth gradient norm       |  0.0709 |  0.1069 |     **−0.0361** |
| OpenDrawer | mean endpoint-depth percentile drift |  0.0460 |  0.0930 |     **−0.0470** |
| Pick Coke  | mean endpoint-depth percentile drift |  0.0459 |  0.0714 |     **−0.0255** |

6. OpenDrawer depth-change dynamics：幅度阈值 turning / prominent peaks

为避免把小幅度 token-level wiggles 当成有意义的 turning，使用：

```text
delta_t = y_t - y_{t-1}
turning_t = 1[delta_t * delta_{t+1} < 0
              且 |delta_t| > epsilon
              且 |delta_{t+1}| > epsilon]
```

其中 `y_t = feat_depth_change_ratio_prev`，`epsilon = 0.05`。同时计算
prominence-filtered peak rate 和 median inter-peak interval；全程不使用
operational onset。

| 指标 | Failure (n=8) | Success (n=12) | Failure−Success | p-value |
| --- | ---: | ---: | ---: | ---: |
| qualified-pair turning rate | 0.754 | 0.694 | +0.059 | 0.122 |
| all-pairs turning rate | 0.268 | 0.415 | −0.147 | 0.0077 |
| prominent peaks / 10 steps | 2.188 | 2.523 | −0.335 | 0.058 |
| median inter-peak interval | 3.688 | 3.083 | +0.604 | 0.102 |

解释：失败样本不是显著峰更密集，而是更容易进入长时间低变化尾部；图中看到的高频 wiggles 多数是小幅度变化。因此 amplitude-thresholded turning 适合做失败模式描述，暂不作为单独在线预警器。

7. 两条 episode-level depth 规则验证（2026-09-07）

在 20 个完整 OpenDrawer episode 上验证：

```text
Rule A: episode median(feat_depth_change_ratio_prev) < 0.22 -> failure
Rule B: fraction(feat_depth_change_ratio_prev < 0.10) > 0.14 -> failure
```

两条规则结果完全一致：

| Rule | TP | TN | FP | FN | Accuracy | Failure recall | Success specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule A | 8 | 12 | 0 | 0 | 100% | 100% | 100% |
| Rule B | 8 | 12 | 0 | 0 | 100% | 100% | 100% |

对应的失败 episode 为 `C00, C02, C07, C08, C09, C10, C11, C18`。分组统计为：

```text
episode median: failure median=0.13, success median=0.37, AUC=1.0, p=0.00024
low-ratio fraction: failure median=0.402, success median=0.032, AUC=1.0, p=0.00023
```

阈值分离间隔为：median 约 `0.19–0.27`，low-ratio fraction 约
`0.118–0.161`。但这两个指标高度相关（Pearson r=-0.925），本质上是同一
stagnation 现象的两种表达，不能视为独立证据。

重要限制：上述 100% 分离是完整 episode 的 retrospective 结果。失败局大多
运行到 112 步，而成功局较短，规则会利用失败后期的停滞信息，不能直接等同于
在线 failure onset 或提前预警。若要用于 RQ2，下一步必须改为 prefix-only
统计，在训练 fold 内选择阈值，并在 held-out episode 上报告 lead time。

复现实验：

```text
python analysis/analyze_opendrawer_amplitude_dynamics.py --epsilon 0.05
python analysis/verify_opendrawer_depth_rules.py
```

主要输出：

```text
data/opendrawer_analysis/opendrawer_amplitude_dynamics_group_summary.csv
data/opendrawer_analysis/opendrawer_depth_rule_stats.csv
data/opendrawer_analysis/opendrawer_depth_rule_episode.csv
data/opendrawer_analysis/opendrawer_depth_rule_verification.png
```

目前可以得出以下结论。
1. OpenDrawer 失败与“低 depth-change / 进度停滞”明显相关。
失败 episode：
• episode median 中位数：0.13
• ratio < 0.1 的比例中位数：0.402
成功 episode：
• episode median 中位数：0.37
• ratio < 0.1 的比例中位数：0.032
因此，失败轨迹整体上更容易进入长期低变化状态。