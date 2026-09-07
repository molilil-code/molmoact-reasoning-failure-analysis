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

