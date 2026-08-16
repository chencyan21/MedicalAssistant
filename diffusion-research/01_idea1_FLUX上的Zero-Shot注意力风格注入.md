# Idea 1：FlowStylist — FLUX 上的 Zero-Shot 层归因注意力风格注入

> 一句话：在 flow-matching DiT（FLUX.1-dev）上重建 StyleID/InstantStyle 式免训练风格注入范式——先做 MM-DiT 的"风格层归因"，再做 RF 反演校正 + 按层×时间步的注入调度，填补该范式在新一代 backbone 上的空白。

---

## 1. 研究动机与问题定义

**背景**：StyleID（CVPR'24 Highlight）、InstantStyle 等证明了"自注意力 K/V 是风格载体"，通过反演 + K/V 替换即可 zero-shot 风格迁移。但这类方法全部建立在 SD1.5/SDXL 的 U-Net 上。

**空白**（综述 §3.3）：
1. **架构断裂**：FLUX/SD3 的 MM-DiT 采用文图联合注意力，K/V 中文本与图像 token 拼接，SD 时代的 K/V swap 无法直接移植；"哪些层/哪些头承载风格"在 MM-DiT 上完全未知（ReFlex, ICCV'25 仅做了初步特征分析）。
2. **反演断裂**：DDIM 反演在 RF 模型上误差显著更大（DirectEdit, ICML'26 指出步级误差问题），RK-Solver/定点迭代校正尚未与风格注入结合。
3. **调度断裂**：Scheduled Style Injection（arXiv:2605.26538）刚在 SD1.5 上提出按层×时间步调度，FM 模型上完全空白。

**问题定义**：给定内容图 I_c、风格图 I_s，在 FLUX.1-dev 上免训练生成 I_out，最大化风格相似度的同时保持内容结构——即"FLUX 版 StyleID"，并进一步给出**风格-内容 Pareto 前沿的可控滑杆**。

**为什么值得做**：① 真空位，社区需求明确（FLUX 官方 Redux 解耦能力有限）；② 免训练方法算力门槛低，适合快速产出；③ 机理分析（层归因）本身有独立贡献；④ 可与 KontextBench/StyleBench 直接评测。

---

## 2. 相关工作与差异化

| 工作 | 与本 idea 的关系 |
|---|---|
| StyleID (2312.09008) | 范式源头；本文将其三组件（query preservation、温度缩放、latent AdaIN）移植并推广到 MM-DiT |
| InstantStyle (2404.02733) | 层归因思想源头；本文对 MM-DiT 做系统性层×头×token 类型归因 |
| RF-Inversion (2410.10792) / RK-Solver (2509.12888) | 提供 FLUX 反演基础；本文在其上加反演误差校正 |
| Scheduled Style Injection (2605.26538) | 调度思想；本文扩展到 FM 模型并联合频段维度（衔接 Idea 2） |
| OmniStyle (2505.14028) | 数据驱动 SOTA，作为对比上界；本文主打免训练可控性 |
| ReFlex (2507.01496) | MM-DiT 特征分析的先例，本文扩展到风格归因 |

**差异化**：首个在 flow-matching DiT 上系统解决"反演—归因—注入—调度"全链路的免训练风格迁移方法。

---

## 3. 方法设计

### 3.1 总体管线

```
内容图 I_c ──► RF反演(校正) ──► z_T^c ──┐
                                        ├─► 联合采样：按层×时间步调度做风格KV注入 ──► I_out
风格图 I_s ──► RF反演(校正) ──► z_T^s ──┘        ▲
                                    归因图谱 A(l, h, t)（离线一次性分析）
```

### 3.2 模块一：MM-DiT 风格层归因（机理贡献）

- 设计探针实验：固定内容 prompt，扫过 100+ 风格图，记录每层 double/single-stream 块中图像 token 的 K/V 统计量（通道均值方差 = AdaIN 统计、与 CSD 风格嵌入的相关性、注意力熵）。
- 输出 **风格归因图谱** A(l, h)：识别 FLUX 中类似 SDXL"风格块"的关键层/头。假设：风格信息集中于中后段 single-stream 块的图像 K/V（待验证）。
- 同样归因"内容/布局块"（类似 InstantStyle 的 `down_blocks.2.attentions.1`），用于定义注入禁区。

### 3.3 模块二：校正 RF 反演

- 基线：RF-Inversion 的向量场构造；
- 校正：引入 RK-Solver（高阶龙格-库塔）或定点迭代减小步级误差；对内容图额外采用 StyleSSP 式**反演 latent 低频削弱**（衔接 Idea 2 的频域手段）以保布局。

### 3.4 模块三：调度化风格注入

在采样的速度场估计中，对归因图谱选定的层做 K/V 替换，注入强度由调度函数控制：

- **按层**：K_v(l) ← α(l)·K_v^s + (1−α(l))·K_v^c，α(l) 由归因图谱 A 归一化得到；
- **按时间步**：α(l,t) = α(l)·s(t)，s(t) 单调递减（早期强注入定风格基调，晚期弱注入保细节）——呼应 Fourier Space Perspective（2505.11278）"先结构后纹理"结论；
- **Query preservation**：Q̃ = γ·Q_c + (1−γ)·Q_cs，γ 可学习为 γ(t) 或作为用户滑杆；
- **MM-DiT 特有问题**：联合注意力中文本 token 与图像 token 拼接，K/V 替换需仅在图像 token 段进行，并用因果掩码实验验证文本条件不被污染。

### 3.5 输出：风格-内容 Pareto 滑杆

将 (γ, s(t) 的幅度, 注入层集合) 参数化为 1–2 个用户可控标量，扫描生成 Pareto 前沿（风格相似度 vs 内容保持度），提供连续滑杆——这是相对 StyleID 固定 γ=0.75 的直接改进。

---

## 4. 实验方案

**Backbone**：FLUX.1-dev（主）；SD3.5-Medium（验证架构泛化）；SD1.5（复现 StyleID 作为 sanity check）。

**Benchmark**：StyleBench（40 内容 × 73 风格）；KontextBench 风格子集；自建 20 内容 × 20 风格难例集（强风格 + 精细结构）。

**Baselines**：StyleID/InstantStyle（SD1.5 上跑，跨 backbone 对比）、FLUX Redux、OmniStyle（数据驱动上界）、RB-Modulation、CSGO。

**指标**：
- 风格：CSD score（注意用 CSD+ 的 CSLS 读出，2605.09030）、ArtFID、CLIP-I；
- 内容：DINOv2 结构相似度、LPIPS、DreamSim、CFSD；
- 人工评测：成对偏好（≥30 人，报告一致性）。

**消融**：① 归因图谱 vs 全层注入 vs 随机层；② 调度 s(t) 形状（常数/线性/余弦/单调递减）；③ 反演校正开/关；④ γ 固定 vs γ(t)；⑤ 文本 token 掩码的必要性。

**预期结果**：风格相似度接近 OmniStyle（免训练 vs 训练的差距 <10%），内容保持显著优于 Redux，且提供前两者不具备的连续滑杆。

---

## 5. 可行性与风险

**可行性**：免训练方法，单卡 24–48GB（量化 FLUX-dev）即可跑全部实验；反演与注意力修改代码可从 RF-Inversion/diffusers 直接改。预计 2–3 个月出主结果。

**风险与对策**：
1. **MM-DiT 联合注意力导致注入污染文本条件** → 图像 token 段隔离 + 文本掩码消融（§3.4）；若失效退而求其次只改图像自注意力分量。
2. **RF 反演误差使内容崩坏** → RK/定点校正 + 低频削弱；极端情况改用 SDEdit 式部分加噪（牺牲一定保真换鲁棒）。
3. **风格归因图谱不显著（风格分散于全部层）** → 转为"逐头归因 + 稀疏选择 top-k 头"，或将归因本身写成分析论文（仍有价值）。
4. **与 OmniStyle2 数据驱动路线差距过大** → 将本文定位为"可控性 + 零成本"路线，并讨论作为数据引擎（用本文生成训练对）的协同。

**目标会议**：CVPR/ICCV/NeurIPS 2027（2026 下半年投稿窗口紧，视进度可选 SIGGRAPH Asia）。

---

## 6. 时间规划（建议）

| 周 | 任务 |
|---|---|
| 1–2 | 复现 RF-Inversion + StyleID（SD1.5 sanity check）；搭 FLUX 反演管线 |
| 3–5 | 归因探针实验，产出风格归因图谱 A |
| 6–8 | 注入机制 + 调度实现，主实验 |
| 9–10 | 消融 + 人工评测 + Pareto 前沿扫描 |
| 11–12 | 写作与投稿 |

## 未确认项

- Scheduled Style Injection（2605.26538）、ReFlex（2507.01496）为预印本，结论待同行评审；
- FLUX 内部"风格层"位置为先验假设，需探针实验验证。
