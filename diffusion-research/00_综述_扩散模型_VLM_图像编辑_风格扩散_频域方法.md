# 综述：扩散模型骨干、VLM 图像编辑、风格扩散与频域方法的研究进展（2022–2026.08）

> 调研时间：2026 年 8 月。覆盖五个交叉主题：扩散模型 backbone 演进、VLM×图像编辑、扩散风格迁移、频域（小波/傅里叶）方法、以及四者交叉处的研究空白。本文是后续 Idea 文档（01–04）的综述基础。

---

## 1. 扩散模型骨干网络：从 U-Net 到 Flow-Matching DiT

### 1.1 SD2.x 系列

Stable Diffusion 2.0/2.1（Stability AI + CompVis，2022.11–12）沿用 Latent Diffusion 框架（arXiv:2112.10752）：8 倍下采样 VAE + 865M U-Net。与 SD1.x 的关键差异：
- 文本编码器由 CLIP ViT-L/14 换成 **OpenCLIP ViT-H/14**（1024 维），训练数据剔除 NSFW 内容；
- 768×768 版本采用 **v-prediction** 参数化（源自 Progressive Distillation, arXiv:2202.00512），高分辨率信噪比曲线更优；512 版本仍为 ε-prediction。

**社区现状**：SD2.x 因 prompt 习惯不兼容、人物生成能力削弱，社区采纳度远低于 SD1.5。SD1.5 至今仍是微调生态最大的底座（ControlNet、LoRA、InstructPix2Pix 等经典方法均建于其上）。**以 SD2.x 为科研 backbone 的合理动机**：① 单卡可全参微调，方法学验证成本低；② 865M 规模便于做机理分析与消融；③ 与 SD1.5 基线可公平对比。代价是生成质量与文本理解明显落后于 DiT 代模型，若目标为 SOTA 编辑效果，说服力有限。

### 1.2 SD3 / SD3.5：MMDiT + Rectified Flow

**SD3**（Stability AI，2024.03，arXiv:2403.03206，ICML 2024）：
- **MMDiT**：文本与图像 token 各有独立权重但在同一注意力中双向交互；
- **Rectified flow** 目标：z_t = (1−t)x₀ + tε，预测速度 v ≈ ε − x₀；提出 **logit-normal 时间步重加权**——在对比 60 种轨迹后证明其全步数区间最优；
- 三文本编码器（CLIP-L + OpenCLIP-G + T5-XXL）。

**SD3.5**（2024.10）：Large 8B（QK-Norm）、Large Turbo（蒸馏）、Medium 2.5B；Stability Community License（科研免费）。SD3 论文是 flow-based backbone 的必读消融文献。

### 1.3 FLUX.1：Flow Matching 的规模化

**FLUX.1**（Black Forest Labs，2024.08；技术细节见 Kontext 报告 arXiv:2506.15742）：
- 12B 混合 Transformer：若干 **double-stream 块** + 38 个 **single-stream 块**；**3D RoPE** 提供分辨率/宽高比泛化；
- T5-XXL（256 token）+ CLIP pooled 双文本编码器；flow matching 目标 + **guidance distillation**；16 通道高重建 VAE。
- 版本：**dev**（开放权重、非商业许可，科研免费）、**schnell**（4 步，Apache 2.0）、pro（API-only）。

### 1.4 Flow Matching 的理论优势

- **Flow Matching**（Lipman et al., ICLR 2023, arXiv:2210.02747）与 **Rectified Flow**（Liu et al., 2022, arXiv:2209.03003）：学习接近直线的概率路径，ODE 离散误差小 → 少步采样质量损失显著降低（4 步 vs 传统 25–50 步），且更利于蒸馏。
- **注意**：SD3 论文指出原始均匀采样的 RF 在多步区间反而退化，logit-normal 重加权不可省略；"直线性"仅在单轮 reflow 后近似成立。

### 1.5 2025–2026 新动向

| 模型 | 机构 | 时间 | 要点 | 许可 |
|---|---|---|---|---|
| FLUX.1 Kontext (dev) | BFL | 2025.06 (arXiv:2506.15742) | 参考图 VAE token 序列拼接进视觉流，统一局部/全局编辑与风格参考；KontextBench | 非商业 |
| Qwen-Image / -Edit | 阿里 | 2025.08 (arXiv:2508.02324) | 20B MMDiT + 冻结 Qwen2.5-VL 条件编码；VAE+Qwen2.5-VL 双编码兼顾保真与语义 | Apache 2.0 |
| Step1X-Edit | 阶跃星辰 | 2025.04 (arXiv:2504.17761) | MLLM 解析指令 + connector + FLUX 式 DiT；GEdit-Bench | Apache 2.0 |
| FLUX.2 | BFL | 2025.11 | dev 仍非商业，klein 4B 为 Apache 2.0 | 混合 |

### 1.6 Backbone 选型结论

| | SD1.5 | SD2.x | FLUX.1-dev | FLUX.1-schnell / Qwen-Image |
|---|---|---|---|---|
| 规模/成本 | 860M，单卡全参微调 | 865M，同左 | 12B，需量化/多卡 | 12B / 20B |
| 训练目标 | ε-DDPM | v-prediction | flow matching + 引导蒸馏 | flow matching |
| 生成质量 | 低（512px） | 中低 | 高 | 中高 |
| 许可 | OpenRAIL | OpenRAIL++ | 非商业（科研 OK） | Apache 2.0 |
| 生态/基线 | 最大 | 弱 | 快速增长 | 快速增长 |

**建议**：方法学研究 + 经典基线对比 → **SD1.5/SD2.x**（SD2.x 仅在需要 v-prediction/OpenCLIP-H 时有独特价值）；面向 SOTA 投稿（2026 顶会）→ **FLUX.1-dev** 为主实验底座，**Qwen-Image-Edit 或 schnell** 作开源发布版本以规避许可问题。

---

## 2. VLM 与指令式图像编辑

### 2.1 指令编辑方法脉络

- **InstructPix2Pix**（Berkeley, arXiv:2211.09800, CVPR'23）：GPT-3+PnP 合成三元组微调 SD1.5，免反演编辑的开山之作；局限是指令理解弱、易全局漂移。
- **MagicBrush**（arXiv:2306.10012, NeurIPS'23）：首个真实标注多轮编辑基准。
- **Emu Edit**（Meta, arXiv:2311.10089）：16 类任务 + 任务嵌入多任务训练。
- **UltraEdit**（字节, arXiv:2405.19779）：400 万样本 + region-anchored 局部编辑数据。
- **ICEdit**（浙大+哈佛, arXiv:2504.20690, NeurIPS'25）：FLUX 上双联画 in-context prompt + LoRA-MoE，仅 5 万数据/200M 参数达强效果；提出 **VLM 早筛推理时缩放**（早期去噪步筛选初始噪声）。
- **FireEdit**（中山+腾讯混元, arXiv:2503.19839, CVPR'25）：开放词汇检测器生成 region tokens 增强 VLM 细粒度感知；Time-Aware Target Injection 按去噪阶段调强度。
- **SmartEdit**（Google, arXiv:2312.10939）：LLaVA 输出经 BIIF 转译为扩散条件，专攻推理型复杂指令。

### 2.2 VLM 的四类角色

1. **指令理解/数据引擎**：MGIE（Apple, arXiv:2309.17102）；HQ-Edit/UltraEdit 用 GPT-4V 构造数据与评估。
2. **CoT 编辑规划**：InsightEdit（字节, arXiv:2411.04423）CoT 分解复杂指令；RetouchIQ（CVPR'26）MLLM agent 生成 Lightroom 参数 + RL。
3. **Reward/评估器**：VIEScore（arXiv:2405.14867）；**EditReward**（ICLR'26，Qwen2.5-VL 骨干，多维成对排序，用于数据筛选与 best-of-N）；EditHF-1M（百万级人类偏好 + RL 微调 Qwen-Image-Edit）。
4. **区域定位**：CAMILA（NeurIPS'25, arXiv:2509.19731）MLLM 输出 [MASK] token 经解码生成二值 mask 调制 cross-attention；SmartFreeEdit（arXiv:2504.12704）MLLM 分解指令→SAM 分割→inpainting。

### 2.3 统一理解-生成模型

Emu3（智源, arXiv:2409.18869，纯 AR）；Janus-Pro（DeepSeek, arXiv:2501.17811，理解/生成编码解耦）；Show-o2（AR+离散扩散混合）；**Bagel**（字节, arXiv:2505.14683，MoT 双编码器，涌现编辑能力）；OmniGen2（智源, arXiv:2506.18871，Phi-3+VAE+RF）。**启示**：编辑本质是"理解+生成"耦合任务，token 拼接 in-context 编辑正取代外挂 adapter；但 AR 路线像素保真度逊扩散，"MLLM 语义 + 扩散解码"短期更实用。

### 2.4 可行结合点

① VLM 条件替换/融合（Step1X-Edit 路线）；② VLM 区域定位 + 注意力调制（CAMILA 范式）；③ VLM Reward 闭环（数据筛选 / GRPO 后训练 / best-of-N）；④ CoT 编辑规划；⑤ VAE token + VLM 语义 token 双流条件（Bagel/Kontext 思路）。

---

## 3. 风格扩散：风格-内容解耦与 Zero-Shot 风格注入

### 3.1 注意力类免训练方法

- **StyleAligned**（Google, arXiv:2312.02133, CVPR'24 Oral）：自注意力 K/V 上施加 AdaIN + 共享注意力；证明自注意力 K/V 是风格主要载体。
- **StyleID**（SKKU, arXiv:2312.09008, CVPR'24 Highlight）：DDIM 反演 + 解码器第 6–11 层 K/V 替换；三组件——Query preservation（γ=0.75）、注意力温度缩放（τ=1.5）、初始 latent AdaIN。局限：γ 全局固定、需两次反演。
- **InstantStyle**（InstantX, arXiv:2404.02733）：CLIP 图像特征减内容文本特征做显式解耦；发现 SDXL 两个"风格块"（`up_blocks.0.attentions.1`、`down_blocks.2.attentions.1`）——"层归因"思想被大量沿用。
- **Scheduled Style Injection**（arXiv:2605.26538，预印本）：将 γ 扩展为按层×按时间步调度，刻画风格-内容 Pareto 前沿——目前仅在 SD1.5 上验证。

### 3.2 Adapter 与数据驱动方法

IP-Adapter（腾讯, arXiv:2308.06721）解耦交叉注意力；StyleShot（arXiv:2407.01414）免测试时微调 + StyleBench；CSGO（arXiv:2408.16766）style tokens + IMAGStyle 21 万三元组；StyleStudio（西湖大学, arXiv:2412.08503, CVPR'25）跨模态 AdaIN + 风格化 CFG（风格元素选择性开关）；**OmniStyle**（南大+字节, arXiv:2505.14028, CVPR'25）首个百万级三元组 + FLUX-dev 端到端；**OmniStyle2**（arXiv:2509.05970）"去风格化造数据"范式——标志该方向从机制设计转向数据驱动。

### 3.3 反演与 FLUX 时代

- **RF-Inversion**（Google, arXiv:2410.10792, ICLR'25）：FLUX 上反演与编辑的奠基工作；
- **RK-Solver 反演**（arXiv:2509.12888）、**DirectEdit**（ICML'26）：修正 RF 反演步级误差；
- **StyleSSP**（浙大+字节, arXiv:2501.11319）：反演 latent 低频削弱保布局 + 反演阶段负向引导；
- FLUX.1 Redux（官方图像条件适配器，解耦能力有限）。

**关键判断**：截至 2026 年中，**针对 FLUX/SD3 的 zero-shot 注意力风格注入（StyleID 式）基本空白**——MM-DiT 联合注意力使 K/V swap 不能直接移植，RF 反演误差更大。这是明确的真空位。

### 3.4 评估体系

CSD score（arXiv:2404.01292）是事实标准，但 2026 年工作（arXiv:2605.09030）指出其余弦分数系统性失效，提出 CSD+；内容侧用 CLIP/DINOv2/LPIPS/DreamSim；Benchmark 有 StyleBench、OmniFilter。**缺乏公认、与人类判断强对齐的统一 benchmark**。

---

## 4. 频域方法：小波/傅里叶 × 扩散模型

### 4.1 机理分析

- **FreeU**（NTU, arXiv:2309.11497, CVPR'24）：傅里叶域分析 U-Net——backbone 主去噪（抑制高频），skip 注入高频；推理时 backbone 特征放大 + skip 频域衰减即可免训练提质。**注意**：结论针对 U-Net，FLUX 的 MM-DiT 无 skip，机理需重新验证。
- **A Fourier Space Perspective on Diffusion Models**（Cambridge, arXiv:2505.11278, ICML'25）：证明高频在 SNR 意义下更早被破坏，逆过程"先低频结构、后高频细节"——为**分时间步的频域调度**提供理论依据。

### 4.2 幅度-相位分解与风格

信号处理共识：**相位编码空间结构，幅度编码纹理/光照/对比度**（FDA, CVPR'20, arXiv:2004.10998）。
- **Fourier Phase Diffusion**（中科大等, IJCAI'25 Oral）：风格化时向中间样本注入内容图相位谱锁定结构，不干扰幅度（风格）生成，免训练超 SOTA；
- **NeuralRemaster ϕ-PD**（Toyota Research 等, arXiv:2512.05106）：把前向噪声改为"保相位、随机化幅度"的结构化噪声 + 频选截止 FSS 控制结构刚性，模型无关。

### 4.3 小波扩散

WSGM（Mallat 组, arXiv:2208.05003, NeurIPS'22）跨尺度小波系数条件生成；WaveDM（arXiv:2305.13819）"低频扩散 + 高频确定性回归"快 100 倍；LWD（arXiv:2506.00433）小波能量图定位高细节区域 + 高频掩码监督实现 4K 生成。

### 4.4 频域引导编辑与加速

- **FreSca**（Rochester 等, arXiv:2504.02154）：发现 **CFG 差分项 Δεt 是最适合频域操纵的对象**，每步 FFT 分频独立缩放，免训练、在 SD3 上验证——最接近"频域 prompt-to-prompt"的现有工具；作者自述二值掩码是可改进点；
- **FreqCa**（arXiv:2510.08669）：FLUX 上低频缓存 + 高频 Hermite 插值加速——目前唯一明确在 FLUX 上做频域分析的工作（服务于加速而非编辑）。

### 4.5 关键结论

"低频=结构、高频=纹理/风格"的对应在 **latent 空间与 CFG 差分项上比在像素空间更稳定、更语义化**。面向 FLUX 的频域风格/结构解耦编辑**目前基本无人做**，与 §3.3 的空白互为印证。

---

## 5. 交叉空白与研究机会总览

综合四个方向，识别出以下高价值空白（详见各 Idea 文档）：

| # | 空白点 | 依据 | 对应 Idea |
|---|---|---|---|
| G1 | FLUX/SD3 上 zero-shot 注意力风格注入缺失，MM-DiT 层归因未做 | §3.3 | Idea 1 |
| G2 | 频域（相位/幅度、DWT 子带）解耦在 flow-matching 编辑上无人系统研究 | §4.5 | Idea 2 |
| G3 | VLM 区域定位与频域编辑强度调制尚未结合（CAMILA 范式 × FreSca 旋钮） | §2.2/§4.4 | Idea 3 |
| G4 | 编辑 reward 模型与频域保真约束的 RL 后训练闭环 | §2.2(3)/§4.4 | Idea 4 |
| G5 | 风格-内容 Pareto 前沿的三维调度（层×时间步×频段）在 FM 模型上空白 | §3.1/§4.1 | Idea 1/2 共用 |
| G6 | 可信风格评估体系（CSD 已证失效） | §3.4 | 可作为附贡献 |

---

## 参考文献（精选）

1. Esser et al. Scaling Rectified Flow Transformers (SD3). arXiv:2403.03206, ICML 2024.
2. Black Forest Labs. FLUX.1 Kontext. arXiv:2506.15742, 2025.
3. Qwen Team. Qwen-Image Technical Report. arXiv:2508.02324, 2025.
4. Liu et al. Step1X-Edit. arXiv:2504.17761, 2025.
5. Brooks et al. InstructPix2Pix. arXiv:2211.09800, CVPR 2023.
6. Zhang et al. ICEdit. arXiv:2504.20690, NeurIPS 2025.
7. Chung et al. StyleID. arXiv:2312.09008, CVPR 2024 Highlight.
8. Wang et al. InstantStyle. arXiv:2404.02733, 2024.
9. Hertz et al. StyleAligned. arXiv:2312.02133, CVPR 2024 Oral.
10. Rout et al. RF-Inversion. arXiv:2410.10792, ICLR 2025.
11. Wang et al. OmniStyle. arXiv:2505.14028, CVPR 2025.
12. Si et al. FreeU. arXiv:2309.11497, CVPR 2024.
13. FreSca. arXiv:2504.02154, 2025.
14. NeuralRemaster: Phase-Preserving Diffusion. arXiv:2512.05106, 2025.
15. Fourier Phase Diffusion for Style Transfer. IJCAI 2025 Oral.
16. Lipman et al. Flow Matching. arXiv:2210.02747, ICLR 2023.
17. Liu et al. Rectified Flow. arXiv:2209.03003, 2022.
18. Somepalli et al. CSD. arXiv:2404.01292, 2024.
19. FreqCa. arXiv:2510.08669, 2025.
20. CAMILA. arXiv:2509.19731, NeurIPS 2025.
21. EditReward. ICLR 2026.
22. Bagel. arXiv:2505.14683, 2025.

*注：部分 2025–2026 预印本未经同行评审；个别 arXiv 编号建议正式引用前复核（详见各 Idea 文档的"未确认项"）。*
