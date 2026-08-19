# MediX-R1 代码阅读四：Checkpoint、模型合并、评测流水线与当前源码风险

> 本文覆盖训练产物如何保存和恢复、FSDP 分片怎样合并、`eval/` 与 `eval2/` 怎样完成 Generate/Evaluate/Score、17 个医学任务怎样统一，以及当前本地快照中已经确认的阻断、缺陷和验证边界。文中不会读取或展示工作区 `config.py`、`.env` 中的任何密钥值。

## 系列导航

1. [完整训练链路与全部主要分支](./01-完整训练链路与全部主要分支.md)
2. [配置、数据协议与多模态模型适配](./02-配置数据协议与多模态模型适配.md)
3. [Ray、FSDP、vLLM、Reward 与强化学习算法](./03-分布式训练运行时与强化学习算法.md)
4. [Checkpoint、模型合并、评测流水线与源码风险](./04-Checkpoint模型合并评测与源码风险.md)

如果需要逐个文件、类和方法的快速索引，可同时查看 [两个项目源码文件类方法完整索引](../两个项目源码文件类方法完整索引.md)。本篇重点解释这些文件怎样组成“训练产物到评测结果”的交付链。

## 1. 训练结束不等于得到可推理模型

训练阶段保存的是 FSDP SPMD checkpoint：每个 rank 保存自己负责的 model shard、optimizer shard 和额外状态。普通 Transformers 或 vLLM 通常期待一个 Hugging Face 目录，其中包含完整权重、config、generation config、tokenizer/processor。

因此交付链是：

```text
RayPPOTrainer._save_checkpoint()
→ FSDPCheckpointManager.save_checkpoint()
→ rank 分片 + HF 配置目录 + DataLoader 状态
→ model_merger.py
→ actor/huggingface/*.safetensors
→ vLLM / Transformers 推理
→ eval/eval.py
```

## 2. Checkpoint 目录结构

一次全局 step 的典型结构：

```text
checkpoints/<experiment>/
├── checkpoint_tracker.json
└── global_step_<N>/
    ├── actor/
    │   ├── model_world_size_<W>_rank_0.pt
    │   ├── model_world_size_<W>_rank_1.pt
    │   ├── ...
    │   ├── optim_world_size_<W>_rank_0.pt
    │   ├── extra_state_world_size_<W>_rank_0.pt
    │   └── huggingface/
    │       ├── config.json
    │       ├── generation_config.json
    │       ├── tokenizer/processor files
    │       └── 合并后才会出现的权重文件
    ├── critic/                    # 仅 use_critic 时
    └── dataloader.pt
```

`actor/huggingface/` 在初次保存后并不一定有完整模型权重。rank 0 只保存 Hugging Face config、generation config 与 processing class，完整 state dict 仍在各 rank `.pt` 文件中。

## 3. `BaseCheckpointManager` 保存什么状态

抽象类列出的目标是：

```text
FSDP model state
optimizer state
lr_scheduler state
CPU/CUDA/NumPy/Python RNG state
Hugging Face config/tokenizer/processor
```

Driver 另存 `StatefulDataLoader.state_dict()`，所以恢复链试图覆盖模型、优化器、学习率、随机数与数据顺序。

### 3.1 RNG 状态

每个 rank 的 extra state 保存：

```python
{
    "cpu": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state(),
    "numpy": np.random.get_state(),
    "random": random.getstate(),
}
```

恢复时逐项写回。这比只保存 model/optimizer 更接近可复现训练，但 vLLM 内部 RNG、DataLoader 多进程和外部 Reward API 仍可能引入非确定性。

### 3.2 目录锁

`local_mkdir()` 在系统临时目录创建基于 checkpoint 路径 hash 的 FileLock，避免多个 rank 同时建目录。锁失败时只打印 warning，仍尝试 `os.makedirs(exist_ok=True)`，所以目录创建不会因锁服务异常完全停止。

## 4. `FSDPCheckpointManager.save_checkpoint()`

每个 rank 都调用 PyTorch distributed checkpoint state-dict API。两种模式：

| `save_model_only` | 保存内容 |
|---:|---|
| False | model shard、optimizer shard、scheduler 和 RNG |
| True | 只保存 model shard |

所有 rank 保存完后 barrier；rank 0 再写 Hugging Face config、generation config 和 processor/tokenizer；最后再次 barrier。

### 4.1 `save_model_only` 与恢复不对称

`load_checkpoint()` 无条件读取 model、optim 和 extra 三类文件。若 checkpoint 以 `save_model_only=True` 保存，后续直接用标准恢复路径会找不到 optimizer/extra 文件。当前代码没有“仅加载模型”的分支，这是一处明确的数据契约不对称。

## 5. Driver 级 `_save_checkpoint()`

`RayPPOTrainer` 除调用 Worker 保存 Actor/Critic 外，还负责：

1. 根据最近验证分数更新 `best_global_step`；
2. 按 `save_limit` 删除旧 checkpoint；
3. 保存 `dataloader.pt`；
4. 写 `checkpoint_tracker.json`。

Tracker 结构：

```json
{
  "best_global_step": 100,
  "best_val_reward_score": 0.42,
  "last_global_step": 120,
  "last_actor_path": "/abs/path/.../global_step_120/actor"
}
```

`find_latest_ckpt()` 实际主要使用 `last_global_step` 拼出目录，不直接使用 `last_actor_path`。

## 6. Checkpoint 清理策略

`remove_obsolete_ckpt()` 在保存新 checkpoint 前执行。`save_limit` 包含当前即将保存的 checkpoint，因此先把可保留旧 checkpoint 数设为 `save_limit-1`。如果历史 best step 在旧 checkpoint 列表中，会额外保留它。

删除使用 `shutil.rmtree(..., ignore_errors=True)`，外层 try/except 很难捕获多数删除失败；即使删除失败也不会阻止新 checkpoint 保存。对于磁盘紧张场景，建议额外监控实际目录大小，而不是只看日志。

## 7. `_load_checkpoint()`：自动恢复

优先级：

```text
trainer.load_checkpoint_path 显式路径
→ find_last_checkpoint=True 时读取 checkpoint_tracker.json
→ 无路径则从头训练
```

路径最后一级必须包含 `global_step_`。恢复后：

```text
解析 global_step
→ Actor load_checkpoint
→ 可选 Critic load_checkpoint
→ 读取 dataloader.pt
→ 恢复 StatefulDataLoader
```

如果没有 `dataloader.pt`，代码打印提示并从数据开头启动，但 global step、模型和 optimizer 已恢复。这会形成“参数续训、数据顺序重置”的非完全恢复。

## 8. 模型合并入口

### 8.1 `training/merge_model.sh`

这是模板脚本，实际命令仍是：

```bash
python3 scripts/model_merger.py --local_dir "path/to/model/actor"
```

必须手动改为某个 `global_step_<N>/actor`。

### 8.2 `eval2/merge.sh`

它固定指向：

```text
training/checkpoints/medix-r1_2b_dapo/global_step_600/actor
```

若 `huggingface/*.safetensors` 已存在则跳过，否则进入 training 目录执行 merger。当前工作区没有这个 checkpoint，所以脚本的路径检查前提不成立。

## 9. `model_merger.py` 的真实步骤

### 9.1 找 world size

它扫描 `local_dir`，寻找：

```text
model_world_size_<W>_rank_0.pt
```

从文件名提取 world size。找不到则 assert 失败。

### 9.2 读取 DeviceMesh 与 placement

加载 rank0 state dict，取字母排序后的第一个参数作为 pivot：

- 若是 `DTensor`，读取 `device_mesh` 与 `placements`；
- 若不是 DTensor，按单维 fsdp mesh 处理。

当前代码只允许 mesh 名为 `("fsdp",)` 或 `("ddp","fsdp")`。后面虽保留“FSDP+TP”分支，但前面的 assert 不接受含 `tp` 的 mesh，因此该分支在当前条件下不可达。

### 9.3 并行读取 shard

使用最多 32 个线程读取其余 rank 文件，写入共享 list。代码没有保存并检查 futures，因此线程内部的异常不会通过 `future.result()` 主动重新抛到主线程；后续通常会在访问空占位值时以更间接的错误暴露。

### 9.4 按 placement 合并

| Placement | 行为 |
|---|---|
| Replicate | 取第一份 Tensor |
| Shard(dim) | 沿 placement.dim 拼接 |
| Partial | 未实现 |

非 DTensor 参数直接沿 dim 0 `torch.cat()`。这假设普通 Tensor 在不同 rank 上代表分片；若某个普通参数实际是完整副本，直接拼接会产生错误 shape，因此需要用真实 checkpoint 验证参数类型。

### 9.5 创建 Hugging Face 模型

读取 `actor/huggingface/config`，根据 `architectures[0]` 选择：

```text
ForTokenClassification → AutoModelForTokenClassification
ForConditionalGeneration → AutoModelForImageTextToText
ForCausalLM → AutoModelForCausalLM
```

随后在 meta device 创建空模型，`to_empty(cpu)`，最后 `save_pretrained(hf_path,state_dict=merged)`。

### 9.6 可选上传 Hugging Face

传 `--hf_upload_path` 会创建公开 repo：

```python
api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
```

因此该选项不是“默认私有上传”。用户未明确要求时不应运行，更不能在未知模型权利、医疗数据合规或密钥状态下自动上传。

## 10. 模型合并的代码风险

| 风险 | 位置 | 影响 |
|---|---|---|
| 线程异常未显式收集 | shard 并行读取 | 根因可能延迟成后续模糊错误 |
| key 缺失后仍继续 | `except` 只打印 | 局部变量可能未定义或沿用错误值 |
| Partial placement 未实现 | `merge_by_placement()` | 某些分布式布局无法合并 |
| FSDP+TP 分支前后约束冲突 | mesh assert 与后续分支 | 代码看似支持，当前实际上不可达 |
| 统一转 bfloat16 | 读取 shard 时 `.bfloat16()` | 与原 checkpoint 精度不同，属有意压缩但应知晓 |
| 已有权重就跳过 | `eval2/merge.sh` | checkpoint 更新后可能误用旧合并结果 |

## 11. `eval/` 与 `eval2/` 的边界

| 目录 | Candidate | Judge | 数据差异 | 主要用途 |
|---|---|---|---|---|
| `eval/` | 默认 Hub 模型 `MBZUAI/MediX-R1-8B` | 本地 Qwen3-14B 或 OpenRouter | 全部任务用远程/HF 配置 | 通用公开评测 |
| `eval2/` | 固定本地 2B 合并模型 | 根 `config.py` API、也可本地/OpenRouter | `rad_vqa` 改成本地 Dataset 路径 | 本地 checkpoint 定制验证 |

两套 `eval.py` 和 `utils.py` 大部分是复制关系。后续修 bug 时若只改一套，会继续产生行为漂移。

## 12. 评测入口参数

`eval/eval.py` 支持：

```text
--tasks
--config
--output_dir
--num_workers
--generate
--evaluate
--score
--model
--eval_model
--tensor_parallel_size
--judge_server
```

`tasks` 可以是单任务、多个任务、`llm`、`vlm` 或 `all`。

### 12.1 布尔参数解析问题

`--generate/--evaluate/--score` 使用 `type=bool`。在 argparse 中，非空字符串通常都转为 True，所以：

```bash
--generate false
```

也可能得到 True。当前 Shell 脚本传 `true`，能进入目标阶段，但命令行无法可靠地用 `false` 关闭。更标准的方式是 `action="store_true"`/`BooleanOptionalAction` 或自定义字符串解析。

### 12.2 Tensor Parallel 参数类型边界

参数声明 `type=str, default=1`。命令行提供时得到字符串，可直接放入 subprocess command list；省略参数时默认仍是整数 1，`subprocess.Popen()` 的参数列表可能因非字符串项失败。官方脚本显式传 2 或 1，绕开了这一问题。

## 13. 任务选择和配置展开

`tasks.yaml` 分为 `llm` 与 `vlm`。入口把嵌套结构展平成：

```text
all_tasks: 全部任务名
configs_to_load[task]: 对应 YAML 配置
```

遇到 `all` 时加入全部并 break；遇到 `llm`/`vlm` 时加入分类下全部任务；未知任务只 warning。当前未做去重，例如同时传 `llm mmlu_anatomy` 会让同一任务出现两次，可能导致 total expected 与 ID 去重逻辑不一致。

## 14. 当前 17 个评测任务

### 14.1 11 个文本任务

```text
mmlu_clinical_knowledge
mmlu_college_biology
mmlu_college_medicine
mmlu_medical_genetics
mmlu_professional_medicine
mmlu_anatomy
medmcqa
medqa
usmle_sa
pubmedqa
mimic_cxr_report_summarization
```

### 14.2 6 个视觉任务

```text
slake_vqa
rad_vqa
path_vqa
pmc_vqa
pmc_vqa_hard
mimic_cxr_report_generation
```

`eval2/tasks.yaml` 唯一源码差异是 `rad_vqa.path=./datasets_local/vqa-rad`，而当前该目录不存在。

## 15. `load_dataset_with_params()`：统一不同数据集

输出统一为：

```python
{
    "id": "<task>_<index>",
    "image": PIL.Image or None,
    "question": str,
    "answer": str,
    "answer_idx": optional,
}
```

### 15.1 多项选择

YAML 可指定一个 choices 字段、多个 choices 列或固定 choices。代码统一用 A-J 标号拼到 question。若有 `answer_id`，通过 index 取正确选项文本；若有 `answer_column`，直接取答案。

### 15.2 PubMedQA 多字段

当 `question_column` 是 list，代码把多个字段分别加标题，再追加固定 choices；`answer` 也可以是多个列，最终拼成多行标准答案。

### 15.3 MIMIC 特殊处理

报告生成会把 question 固定为“Generate a detailed report based on the scan.”；报告摘要任务把原报告放进 question，并清空 image。

## 16. 评测数据下载的风险

### 16.1 SLAKE

代码检查 `./tmp/slake/imgs.zip`，但执行 `wget URL` 时没有 `-P`，下载文件通常落在当前工作目录；随后却执行 `unzip imgs.zip -d ./tmp/slake`。路径依赖当前目录且前后检查位置不一致，首次下载流程需要实测。

### 16.2 MIMIC/PhysioNet

代码把用户名和密码拼进 shell 命令：

```text
wget ... --user <username> --password <password> <url>
```

这可能通过进程列表、错误日志或 shell 工具暴露凭据。更安全的做法是使用受限配置文件、环境传递给子进程但避免命令行明文，或使用 Python HTTP 客户端的认证接口。本文没有读取 `.env` 或任何真实凭据。

MIMIC 下载条件还使用“目录文件数少于 100”作为是否继续下载的判断，与具体样本是否存在没有一一对应关系。

## 17. Phase 1：Generate

入口先把所有选定任务加载为 `all_samples`，然后循环：

```text
读取 output JSONL 已完成 ID
→ 计算 remaining
→ 未启动则启动 Candidate vLLM
→ 单进程或 ProcessPool 生成
→ 再读 completed IDs
→ 未完成则重试 missing
```

结果文件：

```text
results/<short_model_name>/<short_model_name>.jsonl
```

`convert_to_underscored()` 只保留模型路径最后一段，所以不同组织下同名模型会写到同一目录。

## 18. Candidate vLLM server

`eval/utils.py` 启动：

```text
vllm serve <model>
--tensor-parallel-size <N>
--max-model-len 512000
--max-num-seqs 256
--port 8005
--gpu-memory-utilization 0.95
--enable-chunked-prefill
```

`eval2` 将 max model len 改为 8192、显存利用率改为 0.90，并优先寻找当前 Python 环境中的 `vllm` 可执行文件。

通用评测的 512000 远大于训练脚本 8448 的 prompt+response 上限，也远大于 `generate()` 的 `max_tokens=2048`。它依赖 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 强行放开限制，可能显著增加 KV Cache 规划压力。

## 19. Server readiness 与关闭

`start_vllm_server()` 每 5 秒请求 `/v1/models`，没有总 timeout，也不检查子进程是否已经退出。如果 vLLM 因 OOM、模型不兼容或参数错误立即退出，等待循环可能一直打印点。

遇到非 Connection/Timeout 的 RequestException 时会 break，并返回 server process，即使模型未就绪。随后生成请求才会暴露连接问题。

`kill_server_process()` 使用 psutil 终止全部子进程，再等待 10 秒并 kill 未退出进程。异常时调用 `process.kill()`。

## 20. `generate()`：OpenAI-compatible 多模态请求

Candidate client 固定连接 `http://localhost:8005/v1/`，API key 使用占位值。可通过环境变量提供 system prompt；`FORCE_THINK=1` 时在问题后追加 `<think>`。

图像被转成 JPEG，再编码为 data URL 放入 `image_url`。这适合进程间传输统一请求，但大图会增加 Base64 编码、内存和 HTTP payload 开销。

生成参数固定：

```text
max_tokens=2048
temperature=0.0
top_p=1.0
timeout=180s
```

这与训练时 `max_response_length=4096,temperature=1.0` 不同。评测测的是确定性、最多 2048 token 的模型表现，不是训练 rollout 分布本身。

## 21. 并行生成与断点续跑

多 worker 时使用 `ProcessPoolExecutor`，FileLock 保护 JSONL append，能避免多个进程同时破坏文件。

但是 Driver 对完成的 futures 只做：

```python
for _ in as_completed(futures):
    pass
```

没有调用 `future.result()`，所以子进程异常不会在本轮直接抛出。随后 completed count 不足，外层 while 会重试 missing。如果错误是永久性的，可能形成无限重试。

断点续跑仅以 `id` 是否存在判断，不校验已有行的 response 是否为空、模型是否相同或配置是否变化。复用旧结果目录前应先确认实验 provenance。

## 22. Phase 2：Evaluate

读取 Candidate JSONL 后，对每条 response 调 LLM Judge。Judge 有三种后端：

| 后端 | `eval/` | `eval2/` |
|---|---:|---:|
| 本地 vLLM | 支持 | 支持 |
| OpenRouter | 支持 | 支持 |
| 根 `config.py` API | 不支持 | 默认支持 |

`eval2` 的 configapi 从 `LLM_CONFIG` 读取 base URL、API key 和 model name，但本文没有读取配置内容。

## 23. 思考文本清理

在构造 Judge Prompt 前，如果 Candidate 输出含 `</think>` 或 `</thinking>`，只保留最后一个闭标签之后的文本。这适用于模型把最终答案放在思考标签后。

如果模型使用 `<thinking>...</thinking><answer>...</answer>`，清理后仍保留 `<answer>` 标签；Judge 需要自行理解。若闭标签缺失，全部 reasoning 也会进入 Judge。

## 24. 三轮 Judge 不是三个独立样本

`_run_eval_rounds()` 连续执行 3 次。每轮后把 Judge 输出追加为 assistant message，再追加“Please reevaluate...”作为 user message。因此后两轮能看到前一轮判断，它们不是独立同分布投票。

普通任务最后按 3 个 0/1 分数多数投票；MIMIC 对三个 0-5 分数取平均。多轮可以促使自我修正，但也可能强化第一轮锚定，不能等同于三个独立 Judge。

## 25. Judge 输出解析

每轮：

```text
去掉 think/thinking 前缀
→ 找第一个 { 和最后一个 }
→ json.loads
→ 读取 out["score"]
```

没有 schema 校验、范围检查、解析 retry 或 fallback。Judge 输出 Markdown、多个 JSON、缺 key、字符串分数等都可能抛异常。并行模式又可能吞掉 future exception并无限重试。

## 26. 普通任务与 MIMIC 评分提示

### 26.1 普通任务

Judge 判断 Candidate 是否与标准答案完全或语义匹配，也允许通过选项字母表达正确选择，输出 0/1。

### 26.2 MIMIC

Judge 从临床准确性、完整性和相关性评分 0-5。三轮平均后，Score 阶段再除以每条最大分 5，把任务结果归一化到 0-1。

这是一种 LLM-as-a-Judge 指标，不是放射学专家共识、临床安全认证或自动事实核验。

## 27. Phase 3：Score

入口从 ID 去掉最后一个 `_index` 恢复 task 名称。普通任务累计 correct/incorrect，MIMIC 累计 score obtained/total。最后：

```text
普通任务 accuracy = correct / total
MIMIC normalized score = obtained / (5 * sample_count)
Overall = 各任务结果的算术平均
```

Overall 是任务宏平均，每个任务权重相同，不按样本数加权。

### 27.1 Overall 表格的单位 bug

个别任务的 `Accuracy` 列保存 0-1；构造 Overall 行时却把 `avg_score*100` 写进 `Accuracy`，同时 `Accuracy (%)` 也显示百分比。因此 raw `Accuracy` 列的 Overall 与其他行单位不一致。

### 27.2 重复执行使用 append

Score 文件用 `'a'` 打开。重复运行 score 会把新表追加到旧文件，而不是覆盖。结果归档时需要区分每次运行边界或先清理目标文件。

### 27.3 空结果

如果某个选定任务没有 eval result，普通 accuracy 分母为 0；当前没有跳过或显式提示该任务缺失。

## 28. `eval.sh` 的默认行为

通用脚本：

```text
source .env
CUDA_VISIBLE_DEVICES=0,1
Candidate=MBZUAI/MediX-R1-8B
任务=all
num_workers=128
Generate/Evaluate/Score 全开
TP=2
Judge=local
```

脚本还把 `OPENROUTER_API_KEY` 显式设为空字符串，所以即使外部环境已有 key，也会被覆盖；默认 Judge 是 local，不受影响，但改成 openrouter 前必须调整脚本。

128 个进程对小机器可能造成 CPU、内存、文件句柄和 API 压力。README 已建议按机器降低。

## 29. `eval2/eval.sh` 的默认行为

它固定：

```text
CUDA_VISIBLE_DEVICES=4
Candidate=本地 global_step_600/actor/huggingface
任务=rad_vqa
num_workers=1
TP=1
Judge=configapi
```

运行前检查目录中是否存在 `.safetensors`。当前本地没有 checkpoint 和 `eval2/datasets_local/vqa-rad`，所以脚本不能在当前快照直接完成。

脚本根据自身位置计算项目根目录，路径稳定性比依赖当前工作目录的通用脚本更好。

## 30. MMMU-Medical 独立评测

`eval/eval_mmmu_med.sh` 不走主 `eval.py`，而是调用外部 MedEvalKit。仓库提供 `Qwen3_VL_vllm.py` 适配器，负责：

```text
把 MedEvalKit messages 转成 Qwen3-VL messages
→ processor.apply_chat_template
→ process_vision_info
→ vLLM.generate
→ 去掉 think/thinking 内容
```

### 30.1 MMMU Shell 中的未定义变量

脚本传入：

```text
USE_LLM_JUDGE
GPT_MODEL
API_KEY
BASE_URL
```

但文件内没有定义。注释称 MMMU 不需要 Judge，但是否允许空值取决于外部 MedEvalKit 参数解析。运行前应明确设置或确认框架忽略，而不是假设注释等于执行保证。

### 30.2 适配器边界

当输入含 `messages` 时，适配器把每条 content 当纯文本；图片处理主要依赖另一种 `prompt/image/images` 输入结构。是否覆盖 MedEvalKit 当前 MMMU 样本格式，需要在实际克隆版本中验证。

## 31. Submission 打包

`submit.sh` 要求显式传模型名，把结果目录和可选 MMMU 目录打包为 zip。submission ID 由模型短名的固定 MD5 前 8 位生成，所以同一模型每次运行得到相同 ID，并非真正随机唯一。

脚本使用 Linux 风格 `md5sum`。macOS 默认通常提供 `md5` 而非 `md5sum`，跨平台执行可能需要 coreutils 或修改命令。本文未运行打包和上传。

## 32. 当前源码已确认的 P0 阻断

| 编号 | 位置 | 已确认事实 | 修复前影响 |
|---|---|---|---|
| P0-1 | `training/run_train.sh` | 调用不存在的 `medix-r1_8b_dapo.sh` | 总入口失败 |
| P0-2 | `trainer/main.py`,`ray_trainer.py` | `single_controller/ray` 整个实现缺失 | Trainer import 失败 |
| P0-3 | `workers/sharding_manager/__init__.py` | 导入 `fsdp_vllm.py`，实际只有时间戳文件 | FSDPWorker import 失败 |
| P0-4 | 2B 脚本数据路径 | `MediX-R1/data/medix-rl-data` 不存在 | 2B Dataset 加载失败 |

P0-2 不是简单改一个文件名：缺失的是 RayClassWithInitArgs、RayResourcePool、RayWorkerGroup、colocated Worker class 等一组实现。修复时应从匹配版本的 EasyR1/veRL 来源恢复，而不是临时写空壳绕过 import。

## 33. 当前源码已确认的 P1 运行风险

| 编号 | 位置 | 问题 |
|---|---|---|
| P1-1 | `FSDPWorker.init_model()` | Critic 传错 config 且 `self.optximizer` 拼写错误 |
| P1-2 | `compute_gae_advantage_return()` | 用多元素 Tensor 作为 Python `if` 条件 |
| P1-3 | `medical.py` | HTTP 无 timeout/retry，Judge 判定只检查是否包含 YES |
| P1-4 | `medical.py` | ground truth 强依赖 `>` 模态前缀，异常静默转 0 |
| P1-5 | `ray_trainer.py` | rollout prepare/release 缺少 try/finally |
| P1-6 | `FSDPCheckpointManager` | model-only 保存不能由标准 load 对称恢复 |
| P1-7 | `model_merger.py` | shard future 异常未检查、key 缺失后继续 |
| P1-8 | `eval.py` | `type=bool`、TP default 类型不一致 |
| P1-9 | `eval.py` | ProcessPool 异常不取 result，永久错误可能无限重试 |
| P1-10 | `start_vllm_server()` | 无启动总 timeout、不检查进程提前退出 |
| P1-11 | `score` | Overall raw Accuracy 单位错误，输出文件追加 |
| P1-12 | MIMIC 下载 | 凭据进入命令行，存在泄露风险 |

## 34. README 与当前代码不一致

| README/注释说法 | 当前本地事实 |
|---|---|
| 有 7B/8B DAPO/8B GRPO/30B DAPO 脚本 | 只有 2B DAPO 和 8B GSPO |
| 先运行 `vllm_serve.sh` | 文件不存在 |
| Reward 使用 `chat_with_vllm()` | 当前函数名是 `chat_with_llm()`，并从根 config 调 API |
| `run_train.sh` 可选择实际配置 | 当前固定目标不存在 |
| pip install 后可训练 | 即使依赖安装成功，本地 Ray Controller 源码仍缺失 |

文档意图可能来自另一个提交或发布包。修复时应选定一个版本基线，把 README、脚本和源码一起对齐。

## 35. 当前工作区与验证边界

截至 2026 年 8 月 18 日，本轮验证结果：

| 检查 | 结果 |
|---|---|
| `MediX-R1` 文件数 | 107 个 tracked/current 文件 |
| Python 文件 | 72 个 |
| Python AST | 72/72 通过 |
| Shell 文件 | 11 个 |
| `bash -n` | 11/11 通过 |
| 训练第三方依赖 | 当前工作区 Python 未安装 Ray、Torch、Transformers、vLLM 等 |
| 端到端 import | 受缺失本地模块和环境依赖阻断 |
| GPU 训练 | 未运行 |
| 模型合并 | 当前无 checkpoint，未运行 |
| 评测 | 当前无准备好的本地模型/数据，未运行 |

AST 与 Shell 语法通过只说明文件可被语法解析，不代表类型、import、外部数据、GPU、网络、Reward API 和分布式行为正确。

## 36. 模块职责总览

### 36.1 训练入口与算法

| 路径 | 责任 |
|---|---|
| `training/examples/*.sh` | 实验配置覆盖 |
| `training/examples/config.yaml` | 基础配置 |
| `training/verl/trainer/main.py` | Ray 启动与对象装配 |
| `training/verl/trainer/ray_trainer.py` | Driver 训练循环 |
| `training/verl/trainer/core_algos.py` | Advantage、Policy/Value loss、KL |
| `training/verl/trainer/metrics.py` | 训练指标 |

### 36.2 数据与分布式运行时

| 路径 | 责任 |
|---|---|
| `training/verl/utils/dataset.py` | Dataset、图像视频、Prompt 编码 |
| `training/verl/protocol.py` | DataProto 数据协议 |
| `training/verl/single_controller/base/` | Worker/WorkerGroup/dispatch 抽象 |
| `training/verl/workers/fsdp_workers.py` | FSDP Actor/Critic/Ref/vLLM 综合 Worker |
| `training/verl/workers/actor/` | Actor forward 与 update |
| `training/verl/workers/critic/` | Critic forward 与 update |
| `training/verl/workers/rollout/` | vLLM 生成 |
| `training/verl/workers/reward/` | Reward 动态加载与 tensor 化 |
| `training/verl/workers/sharding_manager/` | FSDP、TP、Ulysses 数据/权重切换 |

### 36.3 模型、Checkpoint 和日志

| 路径 | 责任 |
|---|---|
| `training/verl/models/` | Qwen-VL/FlashAttention patch |
| `training/verl/utils/checkpoint/` | FSDP 保存恢复 |
| `training/scripts/model_merger.py` | 合并 rank shards |
| `training/verl/utils/logger/` | 多后端指标与生成样本日志 |
| `training/verl/utils/flops_counter.py` | FLOPs/MFU 估算 |
| `training/verl/utils/seqlen_balancing.py` | 长度均衡与动态 batch |
| `training/verl/utils/ulysses.py` | Sequence Parallel primitives |

### 36.4 评测和提交

| 路径 | 责任 |
|---|---|
| `eval/eval.py` | 三阶段通用 CLI |
| `eval/utils.py` | Dataset 统一、Candidate、Judge、server、JSONL |
| `eval/tasks.yaml` | 17 个评测任务映射 |
| `eval/eval_mmmu_med.sh` | MMMU-Medical 外部入口 |
| `eval/MedEvalKit_ModelFile/...` | Qwen3-VL MedEvalKit 适配器 |
| `eval/submit.sh` | 结果压缩 |
| `eval2/` | 本地 2B + configapi 的定制副本 |

## 37. 建议的修复和验证顺序

如果下一步要让项目真正可运行，建议严格按依赖顺序，而不是一次改完：

1. **选定来源版本**：确认本仓库对应的 EasyR1/veRL commit；
2. **恢复 Ray Controller**：补齐完整 `single_controller/ray`，验证 import；
3. **恢复 FSDP-vLLM 正确文件名/版本**：避免只重命名但 API 不匹配；
4. **修正总启动脚本和 README**：选择当前实际存在的实验；
5. **写纯 CPU 配置/Reward 单测**：不启动模型先验证 dataclass 与 Reward schema；
6. **写 DataProto/算法 Tensor 单测**：GRPO、gspo_token、filtering、checkpoint tracker；
7. **单 GPU/小模型 smoke**：Dataset→rollout→Reward；
8. **两 GPU 一个 step**：验证 FSDP/vLLM 权重同步与 release；
9. **保存→恢复→合并**：验证 checkpoint 完整闭环；
10. **单任务 eval**：先 `rad_vqa` 少量样本，再扩大任务；
11. **安全整改**：MIMIC 凭据、上传默认公开、日志脱敏；
12. **最后再做完整医学训练和 leaderboard 评测**。

每一步都应保留独立证据，不能用“后面的脚本能跑”反推前面的边界完全正确。

## 38. 推荐的自动化测试清单

### 38.1 配置和数据

```text
CLI override 优先级
Reward file:function 解析
本地/HF/Parquet 路径
Prompt 占位符与图像数量
含/不含模态前缀的 ground truth
DataProto repeat/union/reorder 对齐
```

### 38.2 算法

```text
GRPO 同组均值为0
全组同分 Advantage 为0
RLOO leave-one-out 数值
PPO 非对称 clip
GSPO/gspo_token forward 与 gradient
Token/sequence loss averaging
GAE 多 batch mask
```

### 38.3 运行时

```text
Worker dispatch/collect
FSDP-vLLM 权重 key 映射
TP preprocess/postprocess 不重复样本
prepare/release 异常恢复
空 response Reward
非有限 grad 跳过策略
```

### 38.4 Checkpoint 和评测

```text
普通保存/恢复
model-only 加载策略
best/last tracker
分片缺失时 merger 快速失败
eval boolean parser
server 提前退出检测
ProcessPool exception 传播
Judge JSON schema/retry
Score 空任务和 Overall 单位
```

## 39. 最终心智模型

MediX-R1 的“模型成果”至少有四种不同形态：训练中的 FSDP Actor、生成时的 vLLM runtime、磁盘上的 rank-sharded checkpoint、合并后的 Hugging Face 模型。评测又引入 Candidate、Judge、JSONL 中间结果和聚合分数。排查问题时必须先问当前讨论的是哪种形态、数据经过了哪条转换，以及这个结论来自源码设计、静态检查、真实运行日志还是最终评测。

完成本系列后，你应该能从任意一个现象反向定位：训练进不去先查启动/import；Reward 异常先查 Prompt/ground truth/API；OOM 先分清 FSDP、vLLM 与 dynamic batch；恢复不一致先查 rank shard、RNG、DataLoader；vLLM 加载失败先确认是否已 merge；评测卡住先查 server process、JSONL missing IDs 和 ProcessPool 异常；分数异常再查 Judge 轮次、任务归一化和 Overall 汇总。
