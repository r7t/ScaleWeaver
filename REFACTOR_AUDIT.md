# ScaleWeaver 全仓库重构审计

> 状态：三阶段重构已执行；本文主体保留第一阶段扫描快照，实际处理结果见下节。
>
> 扫描日期：2026-09-13
>
> 目标：识别当前主流程、可重建产物、已失效/已退役代码、兼容层、实验工具与需要产品决策的功能，为后续删减和重构确定边界。

## 0. 决策与实施结果

用户确认的范围：

- U1 删除旧顺序前端及其历史兼容入口；
- U2 保留 MSCX 导入、节奏模仿、完整旋律重调律和稳定往返；
- U3 删除任意 12-EDO 全谱缩放退火器；
- U4 保留 raw、SF2 和 voicebank，删除 neural；
- U5 保留 `baseline.py`、`chord_progression_mscx.py`、`print_chord_progression.py`、`print_attack_ratios.py`、`create_missing_ngrams.py`，删除其余一次性研究脚本；
- U6 保留所有有效音阶和风格预设；
- U7 保留四声部与五声部；
- U8 保留人工和弦进行；
- U9 删除旧 Python API 兼容层，继续以 CLI、JSON 和 IR 作为外部边界；
- U10 保留生成结果、`.mscbackup/`、示例 MSCX 与 `piano.sf2`。

已经执行：

- 删除失效的 overlap/O1/O2 链和配置；
- 删除 SE0 预计算链，统一使用 `spectral_bundle.py`；
- 删除旧顺序前端、12-EDO 全谱缩放器、neural 音频分支和未保留研究脚本；
- 修复 `main.py` 对节奏模仿与完整旋律重调律前端的实际调度；
- 删除 best-of、cleanup 后处理、迭代再生成、旧函数别名和旧配置字段兼容；
- 把所有保留风格迁移到 quartic CSE 多项式、`presence_scale` 与当前退火结构；
- 删除根目录孤立的旧 `tiangan_72.style.json`，保留 `scales/` 中有效预设；
- 清理主入口参数、模块依赖和无调用代码，更新 README 与专题文档。

刻意保留：

- 根目录 `CSE_cache/`。其中现行统一光谱与 JI 缓存重建昂贵，继续供一键流程复用；
- 已删除重复的 `scales/cse_cache/`（约 829 MB），并清除根缓存中的 overlap、SE0 和中断临时文件；
- `output/`、`.mscbackup/`、`always-with-me.mscx`、`tiangan-imitation.ir.json` 和 `piano.sf2`；
- 所有有效四声部、五声部音阶/规则/风格与实验风格变体。

最终验证：

- 41 个当前 Python 文件全部通过 `py_compile`；
- 35 项单元测试全部通过；
- 30 份保留的 style 均与对应四声部或五声部 rules 完整加载；
- MelodyPlan 五声部一小节 smoke test 生成 Score 与 attack analysis 成功；
- 节奏模仿和完整旋律重调律均通过 `main.generate_score()` 五声部端到端测试；
- `always-with-me.mscx` 的 IR 与 MSCX 两个 fixed-point 检查均通过；
- 相对当前 Git 基线，已跟踪文件合计净减少约 6,069 行。

## 1. 审计结论

当前工作区约 **8.6 GB**，其中绝大多数不是源码：

| 类别 | 规模 | 结论 |
| --- | ---: | --- |
| `CSE_cache/` | 5.5 GB | 可重建缓存，未被 Git 跟踪 |
| `scales/cse_cache/` | 829 MB | 可重建的第二份缓存，未被 Git 跟踪 |
| `output/` | 1.7 GB | 生成结果，未被 Git 跟踪；可能有人工挑选成果 |
| `piano.sf2` | 563 MB | 外部音源，未被 Git 跟踪；默认 WAV 流程使用 |
| Python 源码 | 54 个文件，约 15,000 行 | 主流程、兼容层和实验工具混在根目录 |
| 音阶/规则/风格数据 | 约 89 个文件 | 17 套基础音阶与大量天干风格变体 |
| 文档 | 9 个 Markdown | 当前说明与历史补丁记录混杂 |

代码层面存在四类明确历史包袱：

1. **已退役的 SE0 预计算链**仍保留 6 个模块/入口，并通过 `scale_config.py` 维持旧字段校验。
2. **已失效的 O1/O2 overlap 实验链**仍包含 Python、C++、预编译二进制、样式和 manifest，但当前入口无法导入，样式也无法通过配置校验。
3. **旧顺序前端和旧一体化生成器**仍通过 `--legacy-frontend`、动态 import 和兼容 API 连接到主流程。
4. **大量独立研究/诊断/音频功能**没有被默认作曲链调用，但是否删除取决于项目未来定位。

建议先把项目目标收敛为一条明确主链：

```text
piano.py（可选包装）
  ├─ main.py
  │   ├─ joint_frontend_generate.py + melody_plan.py
  │   ├─ frontend_ir.py
  │   └─ accompaniment_generate.py + harmony_annealing.py
  ├─ mscx_export.py
  └─ wavgen8.py → sf2_synth.py
```

统一保留：

- 三层配置：`ScaleDefinition/1`、`CompositionRules/1`、`CompositionStyle/1`；
- MelodyPlan 联合前端；
- 前端 IR 与完整 Score JSON；
- 统一 SE/CSE/CCSE 光谱模型；
- 全曲退火；
- MSCX 导出；
- 用户选择的音频、导入/模仿和诊断功能。

## 2. 扫描边界与风险

仓库只有一个提交 `e96dae1 Initial public release of ScaleWeaver`，当前工作树包含大量未提交修改和未跟踪的新功能。因此 Git 历史不能可靠说明哪些新文件是临时实验，审计主要依据：

- 默认入口 `piano.py`、`main.py` 的实际调用；
- Python import 依赖图；
- CLI `--help` 和模块注释；
- 配置加载器的实际校验；
- 全仓库引用搜索；
- 现有 35 项单元测试；
- 生成目录的 Git ignore 状态和磁盘占用。

当前未提交代码属于用户工作，本阶段没有清理或回退。

扫描过程中还发现 `make_salience_presets.py` 在模块顶层直接写文件，没有 `main()` 保护。执行 `python3 make_salience_presets.py --help` 仍会重写三个文件：

- `scales/tiangan_72_5_high5.style.json`
- `scales/tiangan_72_5_high7.style.json`
- `scales/tiangan_72_5_high57.style.json`

本次探测触发了这次确定性重新生成。这三个文件原本和现在都处于未跟踪状态；后续重构应先修复或删除该脚本，避免 import、测试发现或参数探测产生副作用。

## 3. 当前有效主流程

### 3.1 一键入口

`piano.py` 依次启动：

1. `main.py` 生成 `ScaleWeaverScore/1` JSON；
2. `wavgen8.py` 生成 WAV（可用 `--no-wav` 跳过）；
3. `mscx_export.py` 生成 MSCX 和调律文件。

注意：`piano.py` 当前在内部命令中硬编码了 `--chord-progression a2he0`。用户没有传参数时，它并不完全遵循 rules 中的自动和声。这不是历史兼容功能，而是主入口行为与配置模型不一致，重构时应改为“默认不覆盖配置，只有显式参数才覆盖”。

### 3.2 作曲入口

`main.py` 当前有效路径是：

1. `scale_config.load_scale()` 合并三层配置；
2. `spectral_bundle.ensure_bundle()` 读取或创建统一 SE/CSE/CCSE bundle；
3. `ji_ratio.ensure_loaded_table()` 读取或创建 JI 表；
4. `joint_frontend_generate.generate_frontend_ir()` 生成 MelodyPlan、Lead 和 harmony plan；
5. `accompaniment_generate.generate_from_ir()` 生成低声部；
6. `harmony_annealing` 执行全曲优化；
7. `main.save_score()` 写 Score JSON 和 attack 分析。

### 3.3 核心依赖

下列文件是当前主链直接或间接需要的，应保留并重构：

- `main.py`
- `scale_config.py`
- `joint_frontend_generate.py`
- `melody_plan.py`
- `frontend_ir.py`
- `accompaniment_generate.py`
- `score_builder.py`
- `harmony_rhythm.py`
- `harmony_annealing.py`
- `annealing_config.py`
- `harmonic_realization.py`
- `adaptive_cse_runtime.py`
- `spectral_model.py`
- `spectral_bundle.py`
- `csebundle.py`（名称可在重构时移除，但目前主链使用）
- `subset_cse_cache.py`
- `ji_ratio.py`
- `prime_salience.py`
- `ngram_table.py`
- `measure_timeline.py`

导出链至少需要：

- `mscx_export.py`
- `wavgen8.py` 和所选择的音频后端。

## 4. 可以直接删除或重建的内容

这里的“可以直接删除”表示不影响当前源码和配置语义；第三阶段仍应先记录目标，再执行删除。

### D1. Python/IDE 临时文件

| 路径 | 大小 | 依据 | 恢复方式 |
| --- | ---: | --- | --- |
| `__pycache__/` | 1.4 MB | Python 字节码缓存；已被 ignore | 下次运行自动生成 |
| `.idea/` | 12 KB | 本地 IDE 配置；已被 ignore | IDE 自动生成 |

建议直接删除。

### D2. 数值缓存

| 路径 | 大小 | 依据 | 影响 |
| --- | ---: | --- | --- |
| `CSE_cache/` | 5.5 GB | 全部被 ignore；由统一光谱、JI 和子集缓存构建器生成 | 删除后首次生成会耗时重建 |
| `scales/cse_cache/` | 829 MB | 被 ignore；与根缓存用途重复 | 删除后按需要重建 |

这两处可从代码仓库工作区直接清理。若用户重视首次运行速度，可以保留根目录 `CSE_cache/`，但 `scales/cse_cache/` 应统一迁移后删除，避免同类缓存存在两个位置。

### D3. 已失效的 O1/O2 overlap 实验链

建议整组删除：

- `overlap_bundle.py`（366 行）
- `overlap_precompute.cpp`（300 行）
- `overlap_precompute_native`（预编译二进制）
- `scales/tiangan_72.o1o2.style.json`
- `scales/tiangan_72.overlap_cse.style.json`
- `scales/tiangan_72_overlap_075dcc9e3a5be3bfb166a84b_manifest.json`
- `scales/tiangan_72_overlap_babccac14065d387f60b0872_manifest.json`
- 对应 `CSE_cache/*overlap*` 二进制缓存。

证据：

- `overlap_bundle.py` 导入已经不存在的 `scale_config.resolved_overlap_objective`，当前执行立即报 `ImportError`；
- 两份 overlap style 使用 `overlap_objective`，当前 `CompositionStyle/1` 校验会报 unknown field；
- 默认入口和核心模块没有引用 overlap bundle；
- 两份 manifest 只服务该失效链。

删除约 666 行 Python/C++ 和一批无效数据配置，不影响当前统一 SE/CSE/CCSE 主链。

### D4. 已退役的 SE0 链

建议整组删除：

- `se_model.py`（101 行）
- `se_precompute.py`（262 行）
- `sebundle.py`（239 行）
- `build_native_se_bundle.py`（40 行）
- `build_native_se_bundle_wide_v1.py`（8 行兼容入口）
- `tian_gan_se_precompute_wide_parallel_v1.py`（8 行兼容入口）
- `scales/ji_major_171_cse_native_manifest.json`（旧 native manifest，无当前代码引用）

同时从 `scale_config.py` 删除 `rules.se_parameters` 的兼容校验和警告。

证据：

- 代码和文档明确标记 `rules.se_parameters`、SE0 链为 retired；
- 当前作曲链使用 `spectral_model.py` 和 `spectral_bundle.py`；
- 现有 rules 文件没有使用 `se_parameters`；
- 两个 8 行脚本仅转发旧文件名；
- 当前主链引用 `se_model.py` 的唯一原因是校验一个已经被忽略的旧字段。

预计删除约 658 行 Python，并消除一条容易与统一 SE 指标混淆的旧实现。

### D5. 明确无效的神经人声兼容分支

`wavgen8.py` 声称支持 `--voice-backend neural`，但依赖的 `voice_synth.py` 不在仓库中。该分支当前必然在运行时失败。

若保留 `wavgen8.py`，建议直接删除：

- `_mix_legacy_neural_voice()`；
- `--voice-backend neural` 选项；
- `--sing-voice`、`--sing-rate`、`--sing-cache-dir` 等仅服务缺失模块的参数；
- 文档中的 Neural-TTS/Praat 说明。

这不会影响 raw、SF2 或 voicebank 后端。

### D6. 无效和空操作 CLI 参数

建议从 `main.py` 和 `piano.py` 删除：

- `--best-of`、`--best-of-workers`：只发警告，当前退火始终使用一个固定旋律；
- `--cleanup` / `--no-cleanup`：主生成函数对 `cleanup=True` 直接抛错，false 只是默认值；
- `--no-dedup`：解析后直接丢弃；
- `generate_score(..., **legacy)` 中的 `cse_iterative_optimize` 兼容检查；
- `deduplicate_unisons`、`best_of_workers` 等只服务旧调用者的参数。

影响是移除旧命令行/API 兼容，不改变任何当前可成功执行的结果。

### D7. 重复或失真的历史文档

建议在把仍有效的信息合并进 README 或专题文档后删除：

- `WHOLE_SCORE_ANNEALING.md`：按多轮补丁顺序累积，包含“解压覆盖工程”“上一版结果”等交付记录；
- `scales/README.md`：基本是旧根 README 副本，并引用当前不存在的 `tests/test_spectral.py` 和 `verification.json`；
- 根目录与 `scales/` 下两份 `ATTACK_CSE_SOFT_LIMITS.md`：内容不一致，根目录版本还是旧三键格式。

建议保留一份按当前代码重写的 attack 配置说明，删除另一份；其余历史过程放入 Git 历史，不留在主文档树。

## 5. 很可能可以删除，但需要确认

### P1. 旧顺序前端和旧一体化生成器

候选范围：

- `frontend_generate.py`（104 行）；
- `main.py --legacy-frontend` 和 `frontend_mode='legacy'`；
- `joint_frontend_generate.py` 中“MelodyPlan disabled 时回退 legacy”分支；
- `frontend_ir.build_ir(... generator='legacy_frontend')` 的旧默认；
- `score_builder.py:generate_once()`；
- 只被 `generate_once()` 使用的固定节奏反复重生成音高链；
- `harmony_annealing.initial_score()`、`generate()` 等旧一键兼容包装。

删除理由：

- README 和主入口已将 MelodyPlan 联合前端设为默认；
- 完整生成已经经过前端 IR；
- 旧路径制造了 `main ↔ frontend ↔ accompaniment ↔ harmony_annealing` 循环 import；
- `score_builder.py` 的 1,918 行中同时保留新旧两代生成器，是主要压缩点。

需要确认：

- 是否还需要用旧前端做 A/B 对照或复现历史 seed；
- 是否有仓库外脚本 import `score_builder.generate_once` 或 `harmony_annealing.generate`。

建议：如果项目不承担历史结果复现，整组删除。

### P2. MSCX 节奏模仿和完整旋律重调律

候选范围：

- `imitation_frontend.py`（516 行）
- `retune_melody.py`（294 行）
- `mscx_to_ir.py`（29 行）
- `ir_to_mscx.py`（62 行）
- `mscx_roundtrip.py`（57 行）
- `reference_ir_mscx.py`（181 行）
- `check_mscx_ir_roundtrip.py`（70 行）
- `always-with-me.mscx`
- `tiangan-imitation.ir.json`
- `.mscbackup/`

这组提供两种不同能力：

1. 只复制参考谱节奏、方向和动态小节，再重新生成目标音阶旋律；
2. 保留完整源旋律，以最小 RMS 映射到目标音阶。

当前一键集成已经断裂：`piano.py` 接受 `--imitate-rhythm`、`--imitate-mscx`、
`--imitate-reference` 和 `--retune-melody`，随后把这些参数转发给
`main.py`；但 `main.py` 的 CLI 并未定义对应参数，因此通过 `piano.py`
使用它们会报“不认识的参数”。独立的 `imitation_frontend.py`、
`retune_melody.py` 和转换脚本仍可单独运行。

完整组约 1,209 行。普通算法作曲与 Score → MSCX 导出不依赖它。需要确认未来是否把“参考谱模仿/重调律”视为核心产品能力。

建议：保留 `mscx_export.py`；如果不需要导入和模仿，则删除本组全部。

### P3. 任意 MSCX 全谱转音阶退火

候选：

- `mscx_scale_anneal.py`（891 行）

它将任意 12-EDO MSCX 的全部音符映射到目标音阶并独立退火，和主作曲器的“生成新作品”是另一条产品线。默认入口和测试不依赖它。

建议：如果项目只负责生成新音乐，删除；如果要做现有总谱的微分音改编，保留并移到独立子包。

### P4. 音频后端范围

当前音频层包含：

- `sf2_synth.py`（560 行）：`piano.py` 默认使用；
- `raw_synth.py`（277 行）：内置波形合成，也被 SF2 层复用鼓组和公共常量；
- `voicebank_synth.py`（457 行）：精确 F0 人声音库；
- `wavgen8.py`（300 行）：统一 CLI；
- `piano.sf2`（563 MB，忽略文件）。

可选方案：

- **SF2 完整模式**：保留 `wavgen8.py + sf2_synth.py`，从 `raw_synth.py` 抽出少量公共函数后删除其完整 raw 后端，删除 voicebank；
- **SF2 + raw**：保留器乐双后端，删除 voicebank；
- **全部保留**：保留精确 F0 人声，但仍删除已失效 neural 分支；
- **无音频模式**：删除全部音频代码和 `piano.sf2`，项目只输出 JSON/MSCX。

建议采用“SF2 + raw”，除非精确 F0 人声已经进入实际工作流。

### P5. 独立研究、数据准备和诊断 CLI

| 文件 | 行数 | 作用 | 默认主链依赖 |
| --- | ---: | --- | --- |
| `baseline.py` | 450 | Lead 条件下的 CSE 基线 | 无 |
| `search.py` | 569 | 硬编码 EDO 的 OR-Tools 音阶搜索 | 无；当前环境缺 OR-Tools |
| `build_subset_cse_cache.py` | 56 | 手动预建子集缓存 | 无；主程序可自动建立 |
| `create_missing_ngrams.py` | 133 | 批量生成默认 anti-ABAB n-gram | 无 |
| `make_salience_presets.py` | 33 | 生成 3 份风格预设；当前有导入即写文件问题 | 无 |
| `chord_progression_mscx.py` | 121 | 单独导出块状和弦谱 | 无 |
| `print_chord_progression.py` | 86 | 打印和弦进行 | 无 |
| `print_attack_ratios.py` | 121 | 打印 attack 比例 | 无 |

整组约 1,569 行。这些工具适合独立的研究工具目录，不适合继续与核心模块平铺。

建议：

- 若目标是最小可维护作曲器，删除整组；
- 若仍需要研究分析，保留 `baseline.py` 和两个 print 工具，移到 `tools/`；
- `build_subset_cse_cache.py` 可删，因为运行时会自动创建；
- `make_salience_presets.py` 应删或重写为纯 CLI；
- `search.py` 应拆成另一个项目或删除。

### P6. 音阶与风格预设

当前有 17 套可通过默认 sibling rules/style 校验的基础音阶：

- `harmonic8_270`
- `ji_major_171`
- `major_108`
- `major_12`
- `major_19`
- `major_22`
- `major_31`
- `major_571`
- `major_742`
- `minor_22`
- `pentatonic_12`
- `scale13_41`
- `septimal_meantone_31`
- `stair_171`
- `staircase_108`
- `tiangan_72`
- `zigzag_108`

另外有：

- `tiangan_72_5.*` 五声部规则/基础风格；
- 10 余份 `tiangan_72_5_high*` 实验风格；
- `tiangan_72_gui.*`；
- `ji_major_171.entropy_balance.style.json`；
- 根目录孤立的 `tiangan_72.style.json`，没有同目录 n-gram，且主入口不引用；
- `major_31.tuning.txt`，其余调律文件通常运行时生成。

基础音阶文件本身不增加多少代码，但大量风格变体使配置语义和测试矩阵膨胀。需要确认哪些音阶和天干风格属于正式支持范围。

建议至少删除根目录孤立 `tiangan_72.style.json`；其余按正式支持列表保留，实验预设移到 `examples/` 或删除。

### P7. 四声部与五声部双重支持

当前多个核心模块同时处理：

- 四声部：Bass、Inner、Counter、Lead；
- 五声部：Bass、Inner、Inner2、Counter、Lead。

五声部支持渗透配置校验、声部池、初始化、退火、attack cardinality 和导出。它不是一个能靠删除单文件移除的功能，但若只保留一种编制，可以显著压缩分支和配置。

当前 `piano.py` 默认使用 `tiangan_72_5.rules.json`，说明五声部至少曾是主要工作流。需要用户明确最终要四声部、五声部，还是两者都保留。

### P8. 人工和弦进行

人工进行支持：

- degree 串，如 `4536251`；
- 和弦代码串，如 `a2he0`；
- `bars_per_chord` 和不完整结尾；
- 专门的 `test_manual_progression.py`；
- MelodyPlan 中为人工进行保留的句法划分。

这是活跃且有测试的功能，但它增加了 `main.py`、`scale_config.py`、`harmony_rhythm.py`、`melody_plan.py` 和联合前端中的分支。若用户只使用自动和声，可以整组移除；否则应保留并修复 `piano.py` 的硬编码覆盖。

建议保留，除非明确只需要自动和声。

### P9. 外部 Python API 兼容

代码中有多处“保留旧函数名/参数以免破坏外部 import”的包装：

- `harmony_annealing.initial_score()`
- `harmony_annealing.generate()`
- `harmony_annealing.attack_diagnostics()`
- `main.generate_score(..., **legacy)`
- `print_attack_ratios.py` 的旧函数别名；
- `csebundle.py` 的旧 CSE 命名；
- `adaptive_cse_runtime.py` 的旧统计字段和函数名。

仓库内没有证据表明存在必须兼容的已发布 Python API。若项目只通过 CLI 使用，可以删除大部分包装并重新命名模块；若有外部脚本 import 这些函数，需要给出调用清单后逐步迁移。

建议将“CLI 是唯一稳定接口”设为重构原则。

## 6. 不应直接删除的内容

以下内容目前看似复杂，但属于主算法，不应在没有替代实现和回归测试时直接删除：

- `adaptive_cse_runtime.py`：虽然带有大量 v9/compatibility 注释，仍是当前评分链核心；
- `score_builder.py`：包含大量旧代码，但当前联合前端和伴奏初始化仍调用其中多个内部函数，必须按函数依赖拆除，不能整文件删除；
- `harmony_rhythm.py`：全局运行时状态较重，但当前和弦、节奏和音阶上下文依赖它；
- `harmony_annealing.py`：兼容包装可删，Energy 与优化器需要保留；
- `csebundle.py`：名称有历史感，但当前主流程直接引用；应先把有效接口并入 `spectral_bundle.py` 再删除文件；
- `subset_cse_cache.py`：主退火器会自动调用；
- `ji_ratio.py` 和 `prime_salience.py`：当前 attack 分析和优化器调用；
- `measure_timeline.py`：动态小节、导入和 MSCX 导出共同使用；
- 35 项现有单元测试：应在删减过程中作为回归基线，只删除与明确取消功能绑定的测试。

## 7. 结构性技术债

这些内容不一定是“功能”，但应在第三阶段重构。

### 7.1 循环 import

当前存在：

```text
main
 ├─ joint_frontend_generate ──► main（CLI 配置）
 ├─ accompaniment_generate ──► main（分析函数）
 └─ harmony_annealing
      ├─► joint_frontend_generate（兼容包装）
      └─► accompaniment_generate（兼容包装）
```

建议：

- 新建无副作用的 `runtime.py` 或 `composition_context.py`，承接配置和缓存初始化；
- 将 attack/JI 报告从 `main.py` 移到 `analysis.py`；
- 删除 `harmony_annealing` 的旧一键包装；
- 让 CLI 依赖库模块，库模块不反向 import CLI。

### 7.2 巨型模块职责过多

| 模块 | 行数 | 当前混合职责 |
| --- | ---: | --- |
| `score_builder.py` | 1,918 | 配置全局、旋律、旧前端、伴奏、清理、诊断、Score 构建 |
| `adaptive_cse_runtime.py` | 1,355 | bundle、背景场、attack 门控、统计、兼容 API |
| `harmony_rhythm.py` | 1,216 | 音阶上下文、和弦、和声计划、节奏、声部池 |
| `main.py` | 954 | 运行时配置、JI 分析、统计、生成编排、CLI |
| `mscx_export.py` | 870 | 记谱映射、时值拆分、布局和 XML 拼接 |

建议先删功能，再按稳定边界拆模块；不要在保留全部历史分支的情况下机械拆文件，否则文件数会上升但复杂度不降。

### 7.3 全局可变状态

`main._configure_adaptive_scale()` 会同时修改 `harmony_rhythm`、`score_builder`、`adaptive_cse_runtime` 的模块全局变量和缓存。测试中的“manual → auto 不泄漏状态”已经说明这是现实风险。

建议最终引入一个明确的 `CompositionContext`，把 scale、bundle、JI table、chords、voice pools 和配置作为对象传递。这个改动影响面大，应在删除旧路径后进行。

### 7.4 配置兼容层累积

`scale_config.py`、`annealing_config.py` 同时接受或警告多个旧字段，部分字段已经不影响结果。建议：

1. 确定唯一当前 schema；
2. 对仓库内配置做一次迁移；
3. 删除旧字段兼容；
4. 增加一个“所有正式配置均可加载”的测试；
5. 以后通过 schema 版本升级处理变化。

### 7.5 根目录平铺

54 个 Python 文件全部位于根目录，主代码、CLI、测试、生成器和实验混在一起。完成删减后建议整理为：

```text
scaleweaver/
  config.py
  context.py
  frontend/
  composition/
  spectral/
  export/
cli/
tests/
scales/
docs/
tools/        # 仅放明确保留的研究工具
```

这一步涉及 import 全面调整，应放在功能删减和测试补齐之后。

## 8. 建议的第三阶段顺序

在用户确认不确定项后，按以下顺序实施：

1. 建立“当前正式功能”回归基线：现有 35 项测试、短作品生成、JSON → MSCX、固定 seed 摘要。
2. 删除可重建临时文件和确认不保留的生成产物。
3. 删除 broken overlap 链。
4. 删除 retired SE0 链和旧配置字段。
5. 删除无效 CLI 参数与 neural 音频分支。
6. 按确认结果删除旧前端、MSCX 导入、音频后端、研究工具和预设。
7. 清理失效测试与文档，保留用户选择功能的回归测试。
8. 拆解循环 import 和巨型模块，引入明确的 composition context。
9. 整理包结构、统一 CLI 和配置路径。
10. 运行完整测试及至少一个正式配置的端到端短生成，更新 README。

每一步单独验证，避免一次大改后无法定位行为差异。

## 9. 需要用户确认的删除范围

请对以下编号给出“保留”或“删除”；也可以直接选择后面的推荐档位。

| 编号 | 功能组 | 建议 |
| --- | --- | --- |
| U1 | 旧顺序前端、旧一体化生成器、历史 seed 复现 | 删除 |
| U2 | MSCX 导入、节奏模仿、完整旋律重调律、round-trip | 保留（若常用参考谱创作） |
| U3 | 任意 12-EDO MSCX 全谱转音阶退火 | 删除或独立项目 |
| U4 | 音频后端 | 保留 SF2 + raw，删除 voicebank/neural |
| U5 | baseline、音阶搜索、配置生成、诊断等独立 CLI | 只保留两个 print 诊断工具 |
| U6 | 音阶与风格预设 | 保留 17 套基础音阶；删除/移动实验风格 |
| U7 | 声部编制 | 保留四声部和五声部，或只选一种 |
| U8 | 人工和弦进行 | 保留 |
| U9 | 仓库外 Python import 兼容 | 删除，规定 CLI/JSON/IR 才是稳定接口 |
| U10 | `output/`、`.mscbackup/`、示例 MSCX/IR | 需要确认哪些是需要保存的作品 |

推荐档位：

- **保守瘦身**：执行 D1–D7；删除 U1、U3、U9；保留其余功能。风险最低，能清掉失效代码和兼容噪音。
- **聚焦作曲器（推荐）**：保留 MelodyPlan、四/五声部、自动/人工和声、JSON、MSCX 导出、SF2 + raw；保留 U2；删除 U1、U3、voicebank、研究工具、实验预设和 Python API 兼容。
- **最小核心**：只保留一套指定音阶、一种声部编制、MelodyPlan、统一光谱、JSON 和 MSCX 导出；删除所有导入、音频、研究和兼容功能。

## 10. 当前验证状态

第一阶段扫描前后：

- `python3 -m unittest discover -v`：35 项通过；
- `python3 check_mscx_ir_roundtrip.py always-with-me.mscx`：IR 与 MSCX 固定点均通过；
- 17 套基础音阶的 sibling rules/style 均可加载；
- 两份 overlap style 加载失败，确认是失效配置；
- `overlap_bundle.py --help` 立即 ImportError；
- `piano.py` 的模仿/重调律参数会被转发给不支持它们的 `main.py` CLI；
- `search.py` 需要未安装的 OR-Tools，且不属于主依赖；
- 本阶段没有删除或重构生产代码。
