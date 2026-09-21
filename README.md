# ScaleWeaver

ScaleWeaver 是一套面向微分音音阶、精确 EDO（Equal Division of the Octave）和纯律音响分析的算法作曲工具。它读取音阶、规则与风格三层 JSON 配置，生成四声部或五声部作品，并可输出 JSON、MuseScore 3 的 MSCX 乐谱、调律表和 WAV 音频。

当前代码使用一条统一流程：MelodyPlan 联合前端负责 Lead、节奏与和声；伴奏后端生成其余声部，并在 Lead 与节奏固定的条件下执行全曲模拟退火。项目还支持从 MSCX 借用节奏、完整旋律重调律，以及 MSCX/IR 的稳定往返转换。

## 功能概览

- 任意精确 EDO 中的音阶子集，不把微分音重新量化到十二平均律；
- 四声部 `Bass / Inner / Counter / Lead` 与五声部 `Bass / Inner / Inner2 / Counter / Lead`；
- MelodyPlan/2 段落结构、旋律材料依赖、节奏和和声的联合生成；
- 人工和弦循环与配置驱动的自动和声；
- SE、CSE、CCSE 统一光谱模型与按音阶缓存的原生数值表；
- 固定 Lead、固定节奏的全曲和声退火；
- 狼音、声部交叉、音域、最大跳进和特殊距离约束；
- MSCX 节奏模仿、完整旋律重调律、MSCX/IR 双向转换；
- 原始加法合成、FluidSynth/SF2 和精确 F0 voicebank；
- attack 比例、和弦进行、CSE baseline 等诊断工具。

## 环境要求

核心代码要求 Python 3.9 或更新版本，并依赖 NumPy：

```bash
python3 -m pip install numpy
```

只生成 JSON 与 MSCX 不需要其他 Python 包。SF2 渲染需要系统安装 FluidSynth 动态库；项目根目录的 `piano.sf2` 是默认音源。若动态库不在系统搜索路径中，可通过 `wavgen8.py --fluidsynth-lib PATH` 指定。

大音阶第一次运行时可能需要生成数 GB 的光谱与 JI 缓存。构建速度取决于音阶音数、声部音域、CPU 核数和磁盘速度。

## 快速开始

### 一键生成

`piano.py` 依次生成完整 Score JSON、WAV 和 MSCX。默认配置是：

```text
scales/tiangan_72.json
scales/tiangan_72_5.rules.json
scales/tiangan_72_5_norm.style.json
```

运行：

```bash
python3 piano.py 17171
```

种子可以是整数或任意文本。文本种子会稳定映射成整数，并转换成安全的文件名：

```bash
python3 piano.py always-with-me
```

默认输出目录是 `output/`，缓存目录是 `CSE_cache/`。一次运行通常会产生：

```text
output/<seed>.json
output/<seed>_attacks.json
output/<seed>.wav
output/<seed>.mscx
output/<音阶名>.txt
```

常用示例：

```bash
# 跳过音频渲染
python3 piano.py 17171 --no-wav

# 指定配置
python3 piano.py 666 \
  --scale scales/major_31.json \
  --rules scales/major_31.rules.json \
  --style scales/major_31.style.json \
  --bars 14 --no-wav
```

`piano.py` 默认生成 40 小节；`main.py` 默认生成 48 小节。完整参数可运行 `python3 piano.py --help` 和 `python3 main.py --help` 查看。

### 只生成 Score JSON

```bash
python3 main.py \
  --scale scales/tiangan_72.json \
  --rules scales/tiangan_72.rules.json \
  --style scales/tiangan_72.style.json \
  --seed 666 --bars 48 \
  --cse-dir CSE_cache \
  --output output/666.json
```

如果省略 `--rules` 和 `--style`，加载器会查找与定义文件同名的 `.rules.json` 与 `.style.json`。

## 生成流程

```text
ScaleDefinition/1 + CompositionRules/1 + CompositionStyle/1
                              |
                              v
       MelodyPlan/2 + Lead/节奏/和声联合生成
                              |
                              v
              ScaleWeaverFrontEndIR/1
                              |
                              v
        Bass / Inner / [Inner2] / Counter 初始化
                              |
                              v
          固定 Lead 与节奏的全曲模拟退火
                              |
                              v
                  ScaleWeaverScore/1
                     |             |
                     v             v
               MSCX + 调律表      WAV
```

前端 IR 是流程的稳定边界。它包含逐小节时间轴、Lead 事件、Lead 节奏骨架和完整和声计划。伴奏阶段不得修改 Lead；退火阶段只能改变伴奏音高，不能改变任何起音位置或时值。

绝对时间统一以四分音符为一拍。`measure_map` 是权威时间轴，允许不同小节使用不同拍号和实际时长。

## 三层配置

一套常规配置由三个文件组成：

```text
scales/<name>.json          ScaleDefinition/1
scales/<name>.rules.json    CompositionRules/1
scales/<name>.style.json    CompositionStyle/1
```

### ScaleDefinition/1

定义调律身份：

- `id`、`name`：稳定 ID 与显示名称；
- `edo`：一倍频程的等分数；
- `pcs`：升序、从 0 开始的音阶 pitch class；
- `names`：与 `pcs` 一一对应的音名；
- `base_note`、`base_freq_hz`：step 0 的名称和频率。

绝对 step 的频率始终是 `base_freq_hz * 2 ** (step / edo)`。

### CompositionRules/1

规则文件描述结构与硬约束，主要字段包括：

- `voice_ranges`：各声部的绝对 step 范围；声明 `inner2` 即启用五声部；
- `wolf_intervals`、`wolf_absolute_intervals`：八度等价或绝对距离的硬狼音；
- `cleanup.degree_distances`、`cleanup.step_distances`、`cleanup.soft_penalty`：生成期间的特殊距离软惩罚；
- `melody`：旋律轮廓、音区和进行参数；
- `harmony`：功能分区、稳定度、和弦定义和人工和弦循环；
- `metadata.generator`：节奏密度、跳进平衡、声部音域中心等生成参数；
- `runtime`、`voicebank`、`spectral_parameters`：运行上限、歌声映射和光谱模型。

### CompositionStyle/1

风格文件控制软目标与优化权重：

- `prime_rewards`：2、3、5、7、11、13、17 素数色彩奖惩；
- `ngram_file`：Lead n-gram 偏置表；
- `lead_pc_target_distribution`：Lead pitch class 目标分布；
- `voice_pc_rewards`：伴奏各声部的 pitch class 奖惩；
- `cse_weights`：SE/CSE/CCSE 混合与生成期光谱代价；
- `attack_cse_percentile_limits`：同时起音组合的软百分位限制；
- `melody_plan`：MelodyPlan 结构和联合生成参数；
- `annealing`：全曲目标、初始化、搜索步数和温度。

当前 CSE 损失多项式统一使用 `a4` 至 `a0`；素数显著性缩放字段为 `prime_salience.presence_scale`。加载器会拒绝已经移除的旧字段。

## 四声部与五声部

常规声部顺序是 `bass < inner < counter < lead`。规则文件包含 `voice_ranges.inner2` 时，顺序变为 `bass < inner < inner2 < counter < lead`。

默认五声部配置是 `tiangan_72_5.rules.json`。`tiangan_72_5_norm.style.json` 及 `tiangan_72_5_high*.style.json` 是保留的五声部风格变体。

天干日常风格使用 `annealing.bass_balance` 对超过25%总时值的低音音级施加软惩罚，缓解甲、乙过度集中，保留和声及终止式偏好。该项对 legacy 和 three-tone 都有效，权重设为0可关闭。原因分析、配置和受控比较见 [低音分布均衡](BASS_BALANCE.md)。

## MelodyPlan 与人工和弦进行

默认前端使用 `MelodyPlan/2`。它先决定材料之间的依赖，再由 `joint_frontend_generate.py` 联合选择和声片段、Lead 节奏与 Lead 音高。48 小节作品可使用多种六段式结构；较短作品按递归的 8、4、2、1 小节层级建立材料关系。

天干日常风格已吸收手写旋律的小范围展开与方向变化偏好，以及隔两小节的节奏呼应关系。保留十六分音符和复杂节奏库，仅轻微简化整体节奏，并提高3+1、降低1+3及略降1+2+1的相对权重。统计、作用范围与配置说明见 [手写旋律特点](HANDWRITTEN_MELODY_CHARACTER.md)。第25–28小节的欢快感分析及可泛化的节奏、轮廓、音场与低限情感特征方案见 [天干音阶情感因素建模](TIANGAN_EMOTION_MODEL.md)。

`--chord-progression` 可覆盖规则文件中的和弦进行：

自然音阶规则默认不预置固定和弦循环，生成时使用自动和声。需要固定进行时再通过该参数显式指定。

Meantone 七声音阶（当前为 `major_12`、`major_19`、`major_31`）的自动进行先按种子选择完整的八小节级数路线，每小节一个和弦。路线增加 ii、iii、vi，包含 ii–V–I，并避免 V→ii、V→IV 等反功能连接。短乐句尽可能以 ii–V–I 收束。联合前端沿用整条路线；材料复现中的和声复制仅在与规划一致时保留，否则保留可用的旋律、节奏复现层。CSE 继续优化旋律和实际配器，实际发声仍可能省略小三和弦的三度。

此路径按等五度链结构识别，排除 `ji_major_171` 的二级狼和弦，也不用于当前两份非 meantone 的 22-EDO 七声音阶。显式人工进行优先于自动路线。

可用 `--harmony-model three-tone` 试用按和弦局部比例的“3音”进行。默认仍使用原有逻辑；`--harmony-model legacy` 可明确恢复原模式。新模式先选“3音”组，再选组内和弦，借鉴主功能延长、下属准备、五度连接、大小调色彩与终止回归；天干继续使用自身音高和 32 和弦词汇。固定人工进行仍优先，导入前端 IR 时不能用此参数改写其中的和声。

```bash
python3 main.py --scale scales/tiangan_72.json --cse-dir CSE_cache \
  --harmony-model three-tone --seed 42 --bars 8 \
  --save-frontend-ir output/tiangan_three_tone.ir.json \
  --output output/tiangan_three_tone.json
```

3音引擎 v2 始终保留 val 推导的 EDO 音级，分为“和弦内有（present）／仅音阶内有（implied）／音阶外（outside_scale）”三类。辛甲己的3音为第19步，参与正常功能组与解决评分。规则可设置 `harmony.three_tone_progression` 为 `{"enabled": true, "max_integer": 64, "outside_scale_weight": 0.25, "implied_weight": 0.8, "resolution_weight": 1.2}`。复杂度限制只影响自动生成资格，不删除分类。旧 `special_weight` 接受为音阶外组相对权重的别名。每个片段导出类别、比例及来源、功能亲和度、主功能强度、张力、乐句角色与旧功能标签。缺少真实比例时明确报告数据问题。音乐性权重仍需试听调整，详见 [分类与接入说明](THREE_TONE_HARMONY_PLAN.md)。音阶级3音场、15音场、天干甲—辛双中心与以后代码改造所需的数据结构另见 [3音场与15音场理论备忘录](THREE_FIFTEEN_TONE_THEORY.md)。

```bash
python3 main.py --scale scales/major_31.json \
  --seed 42 --bars 14 --chord-progression 4536251 \
  --output output/manual.json
```

规则定义了单字符和弦代码时，可以输入连续代码或用逗号、空格分隔。纯数字串按音阶 degree 读取。`auto` 表示恢复自动和声。

人工和弦循环的完整周期构成一个乐句，请求的小节数必须是该周期的整数倍。人工模式不会用自动终止式替换最后一个和弦。详细行为见 [MANUAL_PROGRESSION_PHRASES.md](MANUAL_PROGRESSION_PHRASES.md)。

## 两阶段工作流

只生成 MelodyPlan 与前端 IR：

```bash
python3 joint_frontend_generate.py \
  --scale scales/tiangan_72.json \
  --rules scales/tiangan_72.rules.json \
  --style scales/tiangan_72.style.json \
  --seed 666 --bars 48 \
  --plan-output output/666.plan.json \
  --output output/666.frontend.json
```

从现有前端 IR 生成伴奏：

```bash
python3 accompaniment_generate.py \
  --input output/666.frontend.json \
  --scale scales/tiangan_72.json \
  --rules scales/tiangan_72.rules.json \
  --style scales/tiangan_72.style.json \
  --cse-dir CSE_cache \
  --output output/666.json
```

`main.py` 也能在一步生成时保存 IR，或直接读取已有 IR：

```bash
python3 main.py --scale scales/tiangan_72.json --seed 666 \
  --save-frontend-ir output/666.frontend.json \
  --output output/666.json

python3 main.py --scale scales/tiangan_72.json \
  --frontend-ir output/666.frontend.json \
  --output output/666-reharmonized.json
```

读取 IR 时，IR 内的和声计划是权威数据，不能再同时传入 `--chord-progression`。

## MSCX 节奏模仿

节奏模仿只复制参考谱的边界、拍号、休止、起音和时值。参考音高、音程和旋律方向不会进入目标作品；目标音高由当前音阶、和声与旋律代价重新生成。

```bash
python3 main.py \
  --scale scales/tiangan_72.json \
  --rules scales/tiangan_72_5.rules.json \
  --style scales/tiangan_72_5_norm.style.json \
  --imitate-rhythm always-with-me.mscx \
  --imitation-staff-id 1 \
  --output output/imitation.json
```

`--imitate-mscx` 与 `--imitate-reference` 是同一参数的别名。参考源也可以是 `ScaleWeaverMSCXReferenceIR/1` JSON。

只生成模仿前端 IR：

```bash
python3 imitation_frontend.py always-with-me.mscx \
  --scale scales/tiangan_72.json \
  --rules scales/tiangan_72_5.rules.json \
  --style scales/tiangan_72_5_norm.style.json \
  --output output/imitation.frontend.json
```

## 完整旋律重调律

重调律模式保留源旋律的事件次序、节奏和相对轮廓，并把推断出的源音级映射到目标音阶，使总体音分误差最小。它与节奏模仿是互斥的输入模式。

```bash
python3 main.py \
  --scale scales/major_31.json \
  --retune-melody always-with-me.mscx \
  --retune-staff-id 1 \
  --output output/retuned.json
```

也可直接使用 `retune_melody.py`；该入口会完成前端重调律和伴奏生成。

自动和声现在读取重调律后的固定旋律：首拍、次重拍、其余拍点、弱位的起音权重依次为4、3、1.5、1，并乘以实际重叠时长的平方根。比较旋律入和弦比例及边际 CSE（同一配器音域内 `CSE(和弦+旋律音)-CSE(和弦)`），再结合较弱的和弦连接先验选择整条进行。4/4默认在第1、3拍提供换和弦位置，3/4默认每小节一个，复拍子按附点拍分段。跨边界持续音同时参与两侧评分。显式 `--chord-progression` 仍优先。

每个片段的 `melody_inference` 导出局部评分、CSE覆盖率和前5个候选。CSE缺失部分采用同片段已知候选的中性值，不伪造查表结果；候选超过64个时，每片段保留旋律评分最高的64个参加连接搜索。对手写天干前8小节的核对及局限见 [重调律和声推断分析](RETUNE_HARMONY_ANALYSIS.md)。

## MSCX 与 IR 往返

```bash
# MSCX 转参考 IR
python3 mscx_to_ir.py always-with-me.mscx output/reference.json --staff-id 1

# 参考 IR、前端 IR 或完整 Score 转 MSCX
python3 ir_to_mscx.py output/reference.json output/reference.mscx
python3 ir_to_mscx.py output/666.frontend.json output/lead-only.mscx
python3 ir_to_mscx.py output/666.json output/666.mscx
```

外部 MSCX 第一次转换会规范化 XML。此后规范 MSCX 与参考 IR 构成固定点；语义哈希确保修改过的 IR 不会复用过期 XML。检查往返不变量：

```bash
python3 check_mscx_ir_roundtrip.py always-with-me.mscx
python3 check_mscx_ir_roundtrip.py output/reference.json
```

## 光谱模型与缓存

`spectral_model.py` 计算 SE、CSE、CCSE 三种基于频谱混合熵的指标，分别使用功率一次方、二次方和四次方。`spectral_bundle.py` 为 2、3、4、5 音组合建立统一二进制表，每条记录包含原始指标和对应表中的 midrank percentile。

风格中的目标混合是：

```text
CSE_objective = A * CSE + B * SE + C * CCSE
```

混合百分位会针对完整混合分布重新排序，不是三个百分位的线性组合。

`main.py` 会自动复用覆盖当前音域的缓存，缺失时自动构建。也可手动预构建：

```bash
python3 spectral_bundle.py \
  --scale scales/major_31.json \
  --rules scales/major_31.rules.json \
  --style scales/major_31.style.json \
  --out-dir CSE_cache --workers 8
```

缓存由配置和数值模型签名保护。修改音阶 pitch class、模型参数或所需音域后，会生成新缓存。缓存可以删除并重建，但五音表可能很大、重建时间较长。

## 全曲退火

`harmony_annealing.py` 在 Lead 和所有节奏固定的条件下改变伴奏音高。能量可包含：

- 同时起音与持续发声音响的光谱损失；
- 和弦背景场与和弦实现完整度；
- 最差 dyad、triad、tetrad 子集；
- 声部进行、重复、音域与 n-gram；
- pitch class 偏好和素数显著性；
- 八度族与 1:2:3、2:3:4 聚集惩罚。

关键参数位于 `style.annealing`：`objective_weights`、`worst_subset_cse_weights`、`cse_loss_polynomials`、`prime_salience`、`initialization` 和 `search`。初始化包含普通贪心、放宽跳进、扩展低音域和原子时间片联合救援，详见 [ROBUST_INITIALIZER.md](ROBUST_INITIALIZER.md)。

## 输出格式

### ScaleWeaverFrontEndIR/1

包含 `measure_map`、`scale`、`generator`、`harmony_plan`、`lead`、`lead_rhythm`，以及可选的 `melody_plan`。`frontend_ir.py` 会验证时间覆盖、和弦 ID、Lead 单声部性、音阶身份和节奏骨架一致性。

### ScaleWeaverScore/1

包含完整 `voices`、`harmony_plan`、`tuning`、配置快照、初始化与退火诊断、光谱系统信息、验证结果和统计摘要。事件中的 `step` 与 `freq` 是精确调律的权威音高。

### ScaleWeaverAttackIntervals/1

默认写入 `<score>_attacks.json`。它记录五音 sounding attack 的实际音高、相邻与成对距离、canonical 17-limit JI 比例、delete-one 子集比例，以及全部 attack 的素数出现统计。

## 音频渲染

```bash
# 内置加法合成
python3 wavgen8.py output/666.json output/666-raw.wav --backend raw

# FluidSynth/SF2，逐音符 pitch bend 保留微分音
python3 wavgen8.py output/666.json output/666-sf2.wav \
  --backend sf2 --sf2 piano.sf2 --sf2-reverb
```

voicebank 是可选的 Lead 人声叠加层。先创建样本清单，按 `render_plan.csv` 准备精确 F0 WAV，然后检查并渲染：

```bash
python3 voicebank_synth.py --scale scales/tiangan_72.json \
  --init voicebank/tiangan
python3 voicebank_synth.py --scale scales/tiangan_72.json \
  --check voicebank/tiangan

python3 wavgen8.py output/666.json output/666-voice.wav \
  --backend sf2 --sf2 piano.sf2 --sing-lead \
  --voicebank-dir voicebank/tiangan
```

## 保留的工具

- `baseline.py`：对完整 Score 计算 Lead 条件下的四声部 CSE baseline；
- `chord_progression_mscx.py`：把 `harmony_plan` 导出为块和弦 MSCX；
- `print_chord_progression.py`：以文本输出逐小节和弦进行；
- `print_attack_ratios.py`：输出五音 canonical 比例及五个四音子集比例；
- `create_missing_ngrams.py`：为目录中的音阶创建缺失的 anti-ABAB n-gram 文件，绝不覆盖现有文件。

```bash
python3 baseline.py output/666.json --cse-dir CSE_cache \
  --samples 1000 --output output/666.baseline.json
python3 chord_progression_mscx.py output/666.json output/666-chords.mscx
python3 print_chord_progression.py output/666.json --details
python3 print_attack_ratios.py output/666_attacks.json
python3 create_missing_ngrams.py scales
```

## 主要文件

| 文件 | 职责 |
| --- | --- |
| `piano.py` | 一键编排 JSON、WAV、MSCX |
| `main.py` | 主 CLI、运行时配置、前端选择、最终保存与 attack 分析 |
| `scale_config.py` | 三层配置加载、校验与 `ScaleSpec` |
| `melody_plan.py` | MelodyPlan/2 结构生成与校验 |
| `joint_frontend_generate.py` | 联合生成 Lead、节奏和和声 |
| `frontend_ir.py` | FrontEndIR 构造、校验、读写和哈希 |
| `imitation_frontend.py` | MSCX 解析、参考 IR 与节奏模仿前端 |
| `retune_melody.py` | 完整源旋律到目标音阶的最小误差重调律 |
| `accompaniment_generate.py` | 伴奏初始化和退火调度 |
| `score_builder.py` | 旋律与对位候选、节奏和硬规则 |
| `harmony_rhythm.py` | 和弦词汇、和声规划、节奏材料与音阶运行时 |
| `harmony_annealing.py` | 固定节奏全曲退火与指标 |
| `adaptive_cse_runtime.py` | 生成阶段的光谱查询和代价 |
| `spectral_model.py` | SE/CSE/CCSE 数值模型 |
| `spectral_bundle.py` | 统一光谱表构建、mmap 与查询 |
| `subset_cse_cache.py` | 退火所需的最差子集缓存 |
| `ji_ratio.py` | canonical 17-limit JI 表与查询 |
| `prime_salience.py` | 素数显著性模型 |
| `mscx_export.py` | 完整 Score 到 MSCX 与调律表 |
| `mscx_to_ir.py`、`ir_to_mscx.py` | MSCX/IR 双向转换 |
| `raw_synth.py`、`sf2_synth.py` | 两种乐器渲染后端 |
| `voicebank_synth.py` | 精确 F0 voicebank 初始化、校验和混音 |
| `wavgen8.py` | WAV 统一 CLI |

更细的前端边界见 [FRONTEND_IR.md](FRONTEND_IR.md)，MelodyPlan 规则见 [MELODYPLAN.md](MELODYPLAN.md)，同时起音限制见 [ATTACK_CSE_SOFT_LIMITS.md](ATTACK_CSE_SOFT_LIMITS.md)。

## 验证

```bash
python3 -m unittest discover -v
python3 -m py_compile *.py
python3 main.py --help
python3 piano.py --help
python3 wavgen8.py --help
```

## 设计边界

- 音阶必须包含至少三个 pitch class，且 `pcs` 必须从 0 开始、严格升序、没有重复；
- 一个进程同一时刻只有一套活动音阶运行时，不适合并发混用多套配置；
- 光谱表和 JI 表是生成前置数据，超出缓存覆盖音域时会触发新构建；
- MSCX 解析面向 MuseScore 3 的 XML 结构；复杂多声部参考谱应显式指定 `--staff-id`；
- 退火固定 Lead 与所有节奏，若要改变旋律，应重新生成或替换前端 IR；
- `output/`、`.mscbackup/`、示例 MSCX、缓存和 `piano.sf2` 都可能包含昂贵或人工挑选的数据，清理工作区时应按数据资产处理。
