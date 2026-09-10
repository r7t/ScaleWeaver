# ScaleWeaver：统一 SE / CSE / CCSE bundle

## 同时起音的混合 CSE 软上限

风格文件现在可用 `attack_cse_percentile_limits` 设置六类 attack/子集组合的
百分位上限，并用 `fallback_slack` 设置无解时的回退范围。默认六项为
60%/70%/80%/50%/60%/30%，回退范围为 10 个百分点。百分位按当前
`a*CSE + b*SE + c*CCSE` 混合分数重新排序，
并检查同一时刻 attack 集合的全部对应子集。有合规候选时只选择合规项；
确实无解时进入最小超限回退层，不会因为软上限清空候选。cleanup 默认关闭，
需要时显式传入 `--cleanup`。完整定义见 `ATTACK_CSE_SOFT_LIMITS.md`。

本版以混合狼音版工程为基础，保留三层配置、原天干和弦功能表、
八度等价/不等价狼音，以及新版 piano.py 的 WAV 输出入口。

## 一次生成三项表

```bash
python spectral_bundle.py --scale ji_major_171.json --out-dir . --workers 8
python spectral_bundle.py --scale tiangan_72.json --out-dir . --workers 8
```

旧命令 `python csebundle.py ji_major_171.json` 也转入统一预计算。
生成一个 manifest 和三个独立的 float64 文件：

```text
<scale>_spectral_<signature>_manifest.json
<scale>_spectral_<signature>_se_f64.bin
<scale>_spectral_<signature>_cse_f64.bin
<scale>_spectral_<signature>_ccse_f64.bin
```

三项总会一起生成，即使某项的当前权重为零。再次运行会复用有效文件。
需要可读 JSON 时，在命令末尾加 `--export-json`，会额外导出 SE、CSE、
CCSE 各一份 JSON；不重新计算已有表。大音域的 JSON 会明显大于二进制文件。

本包附天干和 171-EDO 五限纯律的已生成三项二进制表。现有规则里的
`cse_wide_bounds`、`cse_inner_bounds` 继续控制三项共同音域。
二、三、四音使用 wide 域，五音使用 inner 域，支持组合中的重复音。
原 `bass2/inner2/bass3/inner3/bass4/inner4/inner5` 七表接口保留。

## C 不再是偏移量

风格文件 `.style.json` 的 `cse_weights` 中：

```json
{
  "CSE_2D_A": 2.0,
  "CSE_2D_B": -1.0,
  "CSE_2D_C": -1.0
}
```

对应 `2*CSE - SE - CCSE`。键名保留，便于旧代码迁移；新含义为：

| 键 | 含义 |
|---|---|
| CSE_2D_A | CSE 系数 |
| CSE_2D_B | SE 系数 |
| CSE_2D_C | CCSE 系数，无常数偏移 |

默认风格文件仍为 `1/0/0`。另附 `ji_major_171.entropy_balance.style.json`
作为 `2/-1/-1` 示例。曾把 C 设为非零偏移量的旧风格必须手动调整；
程序会把该值直接解释成 CCSE 系数，不会继续按旧偏移量解释。

和弦选择、旋律背景场、实际声部、attack、删除音后的相对评分、
前瞻补全、后处理优化和 best-of 四部主项均使用该组合。
纯 CSE 基准报告保留，最终结果额外给出独立三指标的时长加权均值。
为兼容调用者，部分字段和函数仍含 `cse` 字样；最终结果中的
`mean_blended_score` 和 `mean_spectral_metrics` 明确区分混合分数与三项值。
百分位仍各自单独计算，兼容评分接口的第二项是纯 CSE 百分位，不是混合分数百分位。

## 与上传 C++ 对应的数值定义

真正执行的 `raw_se()` 使用 `make_cse_spectrum()`；C++ 中留下的
变带宽、面积补偿 `make_se_spectrum()` 未被调用。本版遵循实际执行路径，
三指标共用固定高斯带宽频谱 S：

- SE：`H(S)`。
- CSE：`H(S*S)`。
- CCSE：`H((S*S)*(S*S))`，即四重卷积，不是三重卷积。

每项最终均扣除同一组音高对应的单音熵算术平均。
预计算只生成一次各单音频谱、FFT 和三种原始熵；和弦将单音频谱相加算 SE，
将缓存 FFT 相加后计算 `F²`、`F⁴`，分别逆变换算 CSE、CCSE。
不能直接相加“各单音已卷积频谱”，那样会丢失不同音之间的交叉项。
重复音保留其频谱贡献与基线份额；全同音组的校正分数精确为零。

规则文件 `.rules.json` 统一使用：

```json
"spectral_parameters": {
  "sigma_hz": 1.0,
  "q_per_100hz": 0.8,
  "model_base_freq_hz": 100.0,
  "max_freq_hz": 4000.0,
  "resolution_hz": 1.0
}
```

使用自然对数、`p > 1e-12` 的熵求和阈值、0.001 的泛音振幅截断，
FFT 逆变换后取实部绝对值。默认 FFT 长度为 8192，按上传 C++
保留**循环卷积**，包括四重卷积的回卷；没有改成更长缓冲区的线性卷积。
这样定义固定，但不应把 CCSE 解释为无回卷的线性四重卷积结果。

CPP 交互入口会把第一个输入音归一化到模型 100 Hz；`score_ratios()`
复现这一约定。生成器表保留实际音区：`f = 100 * 2**(step/edo)`，
不按每个和弦的最低音重新归一化，也不做八度等价折叠。
实际乐曲的输出基频仍来自音阶定义，例如 E4=320 Hz，与模型基频分开。
原来五音背景场超出 inner 域时的整组八度投射行为保留，三指标使用同一投射。

旧 `se_parameters` 会警告并忽略；请迁移为上述 `spectral_parameters`。
`se_model.py`、`se_precompute.py`、`sebundle.py` 等旧 SE0 工具仅保留作
历史复现用途，不再被作曲评分链路调用。新建表应使用 `spectral_bundle.py`
或已更新的 `csebundle.py`。新格式和签名不会误加载旧 SE/CSE 表。

## 一键作曲

```bash
python piano.py 17171
python piano.py 17171 --style ji_major_171.entropy_balance.style.json
```

保留 piano.py 顶部的三个 JSON 全局变量。默认 JSON → WAV → MSCX 与调律文件。
本包不含 piano.sf2；需要现有音源和 FluidSynth 环境。
只生成乐谱可使用本次实际验证命令：

```bash
python piano.py 17171 --style ji_major_171.entropy_balance.style.json --bars 8 --best-of 2 --no-sixteenth --no-wav --output-dir example_entropy
```

## 文件与验证

- `spectral_model.py`：统一数值核心与单独和弦计算接口。
- `spectral_bundle.py`：并行预计算、三文件读写、缓存签名和 JSON 导出。
- `csebundle.py`：旧 CSE bundle 入口兼容层。
- `scale_config.py`：新参数验证及 C 的新语义。
- `adaptive_cse_runtime.py`、`harmony_rhythm.py`、`main.py`：三指标评分链路。
- `reference/CSE_SE_CCSE_unified.cpp`：用户上传的原始算法。
- `tests/test_spectral.py`：C++ 对照、重复音、音区、三文件、并行及权重测试。
- `example_entropy/`：非零 CCSE 权重的实际生成结果。

运行 `python -m unittest discover -s tests -v`；C++ 对照测试需要 g++，
其余数值部分仅依赖 NumPy。测试结果和样例核对见 `verification.json`。
