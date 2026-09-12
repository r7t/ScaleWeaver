# 同时起音的光谱百分位软限制

该功能在生成每个新音时检查同一时刻起音的 2、3、4 音组合。持续中的旧音不属于同时起音组；持续音响由其他光谱目标处理。

这里的指标是风格配置指定的完整混合：

```text
objective = A * CSE + B * SE + C * CCSE
```

`A`、`B`、`C` 分别是 `style.cse_weights` 中的 `CSE_2D_A`、`CSE_2D_B`、`CSE_2D_C`。程序会对每张原生表中的完整混合值重新排序并计算 midrank percentile。

## 配置

字段位于 CompositionStyle/1 顶层：

```json
{
  "attack_cse_percentile_limits": {
    "two_note_attack": {
      "dyad": 0.60
    },
    "three_note_attack": {
      "dyad_subsets": 0.70,
      "triad": 0.50
    },
    "four_note_attack": {
      "dyad_subsets": 0.80,
      "triad_subsets": 0.60,
      "tetrad": 0.30
    },
    "fallback_slack": 0.10
  }
}
```

所有数值必须是 `0.0` 到 `1.0` 的有限小数。字段存在时，以上分组和子字段必须完整；省略整个字段会关闭此功能。

## 检查范围

- 两音 attack：检查完整 dyad；
- 三音 attack：检查三个 dyad 与完整 triad；
- 四音 attack：检查六个 dyad、四个 triad 与完整 tetrad；
- 一音及五音 attack：当前没有这组软门限。

每个百分位使用与该子集音数和音域对应的统一光谱表。

## 候选选择

限制采用可放宽的门限，不能单独造成“无候选”：

1. 只要存在完全满足所有门限的候选，就只保留这些候选；
2. 若所有候选都超限，先找到最小的最大超限值；
3. 保留最大超限值不高于“最小值 + `fallback_slack`”的候选；
4. 在该候选层中按最大超限和总超限增加连续代价，再交给正常音乐代价采样；
5. 数值异常时仍回退到原候选集合。

最终 Score 的 `simultaneous_attack_cse_limits` 记录启用状态、混合权重、六项门限、检查数量、最大超限值和少量超限示例。
