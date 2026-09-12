# 不失败的伴奏初始化

新版伴奏初始化按以下顺序降级，Lead 与和声计划始终不变：

1. 在正常音域和正常最大跳进内贪心选择。
2. 若纵向合法候选存在但全部超过最大跳进，允许高代价的大跳。
3. 若固定的低 Lead 使正常伴奏音域无解，临时向下扩展伴奏候选音域。
4. 若多次普通节奏初始化仍然失败，在 Lead 与和弦变化点形成的原子时间片上联合求解 Bass、Inner、Counter。

所有阶段都保留狼音禁用、秒度禁用、严格 `Bass < Inner < Counter < Lead` 和 CSE 表范围。
扩展音域事件会写入诊断字段，现有退火器也能读取这些初始音高。

相关 style 参数位于 `annealing.initialization`：

```json
{
  "max_attempts": 24,
  "seed_stride": 1000003,
  "relax_melodic_jumps": true,
  "relaxed_jump_extra_cost": 12.0,
  "atomic_rescue": true,
  "atomic_rescue_lower_octaves": 1,
  "rescue_beam_width": 384,
  "expanded_range_extra_cost": 20.0
}
```

天干音阶的 Lead 配置下界同时由 `-72` 提高到 `-35`；其实际最低音为 `-30`，
高于 Counter 的 `-42`，且两者最低可用音相差 12 步，不触发天干狼音。
没有显式 `voice_ranges` 的通用音阶，Lead 默认下界改为约 `-0.30 EDO`，同样高于 Counter 下界。
