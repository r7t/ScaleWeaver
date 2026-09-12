# MelodyPlan 联合前端

新版默认前端分为两个层次：`melody_plan.py` 先决定材料之间的依赖，
`joint_frontend_generate.py` 再联合选择 Lead 音高、Lead 节奏与和弦进行。
结果写入 `ScaleWeaverFrontEndIR/1`，再由后端生成三或四个伴奏声部并执行全曲退火。

## 计划结构

- 每 8 小节递归分为 4、2、1 小节；底层时间网格仍为半小节。
- `MelodyPlan/2` 的材料叶节点可长 0.5、1 或 2 小节。1 小节叶必须从整小节
  起点开始，2 小节叶必须从二进制对齐的两小节块起点开始；因此不会出现
  `x|xxxx|xx|x` 一类错位分块。
- 两小节块的两个一小节子块之间只允许复制节奏，不复制旋律或和声。
- 四小节块的两个两小节子块以新写为主：偶尔复制节奏，和声复制极少；若复制旋律，只复制第二个子块开头半小节。
- 八小节块可以复制较长的开头旋律、节奏或和声材料。
- 48 小节直接选择 2–4 个材料族的六段式结构；内置形式包括
  `AABBAB`（AA′BB′A″B″）、`AAB_CBC`（AA′BCB′C′）、`ABCABC`、
  `ABACBC` 与 `ABCDAC`。同一材料族的后继段依赖最近一次出现，形成逐次变奏链。
- 合法继承模式只有 `FRESH`、`H`、`R`、`RH`、`MR`、`MRH`。
- `M` 表示旋律音高，`R` 表示 Lead 节奏，`H` 表示和弦进行；复制旋律必然同时复制节奏。
- 复制长度可从半小节到 8 小节，同一段内部可以采用不同继承等级。
- 旋律复制支持原音区和整体移高/移低八度；节奏支持精确复制和近似复制。
- 每 8 小节末尾继续强制使用现有锚定和弦。
- 跨段默认更偏向两个材料族、2–4 小节复现以及 `MR/MRH`，使主题身份更多
  由旋律本身承担，而不只是复制节奏外壳。

## 联合生成

每个半小节同时枚举少量和声片段、节奏单元与旋律实现，并用 beam 保留综合代价最低的路径。
内部代价包括音区曲线、密度曲线、旋律进行与 n-gram、和弦背景场、强拍和弦音、
和弦连接和句尾收束。所有权重位于 style JSON 的 `melody_plan.joint_generation`。

联合生成器会读取每小节的 `bar_roles`：主题陈述倾向上行，主题回答倾向下行，
发展段使用方向交替的波形，高潮提高音区和密度，清算与终止逐步降低音区和密度，
并增强小节末尾向当前和弦根音落定的倾向。具体轮廓、方向、音区偏移、密度倍率
和落根权重位于 `melody_plan.bar_role_profiles`。

12、19、31 平均律大调和 171 平均律五限纯律大调另外启用滑动窗口跳进平衡：
它以最近 24 个音程中三度及以上、五度及以上跳进的软目标比例修正候选代价，
在比例不足时鼓励跳进，超过目标时自动收敛；重拍更容易承担结构性跳进，
十六分音符等短音则降低跳进倾向。该机制不强制配额，也不改变天干音阶。
参数位于各自然音阶 rules JSON 的 `metadata.generator.lead_motion`。

允许十六分音符时，高密度半小节会保留至少一个不少于
`dense_half_min_notes` 个音的候选；默认密度基准为每小节 9 音，候选注入概率由
`dense_half_candidate_probability` 控制。使用 `--no-sixteenth` 时，4/4 拍的半小节
仍因节奏网格限制而最多容纳 4 音。

`melody_plan.joint_generation.allow_cross_unit_ties` 默认开启。前端分别以
`cross_half_bar_tie_probability` 和 `cross_bar_tie_probability` 选择少量半小节、
小节入口作为延音，而不是新起音。延续音必须是进入和弦的和弦音；和声变化时还
必须是前后和弦的共同音。段落开头保留新起音，终止位置降低延音概率，连续时长由
`max_tied_duration_beats` 限制。它们在 IR 中合并为可跨半小节、跨小节的 Lead
长音。MuseScore 导出会把跨小节长音拆成带连音线的记谱片段。

```bash
python main.py --scale scales/tiangan_72.json --seed 666 --bars 48 \
  --save-frontend-ir 666.frontend.json --output 666.json
```

只生成计划与前端 IR：

```bash
python joint_frontend_generate.py --scale scales/tiangan_72.json \
  --seed 666 --bars 48 --plan-output 666.melodyplan.json \
  --output 666.frontend.json
```

联合前端是当前唯一的内部创作前端。外部旋律来源通过前端 IR、节奏模仿或完整旋律重调律接入。
