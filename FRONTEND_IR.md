# ScaleWeaver 两阶段前端 IR

程序在 Lead/和声计划与三或四个伴奏声部之间建立明确边界：

```text
MelodyPlan：分形依赖 + 旋律/节奏/和声联合生成
                         ↓ ScaleWeaverFrontEndIR/1
后端：Counter/Bass/Inner 初始化 + 全曲退火 + 统计与导出
```

`main.py` 默认调用 MelodyPlan 联合前端，也可读取已有 IR、MSCX 节奏模板或完整旋律重调律源。

## 两阶段命令

先生成中间表示：

```bash
python joint_frontend_generate.py \
  --scale scales/tiangan_72.json \
  --rules scales/tiangan_72.rules.json \
  --style scales/tiangan_72.style.json \
  --seed 666 --bars 48 --output 666.frontend.json
```

再读取它生成完整四或五声部结果：

```bash
python accompaniment_generate.py \
  --input 666.frontend.json \
  --scale scales/tiangan_72.json \
  --rules scales/tiangan_72.rules.json \
  --style scales/tiangan_72.style.json \
  --output 666.json
```

也可以继续使用 `main.py`：

```bash
# 一步生成，同时留下 IR
python main.py --scale scales/tiangan_72.json --seed 666 \
  --save-frontend-ir 666.frontend.json --output 666.json

# 从已有 IR 继续
python main.py --scale scales/tiangan_72.json \
  --frontend-ir 666.frontend.json --output 666.json
```

## IR 的稳定边界

`ScaleWeaverFrontEndIR/1` 顶层包含：

- `seed`、速度、拍号、小节数；
- `measure_map`：每小节的绝对起点、实际时长和独立拍号；它是动态小节的
  权威时间轴，普通固定拍号作品也会得到等价的均匀时间轴；
- `scale`：音阶 ID、EDO、音级和基频身份；
- `generator`：前端名称、十六分音符开关、动机参数、已解析人工和弦进行；
- `harmony_plan`：逐小节和弦片段；
- `lead`：完整 Lead 事件；
- `lead_rhythm`：与 Lead 音高分离的逐小节起音/时值骨架及旧节奏型库元数据。

`frontend_ir.py` 会验证音阶匹配、和弦是否存在、和声片段是否完整填满每小节、
Lead 是否单声部且时间合法，以及显式节奏骨架是否与 Lead 事件一致。
所有绝对时间继续统一使用四分音符为 1 拍；`measure_map` 允许逐小节变拍、
不满小节和任意长度小节，后端不再依赖 `bar * beats_per_bar` 定位这些作品。

## MSCX 双向转换与稳定性

`mscx_to_ir.py` 输出含最高线语义及受语义哈希保护的完整规范 MSCX 载荷；
`ir_to_mscx.py` 支持 `ScaleWeaverMSCXReferenceIR/1`、
`ScaleWeaverFrontEndIR/1` 和 `ScaleWeaverScore/1`。未修改导入 IR 时会回写完整
原谱；语义被修改时不会复用过期载荷，而是按新语义重建规范谱。

外部谱第一次转换允许进行 XML 格式规范化。自第一次规范输出起，以下不变量
成立：`MSCX1 == MSCX2`、`IR2 == IR3`；从外部 MSCX 开始时则有
`IR1 == IR2`、`MSCX2 == MSCX3`。

未来 MelodyPlan 写作器只需要输出同一 IR。后端不会要求它复制旧生成器内部的
动机、轮廓或随机过程，也不会修改输入的 Lead 与和声计划。

## 文件职责

- `frontend_ir.py`：IR 构造、校验、指纹和 JSON 读写。
- `melody_plan.py`：生成分形结构和半小节继承动作。
- `joint_frontend_generate.py`：按计划联合实现和声、Lead 节奏与音高。
- `imitation_frontend.py`：读取 MSCX/参考 IR 的时间结构并重新创作目标音高。
- `retune_melody.py`：保留完整源旋律并映射到目标音阶。
- `accompaniment_generate.py`：读取 IR，生成三或四个伴奏声部并调用退火器。
- `harmony_annealing.py`：实现固定 Lead、固定节奏的全曲优化。
- `main.py`：配置、一步/两阶段调度、统计及最终 JSON 保存。

和声复制只指 `harmony_plan`，不包含 Bass/Inner/Counter 的实际音高。这保证
后续分形 MelodyPlan 可以自由重写前端，同时伴奏生成逻辑保持独立。
