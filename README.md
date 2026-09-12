# Worm-0：持续运行的认知内核 v0

研究"代码怎样产生'想'"：一个外界输入只是扰动、不是启动信号的认知内核。
即使没有新信息，记忆再激活、世界模型预测、未解决的误差、学习进度也持续改变内部状态。

- 研究问题、公理、判据：[RESEARCH_PLAN.md](RESEARCH_PLAN.md)
- 先行者源码调查：[PRIOR_WORK.md](PRIOR_WORK.md)
- v0 实验结论：[FINDINGS_v0.md](FINDINGS_v0.md)

## 运行

```
python -m pip install numpy matplotlib   # matplotlib 仅画图用
python tools/gradcheck.py                # 手写反传的有限差分校验
python src/experiment.py --quick --seeds 1   # 冒烟（~10 秒）
python src/experiment.py --seeds 3           # 全量 E0（几分钟）
```

结果输出到 `results/`：verdicts.txt（S1–S4 判定）、summary.csv、ticks.csv、figure_e0.png。

## 结构

```
src/world.py      虫世界：物理后果 only（能量/完整度/晕厥/麻痹），无标签无奖励
src/kernel.py     固定随机 recurrent 内核，每拍必跑（时间恒在流动）
src/memory.py     情景记忆：h 为键，检索→再激活电流；晕厥修剪
src/nets.py       两层 MLP 手写反传（世界模型 + ΔC 预测器）
src/brain.py      经验在线分区 + 学习进度 + 回放分配 bandit（探究欲所在）
src/agent.py      接线：策略=想象 rollout 打分（想象不进训练）；关切广播
src/experiment.py E0 协议：醒-冻-醒-变-醒-冻-醒，三臂对照，S1–S4 自动判定
tools/gradcheck.py
```

## 无标签审计（代码里可验证）

进入大脑的训练信号只有三种：自监督预测误差、认知资本 C 的实测变化、
回放选择的内部误差下降奖励。世界真值（营养/毒物位置、物质身份）只用于
实验者的事后测量，从不进入大脑。
