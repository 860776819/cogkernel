# soul0/ —— Soul-0 代码

四个文件夹，各管一件事：

| 文件夹 | 是什么 |
|---|---|
| `core/` | 小夕当前最底层的"身体"：状态（组分与调节自由度）和变化规则（方程）。`proto0.py` 是第一代最小存在组织，`proto1.py` 在其上加了调节自由度。 |
| `experiments/` | 我们拿它做实验的程序。按研究顺序：`run_e0.py`（E0 最小存在组织）→ `e05_discrimination.py` / `e05_horizon_check.py`（E0.5 内部差异审计）→ `run_e1.py`（E1 调节自由度）→ `run_atlas0.py`（Atlas-0 状态空间图谱）。 |
| `analysis/` | 分析实验结果数据的程序（目前是 Atlas-0 的翻转面与可达性统计）。 |
| `visualization/` | 把结果做成可看的观察台（`build_observatory.py` 生成可双击打开的 3D 工作台）。 |

## 怎么运行

在任意目录都可以，脚本自己会找到仓库位置：

```
python soul0/experiments/run_e0.py        # 例：跑 E0
python soul0/visualization/build_observatory.py   # 例：重建观察台
```

## 研究顺序

E0 → E0.5 → E1 → Atlas-0 → Observatory。
每一步的设计、判据与结论见 [`docs/soul0/README.md`](../docs/soul0/README.md)；
实验数据在 [`results/`](../results/)。
