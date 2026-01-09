# 3D-Molecule-Align

## 项目简介

本项目聚焦于反应物/产物 3D 分子坐标的原子索引映射（atom mapping）问题，提供传统几何基线方法与学习模型（EGNN、扩散模型）的训练与评估脚本，并给出统一的验证指标与结果统计。

## 方法概览

- 基线方法：Distance、QML、Inertia-Hungarian（均依赖 `rmsd`）
- 学习方法：EGNN 监督匹配网络、Hybrid Diffusion（噪声预测 + 对齐损失）
- 评估指标：严格准确率（Acc.）、平均汉明距离（Avg. HD）、平均耗时（ms/mol）

## 项目结构

```
rmsd_test/
├── data/
│   ├── raw/                    # 原始数据（下载后）
│   └── ready/                  # 预处理后的 pkl 数据
├── models/                     # 训练权重（如 EGNN）
├── results/                    # 基线与 EGNN 的结果与图表
├── scripts/
│   ├── data_process/           # 下载、预处理、切分数据
│   ├── train/                  # 训练脚本（EGNN / Diffusion）
│   └── test/                   # 评估脚本（Baselines / EGNN / Diffusion）
└── src/                        # 核心模型与算法实现
```

## 安装依赖

```bash
pip install -r requirements.txt
```

> 如需 GPU 加速，请确保本机 CUDA 与 PyTorch / PyG 版本匹配。

## 数据准备

#### 方法一：原始数据下载与处理

```bash
cd scripts/data_process
python download_data.py
cd ../../data/raw/
tar -xjvf RDB19-Rad_Reactions.tar.bz2
cd ../../scripts/data_process/
python preprocess_data.py
python split_datasets.py --data-dir ../../data/ready/
```

#### 方法二：使用已预处理的数据

从指定huggingface位置直接下载处理好的数据，- [Download Weights and Data](https://huggingface.co/SII-SikoraMi/3d-mol-mapping/tree/main)

## 运行与评估

### Baseline（Distance / QML / Inertia-Hungarian）

```bash
cd scripts/test
python test_reorder_distance.py --dataset ../../data/ready
python test_reorder_qml.py --dataset ../../data/ready
python test_reorder_hungarian_algorithm.py --dataset ../../data/ready
```

### EGNN 训练与评估

```bash
cd scripts/train
python egnn.py
```

```bash
cd scripts/test
python test_reorder_egnn.py
```

> EGNN 默认权重路径为 `models/best_gnn_model.pth`，如有调整请在脚本中修改。

### Hybrid Diffusion（DDP）

训练（脚本内默认配置，按需改动）：

```bash
cd scripts/train
python diffusion_train_hybrid_ddp.py
```

评估（支持 argparse 参数，以下为默认值）：

```bash
cd scripts/test
python diffusion_eval_hybrid_ddp.py \
  --data-dir ../../data/ready \
  --output-dir ../../outputs_diffusion_hybrid \
  --checkpoint-path ../../outputs_diffusion_hybrid/best_diffusion_model.pt \
  --val-paths ../../data/ready/rgd1_val.pkl ../../data/ready/t1x_val.pkl \
  --batch-size 512 \
  --num-workers 8 \
  --num-sampling-steps 50
```

说明：
- `--checkpoint-path` 省略时默认使用 `--output-dir/best_diffusion_model.pt`
- `--val-paths` 省略时默认使用 `--data-dir/rgd1_val.pkl` 与 `--data-dir/t1x_val.pkl`
- 评估结果输出到 `outputs_diffusion_hybrid/`（按数据集分别保存 JSON）

## 测试结果

验证集结果（Acc.↑ / Avg. HD↓ / 耗时↓ ms/mol）：

| 方法 | T1x (ID) Acc.↑ | T1x Avg. HD↓ | T1x 耗时↓ (ms/mol) | RGD1 (ID) Acc.↑ | RGD1 Avg. HD↓ | RGD1 耗时↓ (ms/mol) | RDB19-Rad (OOD) Acc.↑ | RDB19-Rad Avg. HD↓ | RDB19-Rad 耗时↓ (ms/mol) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Distance | 2.18% | 8.12 | 0.03 | 0.41% | 11.21 | 0.03 | 0.00% | 19.08 | 0.03 |
| QML | 23.23% | 3.87 | 0.58 | 16.70% | 4.74 | 1.07 | 13.66% | 6.14 | 3.66 |
| Inertia-Hungarian | 27.00% | 5.14 | 0.42 | 23.47% | 6.88 | 0.48 | 29.73% | 7.35 | 0.50 |
| EGNN | 43.57% | 2.38 | 0.62 | 68.09% | 1.13 | 0.25 | 34.64% | 3.11 | 1.49 |
| Diffusion | 75.50% | 0.71 | 16.63 | 85.58% | 0.42 | 16.33 | 57.32% | 1.70 | 17.23 |

## 数据说明

原始数据来源：
- [T1x 数据集](https://figshare.com/articles/dataset/Transition1x/19614657)
- [RGD1 数据集](https://figshare.com/articles/dataset/model_reaction_database/21066901)
- [RDB19-Rad 数据集](https://zenodo.org/records/11493786)

预处理后的 pickle 文件存储的是 `AtomMapping` 对象列表，结构如下：

```
pickle 文件
└── [AtomMapping对象1, AtomMapping对象2, ..., AtomMapping对象N]

AtomMapping 对象
├── structure_ref (MolStructure)
│   ├── atoms (np.ndarray, dtype=int)        # 原子序数 (N,)
│   └── coordinates (np.ndarray, float64)    # 坐标 (N, 3)
├── structure_cand (MolStructure)
│   ├── atoms (np.ndarray, dtype=int)
│   └── coordinates (np.ndarray, float64)
├── mapping_indices (np.ndarray, dtype=int) # 候选→参考映射 (N,)
└── source (str)                            # "RDB19_*", "RGD1_*", "T1x_*"
```

其中 `mapping_indices[i]` 表示候选分子第 `i` 个原子映射到参考分子中的位置。

## 资源与链接

- [Download Weights and Data](https://huggingface.co/SII-SikoraMi/3d-mol-mapping/tree/main)

## 许可证

MIT License
