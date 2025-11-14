# 3D-Molecule-Align

## 项目概述

本项目为 Coding with AI 的课程大作业，旨在通过人工智能方法实现分子 3D 坐标的对齐，基线采用基于惯性矩的匈牙利算法，并通过映射严格匹配方式验证重排序算法的准确性。

## 项目结构

```
rmsd_test/
├── data/                       # 测试数据目录
│   ├── raw/                    
│   └── ready/                  
├── src/                        # 源代码目录
│   ├── inertia_hungarian.py    # 基于惯性矩的匈牙利算法实现
│   └── utils.py                # 工具函数（包含MolStructure和AtomMapping类）
├── scripts/                    # 测试脚本目录
│   ├── download_data.py        # 数据下载脚本
│   ├── preprocess_data.py      # 数据预处理脚本
│   └── test_hungarian_algorithm.py # baseline 测试脚本
├── results/                    # 测试结果目录
│   └── inertia_hungarian_validation_results.txt
├── requirements.txt
└── README.md
```

## 安装依赖

使用以下命令安装项目依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

#### 方法一：原始数据下载与处理

```bash
cd scripts
python download_data.py
cd ../data/raw/
tar -xjvf RDB19-Rad_Reactions.tar.bz2
python ../../scripts/preprocess_data.py
```

#### 方法二：使用已预处理的数据

将预处理好的数据文件放入 `data/ready/` 目录中。

### 2. 运行测试脚本

```bash
cd scripts
python test_reorder_hungarian_algorithm.py
python test_reorder_distance_algorithm.py
python test_reorder_qml_algorithm.py
```
测试结果将保存在 `results/` 文件夹下


## 数据说明
原始数据来源于开源数据集
- [T1x数据集](https://figshare.com/articles/dataset/Transition1x/19614657)，元素类型涵盖 C, H, O, N，7 个重原子以下，包含 10073 个反应
- [RGD1数据集](https://figshare.com/articles/dataset/model_reaction_database/21066901)，元素类型涵盖 C, H, O, N，10 个重原子以下,包含 176992 个反应
- [RDB19-Rad数据集](https://zenodo.org/records/11493786)，元素类型涵盖 C, H, O, N, S，19 个重原子以下，包含 5600 个反应

预处理后的 pickle 文件存储的是 `AtomMapping` 对象的列表，每个对象包含以下层级结构：

```
pickle文件
└── [AtomMapping对象1, AtomMapping对象2, ..., AtomMapping对象N]

AtomMapping对象
├── structure_ref (MolStructure) - 参考分子结构
│   ├── atoms (np.ndarray, dtype=int) - 原子序数数组，形状为 (N,)，N为原子数
│   └── coordinates (np.ndarray, dtype=float64) - 原子坐标数组，形状为 (N, 3)
├── structure_cand (MolStructure) - 候选分子结构
│   ├── atoms (np.ndarray, dtype=int) - 原子序数数组，形状为 (N,)，N为原子数
│   └── coordinates (np.ndarray, dtype=float64) - 原子坐标数组，形状为 (N, 3)
├── mapping_indices (np.ndarray, dtype=int) - 原子映射索引数组，形状为 (N,)
└── source (str) - 数据来源标识（"RDB19_*", "RGD1_*", "T1x_*"）
```
其中 mapping_indices[i] 表示候选分子中第i个原子映射到参考分子中的位置

## 算法说明

### 基于惯性矩的匈牙利算法（inertia_hungarian）
### 距离重排序算法（distance）
### QML重排序算法（qml）
都直接调用 [rmsd GitHub仓库](https://github.com/charnley/rmsd)

### AI 相关的重排序算法（需实现）

## 测试结果

以下是在三个数据集上各算法的严格正确率测试结果：

| 算法名称 | RDB19 | RGD1 | T1x |
|---------|------------|-----------|----------|
| inertia_hungarian | 31.04% | 23.36% | 27.02% |
| qml | 14.09% | 16.63% | 23.74% |
| distance | 0.02% | 0.40% | 2.10% |


## 许可证

MIT License
