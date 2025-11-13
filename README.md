# 3D-Molecule-Align

## 项目概述

本项目旨在测试不同的3D坐标对齐方法，特别是基于惯性矩的匈牙利算法在分子结构匹配中的性能。项目主要实现了分子结构原子重排序功能，并通过严格匹配方式验证重排序算法的准确性。

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
python test_hungarian_algorithm.py
```
测试结果将保存在 `results/` 文件夹下


## 数据说明
原始数据来源于开源数据集
- [T1x数据集](https://figshare.com/articles/dataset/Transition1x/19614657)，元素类型涵盖 C, H, O, N，7 个重原子以下，包含 10073 个反应
- [RGD1数据集](https://figshare.com/articles/dataset/model_reaction_database/21066901)，元素类型涵盖 C, H, O, N，10 个重原子以下,包含 176992 个反应
- [RDB19-Rad数据集](https://zenodo.org/records/11493786)，元素类型涵盖 C, H, O, N, S，19 个重原子以下，包含 5600 个反应

预处理后的 pickle 数据包含以下信息：
- 参考分子结构（Reference Structure）
- 候选分子结构（Candidate Structure）
- 正确的原子映射关系（Atom Mapping）
- 数据来源标识（Source）


## 算法说明

### 基于惯性矩的匈牙利算法（inertia_hungarian）

该算法为 baseline，直接调用 [rmsd GitHub仓库](https://github.com/charnley/rmsd)

## 许可证

MIT License
