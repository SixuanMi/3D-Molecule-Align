# inertia_hungarian.py
import numpy as np
from typing import Any

# 从 rmsd 模块导入所有需要的函数和常量
from rmsd import reorder_inertia_hungarian as rmsd_reorder_inertia_hungarian

import sys
sys.path.append("/Users/misixuan/Desktop/codingwithai/rmsd_test")

# 导入数据类和工具函数
from src.utils import MolStructure, AtomMapping


def reorder_inertia_hungarian(
    structure_ref: MolStructure,
    structure_cand: MolStructure,
    **kwargs: Any,
) -> np.ndarray:
    """
    适配 MolStructure 类的原子重排序函数，内部调用 rmsd 模块的实现
    
    参数:
        structure_ref: 参考结构（MolStructure实例）
        structure_cand: 需要重排序的候选结构（MolStructure实例）
        **kwargs: 传递给 rmsd.reorder_inertia_hungarian 的额外参数
    
    返回:
        重排序索引数组，使structure_cand[indices]与structure_ref对齐
    """
    # MolStructure 实例的属性
    p_atoms = structure_ref.atoms
    q_atoms = structure_cand.atoms
    p_coord = structure_ref.coordinates
    q_coord = structure_cand.coordinates
    
    # 直接调用 rmsd 模块的核心重排序函数
    return rmsd_reorder_inertia_hungarian(
        p_atoms=p_atoms,
        q_atoms=q_atoms,
        p_coord=p_coord,
        q_coord=q_coord,
        **kwargs
    )

# 使用示例
if __name__ == "__main__":
    # 示例1：创建两个分子结构并进行对齐
    # 参考结构：乙烷分子（简化坐标）
    ethane_ref = MolStructure(
        atoms=["C", "C", "H", "H", "H", "H", "H", "H"],
        coordinates=[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.5, 1.0, 0.0],
            [1.5, -1.0, 0.0],
            [1.5, 0.0, 1.0],
        ]
    )
    
    # 候选结构：原子顺序打乱的乙烷
    ethane_cand = MolStructure(
        atoms=["H", "C", "H", "C", "H", "H", "H", "H"],
        coordinates=[
            [0.0, 1.0, 0.0],  # H
            [0.0, 0.0, 0.0],  # C
            [0.0, -1.0, 0.0], # H
            [1.5, 0.0, 0.0],  # C
            [0.0, 0.0, 1.0],  # H
            [1.5, 1.0, 0.0],  # H
            [1.5, -1.0, 0.0], # H
            [1.5, 0.0, 1.0],  # H
        ]
    )
    
    # 获取重排序索引
    mapping_indices = reorder_inertia_hungarian(ethane_ref, ethane_cand)
    print("重排序索引:", mapping_indices)
    
    # 创建AtomMapping实例
    atom_mapping = AtomMapping(ethane_ref, ethane_cand, mapping_indices)
    
    # 获取映射对
    mapping_pairs = atom_mapping.get_mapping_pairs()
    print("原子映射对:")
    for ref_idx, cand_idx in mapping_pairs:
        ref_atom = ethane_ref.get_atom_symbols()[ref_idx]
        cand_atom = ethane_cand.get_atom_symbols()[cand_idx]
        print(f"参考分子原子 {ref_idx}({ref_atom}) -> 候选分子原子 {cand_idx}({cand_atom})")
