# utils.py
from rmsd import ELEMENT_WEIGHTS, ELEMENT_NAMES, NAMES_ELEMENT, AXIS_REFLECTIONS
from typing import Dict, List, Union, Optional
import numpy as np

def str_atom(atom: int) -> str:
    """将原子序数转换为元素符号"""
    if atom not in ELEMENT_NAMES:
        raise KeyError(f"原子序数 {atom} 未在元素映射表中定义")
    return ELEMENT_NAMES[atom]


def int_atom(atom: str) -> int:
    """将元素符号转换为原子序数"""
    atom_key = atom.capitalize().strip()
    if atom_key not in NAMES_ELEMENT:
        raise KeyError(f"元素符号 {atom} 未在元素映射表中定义")
    return NAMES_ELEMENT[atom_key]


class MolStructure:
    """分子结构数据类，直接存储为 numpy 数组格式"""
    def __init__(self, atoms: Union[List[str], List[int], np.ndarray], coordinates: Union[List[List[float]], np.ndarray]):
        # 处理原子类型：统一转换为 numpy 数组（dtype=int，原子序数）
        if isinstance(atoms, np.ndarray):
            if atoms.dtype != int:
                raise TypeError("numpy 数组类型的 atoms 必须是 int 类型（原子序数）")
            self.atoms = atoms.copy().astype(int)
        elif isinstance(atoms, list) and len(atoms) > 0:
            if isinstance(atoms[0], str):
                # 元素符号列表 -> 原子序数 numpy 数组
                self.atoms = np.array([int_atom(atom) for atom in atoms], dtype=int)
            elif isinstance(atoms[0], int):
                # 原子序数列表 -> numpy 数组
                self.atoms = np.array(atoms, dtype=int)
            else:
                raise TypeError("atoms 必须是元素符号列表、原子序数列表或 int 类型的 numpy 数组")
        else:
            raise ValueError("atoms 不能为空")
        
        # 处理坐标：统一转换为 numpy 数组（dtype=np.float64，适配 rmsd 函数）
        if isinstance(coordinates, np.ndarray):
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError("numpy 数组类型的 coordinates 必须是 (N, 3) 形状")
            self.coordinates = coordinates.copy().astype(np.float64)
        elif isinstance(coordinates, list) and len(coordinates) > 0:
            if len(coordinates) != len(atoms):
                raise ValueError("coordinates 长度必须与 atoms 一致")
            # 列表 -> (N, 3) 的 numpy 数组
            self.coordinates = np.array(coordinates, dtype=np.float64)
            if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
                raise ValueError("coordinates 必须是包含 3 个数值的列表组成的列表（x, y, z）")
        else:
            raise ValueError("coordinates 不能为空")
        
        # 验证原子数与坐标数一致
        if self.atoms.shape[0] != self.coordinates.shape[0]:
            raise ValueError(f"原子数（{self.atoms.shape[0]}）与坐标数（{self.coordinates.shape[0]}）不匹配")
    
    def get_atom_symbols(self) -> List[str]:
        """获取元素符号列表（用于可视化/打印）"""
        return [str_atom(atom) for atom in self.atoms]
    
    def get_atom_coordinates(self) -> np.ndarray:
        """获取原子坐标数组（用于可视化/打印）"""
        return self.coordinates.copy()

class AtomMapping:
    """原子对之间索引的序列映射类，用于表示候选分子到参考分子的原子映射关系"""
    def __init__(self, structure_ref: MolStructure, structure_cand: MolStructure, mapping_indices: np.ndarray, source: str = "test"):
        """
        初始化原子映射类
        
        参数:
            structure_ref: 参考分子结构（MolStructure实例）
            structure_cand: 候选分子结构（MolStructure实例）
            mapping_indices: 重排序索引数组，其中mapping_indices[i]表示候选分子中第i个原子映射到参考分子中的位置
            source: 映射来源描述（字符串，默认"test"）
        """
        if not isinstance(structure_ref, MolStructure):
            raise TypeError("structure_ref 必须是 MolStructure 实例")
        if not isinstance(structure_cand, MolStructure):
            raise TypeError("structure_cand 必须是 MolStructure 实例")
        if not isinstance(mapping_indices, np.ndarray):
            raise TypeError("mapping_indices 必须是 numpy 数组")
        
        # 验证索引数组的有效性
        if mapping_indices.ndim != 1:
            raise ValueError("mapping_indices 必须是一维数组")
        if len(mapping_indices) != len(structure_cand.atoms):
            raise ValueError(f"mapping_indices 长度（{len(mapping_indices)}）必须与候选分子原子数（{len(structure_cand.atoms)}）一致")
        
        self.structure_ref = structure_ref
        self.structure_cand = structure_cand
        self.mapping_indices = mapping_indices.copy().astype(int)
        self.source = source
    
    def get_mapped_candidate(self) -> MolStructure:
        """
        获取根据映射关系重排序后的候选分子结构
        
        返回:
            重排序后的MolStructure实例，其原子顺序与参考分子对齐
        """
        # 重排序候选分子的原子和坐标
        mapped_atoms = self.structure_cand.atoms[self.mapping_indices]
        mapped_coordinates = self.structure_cand.coordinates[self.mapping_indices]
        return MolStructure(mapped_atoms, mapped_coordinates)
    
    def get_mapping_pairs(self) -> List[tuple]:
        """
        获取原子映射对列表
        
        返回:
            列表，其中每个元素是一个元组 (ref_index, cand_index)，表示参考分子中的原子索引和对应的候选分子原子索引
        """
        return [(i, int(cand_idx)) for i, cand_idx in enumerate(self.mapping_indices)]


if __name__ == "__main__":
    # 测试代码
    atoms = np.array([6, 1, 1, 1, 1], dtype=int)
    coordinates = np.array([
        [-0.00000e+00,  1.10000e-05, -0.00000e+00],
        [-9.08157e-01,  6.02462e-01, -0.00000e+00],
        [ 8.70766e-01,  6.55345e-01, -0.00000e+00],
        [ 1.86960e-02, -6.28937e-01,  8.89828e-01],
        [ 1.86960e-02, -6.28937e-01, -8.89828e-01]
    ], dtype=np.float64)
    structure = MolStructure(atoms, coordinates)
    print("元素符号:", structure.get_atom_symbols())
    print("坐标:", structure.get_atom_coordinates())