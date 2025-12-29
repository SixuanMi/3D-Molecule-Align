import h5py
import numpy as np
import pickle
import os
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from ase.io import read

import sys
sys.path.append("/Users/misixuan/Desktop/codingwithai/rmsd_test")
from src.utils import MolStructure, AtomMapping  # 核心数据类

import transition1x as t1x

# --------------------------
# 1. 数据读取
# --------------------------
def load_rgd1_data(h5_file_path: str) -> List[Tuple[MolStructure, MolStructure, str]]:
    """
    从 RGD1_CHNO.h5 文件读取反应物-生成物结构对（原始一一对应关系）
    
    参数:
        h5_file_path: HDF5 文件路径
    
    返回:
        结构对列表，每个元素是 (反应物结构, 生成物结构, 来源) 的 MolStructure 实例元组
        注：原始数据中反应物和生成物的原子是一一对应的（正确映射为顺序索引）
    """

    data_pairs = []

    # 读取RGD1_CHNO.h5文件
    rxns= h5py.File(h5_file_path, 'r')

    for Rind,Rxn in tqdm(rxns.items(), desc="处理 RGD1 数据", total=len(rxns)):
        Rxn_source = "RGD1_" + Rind

        # parse elements
        elements = np.array([Ei for Ei in Rxn.get('elements')], dtype=np.int64)

        # parse geometries
        R_G = np.array(Rxn.get('RG'))
        P_G = np.array(Rxn.get('PG'))

        # print(f"elements: {elements}")
        # print(f"R_G: {R_G}")
        # print(f"P_G: {P_G}")
        # break

        # 创建 MolStructure 实例
        reactant_struct = MolStructure(elements, R_G)
        product_struct = MolStructure(elements, P_G)

        data_pairs.append((reactant_struct, product_struct, Rxn_source))

    print(f"成功读取 {len(data_pairs)} 个有效反应结构对（反应物-生成物一一对应）")
    return data_pairs

def load_t1x_data(h5_file_path: str) -> List[Tuple[MolStructure, MolStructure, str]]:
    """
    从 transition1x.h5 文件读取反应物-生成物结构对（原始一一对应关系）

    参数:
        h5_file_path: HDF5 文件路径
    
    返回:
        结构对列表，每个元素是 (反应物结构, 生成物结构, 来源) 的 MolStructure 实例元组
        注：原始数据中反应物和生成物的原子是一一对应的（正确映射为顺序索引）
    """

    data_pairs = []
    # 读取T1x_CHNO.h5文件
    dataloader = t1x.Dataloader(h5_file_path, only_final=True)
    total_molecules = sum(1 for _ in dataloader)
    
    for idx, molecule in tqdm(enumerate(dataloader, 1), desc="处理 T1x 数据", total=total_molecules):
        Rxn_source = f"T1x_{molecule.get('rxn')}"
        elements = np.array(molecule["reactant"].get("atomic_numbers"), dtype=np.int64)
        R_G = np.array(molecule["reactant"].get("positions"))
        P_G = np.array(molecule["product"].get("positions"))

        # 创建 MolStructure 实例
        reactant_struct = MolStructure(elements, R_G)
        product_struct = MolStructure(elements, P_G)

        data_pairs.append((reactant_struct, product_struct, Rxn_source))

    print(f"成功读取 {len(data_pairs)} 个有效反应结构对（反应物-生成物一一对应）")
    return data_pairs

def load_rdb19_rad_data(folder_path: str) -> List[Tuple[MolStructure, MolStructure, str]]:
    """
    从 Reactions 文件夹中读取反应物-生成物结构对（原始一一对应关系）
    文件夹结构为：
    Reactions/
    ├── 0
    ├──── IRC_educt.xyz
    ├──── IRC_product.xyz
    ├──── ...
    ├── 1
    ├──── IRC_educt.xyz
    ├──── IRC_product.xyz
    ├──── ...
    └── ...

    参数:
        folder_path: 包含反应编号文件夹的路径
    
    返回:
        结构对列表，每个元素是 (反应物结构, 生成物结构, 来源) 的 MolStructure 实例元组
    """

    data_pairs = []

    # 获取所有反应文件夹并过滤掉非目录项
    reaction_folders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            reaction_folders.append(item)
    
    # 遍历每个反应编号文件夹
    for reaction_id in tqdm(reaction_folders, desc="处理 RDB19 数据", total=len(reaction_folders)):
        reaction_folder = os.path.join(folder_path, reaction_id)
        Rxn_source = f"RDB19_{reaction_id}"
        reactant_file = os.path.join(reaction_folder, "IRC_educt.xyz")
        product_file = os.path.join(reaction_folder, "IRC_product.xyz")
            
        # 使用ase读取xyz文件
        reactant_atoms = read(reactant_file)
        product_atoms = read(product_file)
        
        # 提取元素信息（使用原子序数）
        reactant_elements = np.array([atom.number for atom in reactant_atoms], dtype=np.int64)
        product_elements = np.array([atom.number for atom in product_atoms], dtype=np.int64)
        
        # 提取坐标信息
        reactant_positions = reactant_atoms.get_positions()
        product_positions = product_atoms.get_positions()
        
        # 创建 MolStructure 实例
        reactant_struct = MolStructure(reactant_elements, reactant_positions)
        product_struct = MolStructure(product_elements, product_positions)
                
        data_pairs.append((reactant_struct, product_struct, Rxn_source))

    print(f"成功读取 {len(data_pairs)} 个有效反应结构对（反应物-生成物一一对应）")
    return data_pairs


# --------------------------
# 2. 生成原子映射信息
# --------------------------
def create_test_atom_mapping(
    ref_struct: MolStructure,
    target_struct: MolStructure,
    source: str,
    random_seed: Optional[int] = None
) -> AtomMapping:
    """
    创建测试用的 AtomMapping 实例：
    - 打乱目标结构（target_struct）的原子顺序，得到候选结构
    - 正确映射：候选结构 → 参考结构的索引（标准答案）
    - 参考结构保持不变，候选结构为打乱后的目标结构
    
    参数:
        ref_struct: 参考结构（如反应物）
        target_struct: 原始目标结构（如生成物，与参考结构一一对应）
        source: 来源
        random_seed: 随机种子（可复现）
    
    返回:
        AtomMapping 实例：包含（参考结构, 打乱后的候选结构, 正确映射）
    """
    
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 步骤1：生成随机打乱索引（打乱原始目标结构的原子顺序）
    # shuffled_idx：原始目标结构的原子索引 → 打乱后的候选结构的原子索引
    n_atoms = len(target_struct.atoms)
    shuffled_idx = np.random.permutation(n_atoms)
    
    # 步骤2：创建打乱后的候选结构（目标结构打乱后作为待重排序的候选）
    candidate_struct = MolStructure(
        atoms=target_struct.atoms[shuffled_idx],
        coordinates=target_struct.coordinates[shuffled_idx]
    )
    
    # 步骤3：计算正确映射（标准答案）
    # 正确映射定义：candidate_struct 的第 i 个原子 → 参考结构的第 correct_mapping[i] 个原子
    # 即：使用 correct_mapping 重排候选结构可使其原子顺序对应到参考结构
    # 创建逆映射：对于每个原子位置i，找到在shuffled_idx中哪个位置是i
    correct_mapping = np.zeros_like(shuffled_idx)
    correct_mapping[shuffled_idx] = np.arange(n_atoms)
    
    # 步骤4：验证正确性 - 演示如何使用正确映射将候选结构映射回原始结构
    # 这确保了candidate_struct经过correct_mapping映射后能得到原始target_struct
    recovered_struct = MolStructure(
        atoms=candidate_struct.atoms[correct_mapping],
        coordinates=candidate_struct.coordinates[correct_mapping]
    )
    assert np.array_equal(recovered_struct.atoms, target_struct.atoms), "原子映射验证失败（原子不匹配）"
    assert np.array_equal(recovered_struct.atoms, ref_struct.atoms), "原子映射验证失败（原子不匹配）"
    assert np.allclose(recovered_struct.coordinates, target_struct.coordinates), "原子映射验证失败（坐标不匹配）"
    
    # 步骤5：创建 AtomMapping 实例（参考结构 + 候选结构 + 正确映射）
    # mapping_indices表示candidate_struct到ref_struct的映射关系
    return AtomMapping(
        structure_ref=ref_struct,
        structure_cand=candidate_struct,
        mapping_indices=correct_mapping,
        source=source
    )

# --------------------------
# 3. 批量生成测试数据集
# --------------------------
def generate_mapping_dataset(
    data_pairs: List[Tuple[MolStructure, MolStructure, str]],
    output_dir: str = "../data/ready",
    n_test_samples: int = 10000,
    random_seed: int = 42,
) -> str:
    """
    生成原子映射测试数据集
    
    参数:
        data_pairs: 读取的结构对列表
        output_dir: 输出目录
        n_test_samples: 测试样本数量
        random_seed: 随机种子
    
    返回:
        输出文件路径
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择测试样本
    if len(data_pairs) > n_test_samples:
        np.random.seed(random_seed)
        test_indices = np.random.choice(len(data_pairs), size=n_test_samples, replace=False)
        selected_pairs = [data_pairs[i] for i in test_indices]
    else:
        selected_pairs = data_pairs
        print(f"使用全部 {len(selected_pairs)} 个样本")
    
    # 批量创建 AtomMapping 测试样本
    test_mappings = []
    print(f"\n正在生成测试数据...")
    
    #########################
    for idx, (reactant_struct, product_struct, source) in tqdm(enumerate(selected_pairs), total=len(selected_pairs)):
        
        # 确定参考结构和原始目标结构
        ref_struct = reactant_struct
        target_struct = product_struct
        
        # 创建 AtomMapping 实例
        atom_mapping = create_test_atom_mapping(
            ref_struct=ref_struct,
            target_struct=target_struct,
            source=source,
            random_seed=random_seed + idx
        )

        # 添加到测试映射列表
        test_mappings.append(atom_mapping)
    
    # 保存测试数据集
    dataset_name = source.split('_')[0].lower()  # 从source提取数据集名称
    output_path = os.path.join(output_dir, f"{dataset_name}_dataset.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(test_mappings, f)
    
    print(f"测试数据集已保存至：{output_path}")
    print(f"共生成 {len(test_mappings)} 个 AtomMapping 测试样本")
    
    return output_path

# --------------------------
# 主函数：加载数据并生成映射
# --------------------------
if __name__ == "__main__":
    # 加载不同的数据集
    print("=== 开始加载数据集 ===")
    rgd1_data = load_rgd1_data("../../data/raw/RGD1_CHNO.h5")
    t1x_data = load_t1x_data("../../data/raw/transition1x.h5")
    rdb19_data = load_rdb19_rad_data("../../data/raw/Reactions")
    
    # 为每个数据集生成原子映射信息
    print("\n=== 开始生成原子映射数据集 ===")
    
    # 配置参数
    OUTPUT_DIR = "../data/ready"
    N_TEST_SAMPLES = np.inf
    RANDOM_SEED = 42
    
    # 生成RGD1数据集的原子映射
    print("\n生成 RGD1 数据集的原子映射...")
    rgd1_output_path = generate_mapping_dataset(
        data_pairs=rgd1_data,
        output_dir=OUTPUT_DIR,
        n_test_samples=N_TEST_SAMPLES,
        random_seed=RANDOM_SEED,
    )
    
    # 生成T1x数据集的原子映射
    print("\n生成 T1x 数据集的原子映射...")
    t1x_output_path = generate_mapping_dataset(
        data_pairs=t1x_data,
        output_dir=OUTPUT_DIR,
        n_test_samples=N_TEST_SAMPLES,
        random_seed=RANDOM_SEED,
    )
    
    # 生成RDB19数据集的原子映射
    print("\n生成 RDB19 数据集的原子映射...")
    rdb19_output_path = generate_mapping_dataset(
        data_pairs=rdb19_data,
        output_dir=OUTPUT_DIR,
        n_test_samples=N_TEST_SAMPLES,
        random_seed=RANDOM_SEED,
    )
    
    print("\n=== 所有数据集生成完成 ===")
    print(f"RGD1 输出路径: {rgd1_output_path}")
    print(f"T1x 输出路径: {t1x_output_path}")
    print(f"RDB19 输出路径: {rdb19_output_path}")