# test_hungarian_algorithm.py
import numpy as np
import pickle
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm

# 添加项目根目录到Python路径
import sys
sys.path.append("/Users/misixuan/Desktop/codingwithai/rmsd_test")

# 导入必要的模块
from src.utils import MolStructure, AtomMapping
from src.reorder_inertia_hungarian import reorder_inertia_hungarian


def validate_reorder_function(
    test_mappings_path: str,
    verbose: bool = False
) -> Dict[str, float]:
    """
    使用测试数据集验证 inertia_hungarian.py 中的重排序函数
    核心：对比预测映射与 AtomMapping 中的正确映射（标准答案）
    
    参数:
        test_mappings_path: 测试数据集路径（xxx_dataset.pkl）
        verbose: 是否打印每个样本的详细信息
    
    返回:
        验证统计结果（正确率、平均RMSD等）
    """
    # 加载测试数据集（AtomMapping 实例列表）
    try:
        with open(test_mappings_path, 'rb') as f:
            test_mappings = pickle.load(f)
        print(f"成功加载数据集：{os.path.basename(test_mappings_path)}")
    except Exception as e:
        print(f"加载数据集失败：{e}")
        return {}
    
    print("=" * 70)
    print(f"开始验证重排序函数 | 测试样本数：{len(test_mappings)}")
    print("=" * 70)
    
    # 统计变量
    total_samples = len(test_mappings)
    correct_samples = 0
    # rmsd_list = []
    # symmetry_mismatch_samples = 0  # 因对称性导致索引不匹配但RMSD接近0的样本
    error_samples = 0  # 处理出错的样本数
    
    for idx, atom_mapping in enumerate(tqdm(test_mappings, desc="处理样本")):
        try:
            # 从 AtomMapping 中提取数据
            ref_struct = atom_mapping.structure_ref
            cand_struct = atom_mapping.structure_cand
            correct_mapping = atom_mapping.mapping_indices
            sample_id = f"test_sample_{idx:05d}"
            
            # 调用重排序函数预测映射
            predicted_mapping = reorder_inertia_hungarian(ref_struct, cand_struct)
            
            # 验证1：索引完全匹配（严格正确）
            is_strict_correct = np.array_equal(predicted_mapping, correct_mapping)
            
            # # 验证2：RMSD验证（允许对称性等价映射）
            # # 使用预测映射重排候选结构
            # predicted_cand = MolStructure(
            #     atoms=cand_struct.atoms[predicted_mapping],
            #     coordinates=cand_struct.coordinates[predicted_mapping]
            # )
            # # 计算与参考结构的RMSD（参考结构和候选结构应先中心化，避免平移影响）
            # ref_centered = ref_struct.coordinates - np.mean(ref_struct.coordinates, axis=0)
            # pred_centered = predicted_cand.coordinates - np.mean(predicted_cand.coordinates, axis=0)
            # rmsd = np.sqrt(np.mean((ref_centered - pred_centered)**2))
            # rmsd_list.append(rmsd)
            
            # # 判定：RMSD < 1e-3 Å 视为等价正确（对称性导致索引不同）
            # is_equivalent_correct = rmsd < 1e-3
            
            # 统计
            if is_strict_correct:
                correct_samples += 1
                status = "✅ 严格正确"
            # elif is_equivalent_correct:
                # symmetry_mismatch_samples += 1
                # status = "⚠️  等价正确（对称性）"
            else:
                status = "❌ 错误"
            
            # 打印详细信息
            if verbose:
                source_info = atom_mapping.source if hasattr(atom_mapping, 'source') else 'Unknown'
                print(f"[{sample_id}] [{source_info}] {status} | 原子数：{len(ref_struct.atoms)} | RMSD：{rmsd:.6f} Å")
                if not is_strict_correct and not is_equivalent_correct:
                    print(f"  - 正确映射：{correct_mapping}")
                    print(f"  - 预测映射：{predicted_mapping}")
        except Exception as e:
            error_samples += 1
            print(f"[{sample_id}] ❌ 处理出错: {e}")
    
    # 计算统计结果
    valid_samples = total_samples - error_samples
    if valid_samples > 0:
        strict_accuracy = correct_samples / valid_samples * 100
        # total_correct_rate = (correct_samples + symmetry_mismatch_samples) / valid_samples * 100
        # avg_rmsd = np.mean(rmsd_list) if rmsd_list else 0.0
        # std_rmsd = np.std(rmsd_list) if rmsd_list else 0.0
    else:
        strict_accuracy = 0.0
        # total_correct_rate = 0.0
        # avg_rmsd = 0.0
        # std_rmsd = 0.0
    
    # 输出统计摘要
    print("=" * 70)
    print("验证结果统计：")
    print(f"总样本数：{total_samples}")
    print(f"有效样本数：{valid_samples}")
    print(f"出错样本数：{error_samples}")
    print(f"严格正确率（索引完全匹配）：{correct_samples}/{valid_samples} ({strict_accuracy:.2f}%)")
    # print(f"等价正确率（RMSD < 1e-3 Å）：{correct_samples + symmetry_mismatch_samples}/{valid_samples} ({total_correct_rate:.1f}%)")
    # print(f"因对称性导致索引不匹配的样本数：{symmetry_mismatch_samples}")
    # print(f"平均RMSD：{avg_rmsd:.6f} Å（标准差：{std_rmsd:.6f}）")
    print("=" * 70)
    
    return {
        "dataset": os.path.basename(test_mappings_path),
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "error_samples": error_samples,
        "strict_accuracy": strict_accuracy,
        # "total_correct_rate": total_correct_rate,
        # "avg_rmsd": avg_rmsd,
        # "std_rmsd": std_rmsd,
        # "symmetry_mismatch_count": symmetry_mismatch_samples
    }


def find_all_test_datasets(dataset_dir: str = "../data/ready") -> List[str]:
    """
    查找指定目录下所有的测试数据集文件
    
    参数:
        dataset_dir: 数据集目录路径
    
    返回:
        数据集文件路径列表
    """
    # 获取绝对路径
    abs_dir = os.path.abspath(dataset_dir)
    
    # 检查目录是否存在
    if not os.path.exists(abs_dir):
        print(f"警告：目录不存在：{abs_dir}")
        return []
    
    # 查找所有 .pkl 文件
    dataset_files = []
    for file in os.listdir(abs_dir):
        if file.endswith(".pkl"):
            dataset_files.append(os.path.join(abs_dir, file))
    
    # 按文件名排序
    dataset_files.sort()
    
    return dataset_files


def run_batch_validation(
    dataset_dir: str = "../data/ready",
    verbose: bool = False,
    output_summary: bool = True
) -> List[Dict[str, float]]:
    """
    批量验证所有测试数据集
    
    参数:
        dataset_dir: 数据集目录路径
        verbose: 是否打印每个样本的详细信息
        output_summary: 是否输出汇总结果
    
    返回:
        所有数据集的验证结果列表
    """
    # 查找所有测试数据集
    dataset_files = find_all_test_datasets(dataset_dir)
    
    if not dataset_files:
        print("未找到任何测试数据集文件")
        return []
    
    print(f"找到 {len(dataset_files)} 个测试数据集文件：")
    for file in dataset_files:
        print(f"  - {os.path.basename(file)}")
    
    # 批量验证
    all_results = []
    print("\n" + "="*80)
    print("开始批量验证")
    print("="*80)
    
    for file_path in dataset_files:
        print(f"\n验证数据集：{os.path.basename(file_path)}")
        result = validate_reorder_function(file_path, verbose=verbose)
        if result:
            all_results.append(result)
    
    # 输出汇总结果
    if output_summary and all_results:
        print("\n" + "="*80)
        print("批量验证汇总结果")
        print("="*80)
        print(f"{'数据集名称':<30} {'严格正确率':<12} {'有效样本数':<10}")
        print("-"*80)
        
        for result in all_results:
            print(f"{result['dataset']:<30} {result['strict_accuracy']:12.2f}% {result['valid_samples']:10d}")
            # print(f"{result['dataset']:<30} {result['strict_accuracy']:8.1f}% {result['total_correct_rate']:8.1f}% {result['avg_rmsd']:10.6f} Å {result['valid_samples']:10d}")
        
        # 计算总体平均
        avg_strict_acc = np.mean([r['strict_accuracy'] for r in all_results])
        # avg_total_acc = np.mean([r['total_correct_rate'] for r in all_results])
        # avg_rmsd = np.mean([r['avg_rmsd'] for r in all_results])
        total_valid_samples = sum([r['valid_samples'] for r in all_results])
        
        print("="*80)
        print(f"{'总体平均':<30} {avg_strict_acc:12.2f}% {total_valid_samples:10d}")
        # print(f"{'总体平均':<30} {avg_strict_acc:8.1f}% {avg_total_acc:8.1f}% {avg_rmsd:10.6f} Å {total_valid_samples:10d}")
    
    return all_results


if __name__ == "__main__":
    # 默认参数
    DATASET_DIR = "../data/ready"
    VERBOSE = False  # 是否打印每个样本的详细信息
    OUTPUT_SUMMARY = True  # 是否输出汇总结果
    
    # 运行批量验证
    print("原子重排序算法（inertia_hungarian）验证工具")
    print("=" * 60)
    
    results = run_batch_validation(DATASET_DIR, verbose=VERBOSE, output_summary=OUTPUT_SUMMARY)
    
    # 保存验证结果到文件
    if results:
        output_file = "../results/inertia_hungarian_validation_results.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("原子重排序算法（inertia_hungarian）验证结果\n")
            f.write("=" * 80 + "\n\n")
            
            for result in results:
                f.write(f"数据集: {result['dataset']}\n")
                f.write(f"总样本数: {result['total_samples']}\n")
                f.write(f"有效样本数: {result['valid_samples']}\n")
                f.write(f"出错样本数: {result['error_samples']}\n")
                f.write(f"严格正确率: {result['strict_accuracy']:.2f}%\n")
                # f.write(f"等价正确率: {result['total_correct_rate']:.1f}%\n")
                # f.write(f"平均RMSD: {result['avg_rmsd']:.6f} Å\n")
                # f.write(f"因对称性导致索引不匹配的样本数: {result['symmetry_mismatch_count']}\n")
                f.write("\n" + "-" * 60 + "\n\n")
        
        print(f"\n验证结果已保存到: {output_file}")
