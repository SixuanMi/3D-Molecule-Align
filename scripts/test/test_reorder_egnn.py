import numpy as np
import pickle
import os
import sys
import time
import torch
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import MolStructure, AtomMapping
from scripts.train.egnn import load_trained_model, EGNNAlignmentModel, MoleculeEGNN, EGNNLayer

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.optimize import linear_sum_assignment

# ==========================================
# 绘图工具函数
# ==========================================
def calculate_rigid_transform(A, B):
    """SVD 计算刚体变换: B -> A"""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(BB.T, AA)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_A - np.dot(centroid_B, R.T)
    return R, t

def visualize_alignment(ref_coords, cand_coords, pred_mapping, atom_types, save_path, sample_id):
    """生成 3D 分子对齐图"""
    reordered_cand = cand_coords[pred_mapping]
    R, t = calculate_rigid_transform(ref_coords, reordered_cand)
    aligned_cand = np.dot(reordered_cand, R.T) + t
    diff = ref_coords - aligned_cand
    rmsd = np.sqrt((diff * diff).sum(axis=1).mean())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_types = np.unique(atom_types)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: colors[i] for i, t in enumerate(unique_types)}
    point_colors = [type_to_color[t] for t in atom_types]

    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2], 
               c=point_colors, marker='o', s=80, label='Ref (Truth)', alpha=0.4, edgecolors='k')
    ax.scatter(aligned_cand[:, 0], aligned_cand[:, 1], aligned_cand[:, 2], 
               c=point_colors, marker='x', s=100, linewidth=2, label='Pred (Aligned)')

    for i in range(len(ref_coords)):
        ax.plot([ref_coords[i, 0], aligned_cand[i, 0]],
                [ref_coords[i, 1], aligned_cand[i, 1]],
                [ref_coords[i, 2], aligned_cand[i, 2]], 'k--', alpha=0.15)

    ax.set_title(f"Sample ID: {sample_id} | RMSD: {rmsd:.4f}", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_validation_summary(results, save_dir):
    """绘制性能对比汇总图"""
    if not results: return
    os.makedirs(save_dir, exist_ok=True)
    
    datasets = [r['dataset'].replace('_dataset.pkl', '') for r in results]
    accs = [r['strict_accuracy'] for r in results]
    hams = [r['avg_hamming_distance'] for r in results]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    bars1 = ax1.bar(x - width/2, accs, width, label='Accuracy (%)', color='#2ca02c', alpha=0.85, edgecolor='black')
    ax1.set_ylabel('Strict Accuracy (%)', color='#2ca02c', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, hams, width, label='Hamming Dist', color='#d62728', alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Avg Hamming Distance', color='#d62728', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(hams) * 1.5 if hams else 10)
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=12)
    
    plt.title("Model Performance by Dataset (Validation Set)", fontsize=14, y=1.15)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'validation_summary.png')
    plt.savefig(save_path, dpi=300)
    print(f"📊 性能汇总图已保存: {save_path}")
    plt.close()

def plot_rmsd_distribution(all_rmsds, save_dir):
    """绘制 RMSD 分布直方图"""
    if not all_rmsds: return
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_rmsds, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of RMSD (Validation Set)", fontsize=14)
    plt.xlabel("RMSD (Å)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'rmsd_distribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"📊 RMSD 分布图已保存: {save_path}")
    plt.close()

def process_pkl_data_with_labels(pkl_files):
    data_list = []
    label_list = []
    for file_path in pkl_files:
        dataset_label = os.path.basename(file_path)
        try:
            with open(file_path, 'rb') as f:
                mappings = pickle.load(f)
            for m in mappings:
                try:
                    ref = m.structure_ref
                    cand = m.structure_cand
                    data = Data(
                        x=torch.tensor(ref.atoms, dtype=torch.long),
                        pos=torch.tensor(ref.coordinates, dtype=torch.float),
                        h_t=torch.tensor(cand.atoms, dtype=torch.long),
                        pos_t=torch.tensor(cand.coordinates, dtype=torch.float),
                        mapping=torch.tensor(m.mapping_indices, dtype=torch.long)
                    )
                    data_list.append(data)
                    label_list.append(dataset_label)
                except Exception:
                    continue
        except Exception:
            continue
    return data_list, label_list

def load_validation_files(data_dir):
    pkl_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("_val.pkl")
    ]
    return sorted(pkl_files)

# ==========================================
# 验证逻辑
# ==========================================
def min_swap_count(correct_mapping, predicted_mapping):
    if not np.array_equal(np.sort(correct_mapping), np.sort(predicted_mapping)):
        return len(correct_mapping)
    sorted_correct = np.sort(correct_mapping)
    correct_idx = np.argsort(correct_mapping)
    predicted_sorted_idx = np.searchsorted(sorted_correct, predicted_mapping)
    permutation = correct_idx[predicted_sorted_idx]
    n = len(correct_mapping)
    visited = np.zeros(n, dtype=bool)
    cycle_count = 0
    for i in range(n):
        if not visited[i]:
            cycle_count += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = permutation[j]
    return n - cycle_count

def validate_data_list(data_list, dataset_label, model, batch_size=128, device='cuda', visualize_n=3, vis_dir=None):
    if not data_list:
        return None, []
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=4)
    
    total = len(data_list)
    correct = 0
    swap_counts = []
    hamming_dists = []
    times = []
    dataset_rmsds = [] 
    
    model.eval()
    pbar = tqdm(loader, desc=f"验证")
    
    global_sample_idx = 0
    visualized_count = 0
    dataset_name = os.path.splitext(dataset_label)[0]

    for batch in pbar:
        batch = batch.to(device)
        t0 = time.time()
        with torch.no_grad():
            h_s, h_t = model(batch)
            
        ptr = batch.ptr.cpu().numpy()
        gt_mappings_all = batch.mapping.cpu().numpy()
        pos_s_all = batch.pos.cpu().numpy()
        pos_t_all = batch.pos_t.cpu().numpy()
        atom_types_s_all = batch.x.cpu().numpy()
        atom_types_t_all = batch.h_t.cpu().numpy()

        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i+1]
            curr_h_s = h_s[start:end]
            curr_h_t = h_t[start:end]
            
            similarity = torch.matmul(curr_h_s, curr_h_t.T)
            sim_np = similarity.cpu().numpy()
            
            curr_atoms_s = atom_types_s_all[start:end]
            curr_atoms_t = atom_types_t_all[start:end]
            type_mismatch = curr_atoms_s[:, None] != curr_atoms_t[None, :]
            sim_np[type_mismatch] = -1e9
            
            row_ind, col_ind = linear_sum_assignment(sim_np, maximize=True)
            pred = col_ind
            truth = gt_mappings_all[start:end]
            
            reordered_cand = pos_t_all[start:end][pred]
            R, t = calculate_rigid_transform(pos_s_all[start:end], reordered_cand)
            aligned_cand = np.dot(reordered_cand, R.T) + t
            diff = pos_s_all[start:end] - aligned_cand
            rmsd = np.sqrt((diff * diff).sum(axis=1).mean())
            dataset_rmsds.append(rmsd)

            if visualize_n > 0 and visualized_count < visualize_n and vis_dir:
                if np.array_equal(pred, truth):
                    save_path = os.path.join(vis_dir, f"{dataset_name}_ok_{global_sample_idx}.png")
                    visualize_alignment(
                        pos_s_all[start:end], pos_t_all[start:end], pred, curr_atoms_s, save_path, global_sample_idx
                    )
                    visualized_count += 1

            if np.array_equal(pred, truth):
                correct += 1
                swap_counts.append(0)
                hamming_dists.append(0)
            else:
                swap_counts.append(min_swap_count(truth, pred))
                hamming_dists.append(np.sum(pred != truth))
            
            global_sample_idx += 1

        t1 = time.time()
        batch_avg_time = (t1 - t0) / (len(ptr) - 1)
        times.extend([batch_avg_time] * (len(ptr) - 1))
        
        curr_acc = (correct / len(times)) * 100
        pbar.set_postfix({"Acc": f"{curr_acc:.2f}%"})

    valid = len(times)
    error_samples = total - valid
    avg_swap = np.mean(swap_counts) if swap_counts else 0.0
    avg_hamming = np.mean(hamming_dists) if hamming_dists else 0.0
    avg_time = np.mean(times) * 1000
    strict_acc = (correct / valid * 100) if valid > 0 else 0.0

    return {
        "dataset": dataset_label,
        "total_samples": total,
        "valid_samples": valid,
        "error_samples": error_samples,
        "correct_samples": correct,
        "strict_accuracy": strict_acc,
        "avg_swap_count": avg_swap,
        "avg_hamming_distance": avg_hamming,
        "avg_execution_time": avg_time
    }, dataset_rmsds

def run_batch_validation():
    # DATA_DIR = "data/ready"
    # MODEL_PATH = "models/best_gnn_model.pth"
    # OUTPUT_FILE = "results/gnn_validation_results.txt"
    # VIS_DIR = "results/visualizations"
    
    # if not os.path.exists(DATA_DIR) and os.path.exists("../../data/ready"):
    DATA_DIR = "../../data/ready"
    MODEL_PATH = "../../models/best_gnn_model.pth"
    OUTPUT_FILE = "../../results/val_dataset/gnn_validation_results.txt"
    VIS_DIR = "../../results/visualizations"

    print("="*60)
    print("原子重排序算法（AI-EGNN）训练验证集评估")
    print(f"模型路径: {MODEL_PATH}")
    print("="*60)
    
    os.makedirs(VIS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_trained_model(MODEL_PATH, device=device)
    except FileNotFoundError:
        print("\n!!! 错误: 未找到模型 !!!")
        return

    abs_data_dir = os.path.abspath(DATA_DIR)
    if not os.path.exists(abs_data_dir): return
    val_files = load_validation_files(abs_data_dir)
    if not val_files:
        print("\n!!! 错误: 未找到 *_val.pkl 验证数据文件 !!!")
        return

    results = []
    all_rmsds = []
    print(f"\n开始验证 {len(val_files)} 个数据集 (*_val.pkl)...")
    
    for file_path in val_files:
        data_list, label_list = process_pkl_data_with_labels([file_path])
        if not data_list:
            continue
        label = label_list[0]
        res, rmsds = validate_data_list(data_list, label, model, batch_size=128, device=device, visualize_n=3, vis_dir=VIS_DIR)
        if res: 
            results.append(res)
            all_rmsds.extend(rmsds)
            
    if results:
        print("\n" + "="*95)
        print(f"{'数据集':<30} {'准确率':<10} {'平均交换':<10} {'平均汉明':<10} {'耗时(ms)':<10}")
        print("-"*95)
        for r in results:
            print(f"{r['dataset']:<30} {r['strict_accuracy']:6.2f}%    {r['avg_swap_count']:6.2f}     {r['avg_hamming_distance']:6.2f}     {r['avg_execution_time']:6.2f}")
        print("="*95)
        
        plot_validation_summary(results, os.path.dirname(OUTPUT_FILE))
        plot_rmsd_distribution(all_rmsds, os.path.dirname(OUTPUT_FILE))
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            f.write("原子重排序算法（AI-EGNN）验证集结果 (20% Hold-out Set)\n")
            f.write(f"模型路径: {MODEL_PATH}\n")
            f.write("=" * 80 + "\n\n")
            for r in results:
                f.write(f"数据集: {r['dataset']}\n")
                f.write(f"总样本数: {r['total_samples']}\n")
                f.write(f"有效样本数: {r['valid_samples']}\n")
                f.write(f"出错样本数: {r['error_samples']}\n")
                f.write(f"正确样本数: {r['correct_samples']}\n")
                f.write(f"严格正确率: {r['strict_accuracy']:.2f}%\n")
                f.write(f"平均最少交换次数: {r['avg_swap_count']:.2f}\n")
                f.write(f"平均汉明距离: {r['avg_hamming_distance']:.2f}\n")
                f.write(f"平均执行时间: {r['avg_execution_time']:.2f} ms\n\n")
            
    print(f"\n✅ 结果已保存: {OUTPUT_FILE}")
    print(f"✅ 图表已保存: {VIS_DIR} 和 {os.path.dirname(OUTPUT_FILE)}")

if __name__ == "__main__":
    run_batch_validation()
