"""
混合损失扩散模型的多GPU评估脚本
"""
import argparse
import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.diffusion_model import DiffusionAlignmentModel
from scipy.optimize import linear_sum_assignment
import time


# 使用相同的SimpleDataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, max_atoms=50):
        self.pairs = pairs
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        if hasattr(pair, 'structure_ref') and hasattr(pair, 'structure_cand'):
            ref = pair.structure_ref
            cand = pair.structure_cand

            coords_A = torch.tensor(ref.coordinates, dtype=torch.float32)
            types_A = torch.tensor(ref.atoms, dtype=torch.long)
            coords_B = torch.tensor(cand.coordinates, dtype=torch.float32)
            types_B = torch.tensor(cand.atoms, dtype=torch.long)

            mapping = pair.mapping_indices
            n_atoms = len(mapping)
            perm_matrix = torch.zeros(n_atoms, n_atoms, dtype=torch.float32)
            for i, j in enumerate(mapping):
                perm_matrix[i, j] = 1.0
        else:
            coords_A = torch.tensor(pair.get('coords_A', pair.get('coords_ref')), dtype=torch.float32)
            types_A = torch.tensor(pair.get('types_A', pair.get('atom_types_ref')), dtype=torch.long)
            coords_B = torch.tensor(pair.get('coords_B', pair.get('coords_cand')), dtype=torch.float32)
            types_B = torch.tensor(pair.get('types_B', pair.get('atom_types_cand')), dtype=torch.long)
            perm_matrix = torch.tensor(pair.get('perm_matrix', pair.get('alignment')), dtype=torch.float32)

        n_atoms = min(coords_A.size(0), self.max_atoms)

        coords_A_pad = torch.zeros(self.max_atoms, 3)
        coords_B_pad = torch.zeros(self.max_atoms, 3)
        types_A_pad = torch.zeros(self.max_atoms, dtype=torch.long)
        types_B_pad = torch.zeros(self.max_atoms, dtype=torch.long)
        perm_pad = torch.zeros(self.max_atoms, self.max_atoms)
        mask = torch.zeros(self.max_atoms)

        coords_A_pad[:n_atoms] = coords_A[:n_atoms]
        coords_B_pad[:n_atoms] = coords_B[:n_atoms]
        types_A_pad[:n_atoms] = types_A[:n_atoms]
        types_B_pad[:n_atoms] = types_B[:n_atoms]
        perm_pad[:n_atoms, :n_atoms] = perm_matrix[:n_atoms, :n_atoms]
        mask[:n_atoms] = 1

        return {
            'coords_A': coords_A_pad,
            'types_A': types_A_pad,
            'coords_B': coords_B_pad,
            'types_B': types_B_pad,
            'perm_matrix': perm_pad,
            'mask': mask
        }


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'  # 使用新端口避免冲突
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def load_model(checkpoint_path, device, rank):
    """加载混合损失扩散模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = DiffusionAlignmentModel(
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        num_cross_layers=config['num_cross_layers'],
        num_heads=config['num_heads'],
        num_timesteps=config['num_timesteps']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def create_val_loader_ddp(val_paths, batch_size, num_workers, rank, world_size):
    val_data = []
    for path in val_paths:
        if rank == 0:
            print(f"Loading validation data from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            val_data.extend(data)

    if rank == 0:
        print(f"Total validation samples: {len(val_data)}")

    val_dataset = SimpleDataset(val_data)

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return val_loader, val_sampler


def compute_alignment_metrics(pred_perm, gt_perm, mask):
    """计算对齐指标"""
    B, N, _ = pred_perm.shape
    accuracies = []
    hamming_distances = []

    for i in range(B):
        n_atoms = int(mask[i].sum().item())
        if n_atoms == 0:
            continue

        matches = (pred_perm[i, :n_atoms, :n_atoms] * gt_perm[i, :n_atoms, :n_atoms]).sum()
        accuracy = matches.item() / n_atoms
        accuracies.append(accuracy)

        hamming = n_atoms - matches.item()
        hamming_distances.append(hamming)

    return accuracies, hamming_distances


@torch.no_grad()
def evaluate_model_ddp(model, val_loader, device, rank, world_size, num_sampling_steps=50):
    """评估混合损失扩散模型"""
    model.eval()

    local_accuracies = []
    local_hamming_distances = []
    local_perfect_matches = 0
    local_total_samples = 0
    local_inference_times = []

    if rank == 0:
        pbar = tqdm(val_loader, desc=f'Evaluating (Hybrid Diffusion, {num_sampling_steps} steps)')
    else:
        pbar = val_loader

    for batch in pbar:
        coords_A = batch['coords_A'].to(device)
        types_A = batch['types_A'].to(device)
        coords_B = batch['coords_B'].to(device)
        types_B = batch['types_B'].to(device)
        perm_matrix = batch['perm_matrix'].to(device)
        mask = batch['mask'].to(device)

        B = coords_A.size(0)

        # 记录推理时间（包括采样）
        torch.cuda.synchronize()
        start_time = time.time()

        # 扩散采样
        alignment_scores = model.sample(coords_A, types_A, coords_B, types_B, mask, num_steps=num_sampling_steps)

        torch.cuda.synchronize()
        inference_time = (time.time() - start_time) * 1000
        time_per_sample = inference_time / B
        local_inference_times.extend([time_per_sample] * B)

        N = alignment_scores.shape[1]

        # Hungarian算法
        pred_perms = torch.zeros_like(alignment_scores)
        for i in range(B):
            n_atoms = int(mask[i].sum().item())
            if n_atoms == 0:
                continue

            cost_matrix = -alignment_scores[i, :n_atoms, :n_atoms].cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            pred_perms[i, row_ind, col_ind] = 1.0

        # 计算指标
        batch_accuracies, batch_hamming = compute_alignment_metrics(pred_perms, perm_matrix, mask)
        local_accuracies.extend(batch_accuracies)
        local_hamming_distances.extend(batch_hamming)

        local_perfect_matches += sum(1 for acc in batch_accuracies if acc == 1.0)
        local_total_samples += len(batch_accuracies)

    # 收集所有GPU的数据
    local_acc_tensor = torch.tensor(local_accuracies, dtype=torch.float32, device=device)
    local_hamming_tensor = torch.tensor(local_hamming_distances, dtype=torch.float32, device=device)
    local_time_tensor = torch.tensor(local_inference_times, dtype=torch.float32, device=device)
    local_stats = torch.tensor([local_perfect_matches, local_total_samples, len(local_accuracies)],
                               dtype=torch.float32, device=device)

    data_lengths = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(data_lengths, torch.tensor([len(local_accuracies)], dtype=torch.long, device=device))

    dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

    if rank == 0:
        max_length = max([l.item() for l in data_lengths])

        # 收集所有数据
        padded_local_acc = torch.zeros(max_length, dtype=torch.float32, device=device)
        padded_local_acc[:len(local_accuracies)] = local_acc_tensor
        all_accuracies_list = [torch.zeros(max_length, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.gather(padded_local_acc, all_accuracies_list if rank == 0 else None, dst=0)

        padded_local_hamming = torch.zeros(max_length, dtype=torch.float32, device=device)
        padded_local_hamming[:len(local_hamming_distances)] = local_hamming_tensor
        all_hamming_list = [torch.zeros(max_length, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.gather(padded_local_hamming, all_hamming_list if rank == 0 else None, dst=0)

        padded_local_time = torch.zeros(max_length, dtype=torch.float32, device=device)
        padded_local_time[:len(local_inference_times)] = local_time_tensor
        all_time_list = [torch.zeros(max_length, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.gather(padded_local_time, all_time_list if rank == 0 else None, dst=0)

        all_acc = []
        all_hamming = []
        all_times = []
        for i in range(world_size):
            length = data_lengths[i].item()
            all_acc.extend(all_accuracies_list[i][:length].cpu().numpy().tolist())
            all_hamming.extend(all_hamming_list[i][:length].cpu().numpy().tolist())
            all_times.extend(all_time_list[i][:length].cpu().numpy().tolist())

        all_accuracies = np.array(all_acc)
        all_hamming_distances = np.array(all_hamming)
        all_inference_times = np.array(all_times)
        perfect_matches = int(local_stats[0].item())
        total_samples = int(local_stats[1].item())

        results = {
            'method': 'hybrid_diffusion',
            'num_sampling_steps': num_sampling_steps,
            'total_samples': total_samples,
            'mean_accuracy': float(all_accuracies.mean()),
            'median_accuracy': float(np.median(all_accuracies)),
            'std_accuracy': float(all_accuracies.std()),
            'min_accuracy': float(all_accuracies.min()),
            'max_accuracy': float(all_accuracies.max()),
            'perfect_match_count': perfect_matches,
            'perfect_match_rate': perfect_matches / total_samples if total_samples > 0 else 0,
            'mean_hamming': float(all_hamming_distances.mean()),
            'median_hamming': float(np.median(all_hamming_distances)),
            'std_hamming': float(all_hamming_distances.std()),
            'min_hamming': float(all_hamming_distances.min()),
            'max_hamming': float(all_hamming_distances.max()),
            'mean_inference_time_ms': float(all_inference_times.mean()),
            'median_inference_time_ms': float(np.median(all_inference_times)),
            'std_inference_time_ms': float(all_inference_times.std()),
            'min_inference_time_ms': float(all_inference_times.min()),
            'max_inference_time_ms': float(all_inference_times.max()),
            'accuracy_bins': {
                '0.0-0.2': int((all_accuracies < 0.2).sum()),
                '0.2-0.4': int(((all_accuracies >= 0.2) & (all_accuracies < 0.4)).sum()),
                '0.4-0.6': int(((all_accuracies >= 0.4) & (all_accuracies < 0.6)).sum()),
                '0.6-0.8': int(((all_accuracies >= 0.6) & (all_accuracies < 0.8)).sum()),
                '0.8-1.0': int(((all_accuracies >= 0.8) & (all_accuracies < 1.0)).sum()),
                '1.0': perfect_matches
            }
        }
        return results
    else:
        max_length = max([l.item() for l in data_lengths])
        padded_local_acc = torch.zeros(max_length, dtype=torch.float32, device=device)
        padded_local_acc[:len(local_accuracies)] = local_acc_tensor
        dist.gather(padded_local_acc, None, dst=0)

        padded_local_hamming = torch.zeros(max_length, dtype=torch.float32, device=device)
        padded_local_hamming[:len(local_hamming_distances)] = local_hamming_tensor
        dist.gather(padded_local_hamming, None, dst=0)

        padded_local_time = torch.zeros(max_length, dtype=torch.float32, device=device)
        padded_local_time[:len(local_inference_times)] = local_time_tensor
        dist.gather(padded_local_time, None, dst=0)
        return None


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print(f"Evaluation Results (Hybrid Diffusion Model)")
    print(f"Sampling steps: {results['num_sampling_steps']}")
    print("="*60)
    print(f"Total samples: {results['total_samples']}")

    print(f"\nAccuracy Statistics:")
    print(f"  Mean:   {results['mean_accuracy']:.4f}")
    print(f"  Median: {results['median_accuracy']:.4f}")
    print(f"  Std:    {results['std_accuracy']:.4f}")
    print(f"  Min:    {results['min_accuracy']:.4f}")
    print(f"  Max:    {results['max_accuracy']:.4f}")

    print(f"\nHamming Distance Statistics:")
    print(f"  Mean:   {results['mean_hamming']:.2f}")
    print(f"  Median: {results['median_hamming']:.2f}")
    print(f"  Std:    {results['std_hamming']:.2f}")
    print(f"  Min:    {results['min_hamming']:.0f}")
    print(f"  Max:    {results['max_hamming']:.0f}")

    print(f"\nInference Time Statistics (ms/mol):")
    print(f"  Mean:   {results['mean_inference_time_ms']:.3f}")
    print(f"  Median: {results['median_inference_time_ms']:.3f}")
    print(f"  Std:    {results['std_inference_time_ms']:.3f}")
    print(f"  Min:    {results['min_inference_time_ms']:.3f}")
    print(f"  Max:    {results['max_inference_time_ms']:.3f}")

    print(f"\nPerfect Matches:")
    print(f"  Count: {results['perfect_match_count']}")
    print(f"  Rate:  {results['perfect_match_rate']:.2%}")

    print(f"\nAccuracy Distribution:")
    for bin_range, count in results['accuracy_bins'].items():
        percentage = count / results['total_samples'] * 100
        print(f"  {bin_range}: {count:5d} ({percentage:5.2f}%)")
    print("="*60 + "\n")


def eval_worker(rank, world_size, config):
    """每个进程的评估函数"""
    setup_ddp(rank, world_size)

    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\nLoading model from {config['checkpoint_path']}...")

    model, model_config = load_model(config['checkpoint_path'], device, rank)

    if rank == 0:
        print(f"Model loaded successfully")
        print(f"Model config: {model_config}")
        if 'align_loss_weight' in model_config:
            print(f"Alignment loss weight: {model_config['align_loss_weight']}")

    # 对每个数据集单独评估
    all_results = {}

    for val_path in config['val_paths']:
        dataset_name = os.path.splitext(os.path.basename(val_path))[0]

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating on dataset: {dataset_name}")
            print(f"{'='*60}")

        # 创建当前数据集的验证数据加载器
        val_loader, val_sampler = create_val_loader_ddp(
            [val_path],
            config['batch_size'],
            config['num_workers'],
            rank,
            world_size
        )

        # 评估
        if rank == 0:
            print(f"\nEvaluating {dataset_name} with {config['num_sampling_steps']} sampling steps...")

        results = evaluate_model_ddp(model, val_loader, device, rank, world_size, config['num_sampling_steps'])

        if rank == 0:
            print(f"\n--- Results for {dataset_name} ---")
            print_results(results)

            # 添加数据集名称到结果
            results['dataset'] = dataset_name
            all_results[dataset_name] = results

    # 保存所有结果
    if rank == 0:
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # 保存每个数据集的单独结果
        for dataset_name, results in all_results.items():
            results_path = os.path.join(output_dir, f'hybrid_diffusion_evaluation_{dataset_name}.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults for {dataset_name} saved to {results_path}")

        # 保存汇总结果
        summary_path = os.path.join(output_dir, 'hybrid_diffusion_evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary results saved to {summary_path}")

    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid diffusion model DDP evaluation")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "ready"),
        help="Base directory for default validation datasets",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs_diffusion_hybrid"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to model checkpoint (default: <output-dir>/best_diffusion_model.pt)",
    )
    parser.add_argument(
        "--val-paths",
        nargs="+",
        default=None,
        help="Validation dataset paths (default: <data-dir>/rgd1_val.pkl <data-dir>/t1x_val.pkl)",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=50,
        help="DDIM sampling steps",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path or os.path.join(output_dir, "best_diffusion_model.pt")
    val_paths = args.val_paths or [
        os.path.join(data_dir, "rgd1_val.pkl"),
        os.path.join(data_dir, "t1x_val.pkl"),
    ]
    config = {
        "checkpoint_path": checkpoint_path,
        "val_paths": val_paths,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_sampling_steps": args.num_sampling_steps,  # DDIM采样步数
        "output_dir": output_dir,
    }

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    print(f"Using device: cuda")

    if world_size < 2:
        print("Warning: Multi-GPU evaluation works best with 2+ GPUs")
        world_size = 1

    torch.multiprocessing.spawn(
        eval_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
