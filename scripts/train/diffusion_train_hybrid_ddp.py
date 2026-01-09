"""
扩散模型的多GPU训练脚本
结合扩散噪声预测损失和直接对齐损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os
import sys
import json
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.diffusion_model import DiffusionAlignmentModel


# 使用与improved_train_ddp相同的SimpleDataset
class SimpleDataset(torch.utils.data.Dataset):
    """简单的数据集类"""
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
    """初始化DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'  # 使用新端口避免冲突
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()


def create_dataloaders_ddp(train_paths, val_paths, batch_size, num_workers, rank, world_size):
    """创建DDP数据加载器"""
    train_data = []
    for path in train_paths:
        if rank == 0:
            print(f"Loading training data from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            train_data.extend(data)

    val_data = []
    for path in val_paths:
        if rank == 0:
            print(f"Loading validation data from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            val_data.extend(data)

    if rank == 0:
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4
    )

    return train_loader, val_loader, train_sampler, val_sampler


class HybridDiffusionTrainerDDP:
    """扩散模型DDP训练器（混合损失版本）"""
    def __init__(self, model, train_loader, val_loader, train_sampler, val_sampler, config, rank, world_size):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.device = torch.device(f'cuda:{rank}')
        self.model.to(self.device)

        # DDP包装
        self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['lr'] * 0.01
        )

        # 混合精度
        self.scaler = torch.amp.GradScaler('cuda') if config.get('use_amp', True) else None

        # 记录
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_noise_loss': [],
            'train_align_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_noise_loss': [],
            'val_align_loss': [],
            'val_acc': []
        }

    def compute_alignment_loss(self, pred_matrix, gt_perm, mask):
        """
        计算对齐损失（与Improved Model一致）

        pred_matrix: (B, N, N) 预测的对齐分数矩阵
        gt_perm: (B, N, N) 真实的排列矩阵
        mask: (B, N) 有效原子mask
        """
        mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)

        # 裁剪避免数值溢出
        pred_matrix = torch.clamp(pred_matrix, -50, 50)

        # 1. 交叉熵损失（主要损失）
        pred_masked = pred_matrix.masked_fill(~mask_matrix.bool(), -1e9)
        pred_log_prob = F.log_softmax(pred_masked, dim=-1)
        ce_loss = -(gt_perm * pred_log_prob * mask_matrix).sum() / (mask.sum() + 1e-8)

        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            ce_loss = torch.tensor(0.0, device=pred_matrix.device)

        # 2. 双随机矩阵约束
        pred_prob = F.softmax(pred_masked, dim=-1)
        pred_prob = pred_prob * mask_matrix

        row_sum = pred_prob.sum(dim=2)
        col_sum = pred_prob.sum(dim=1)

        row_loss = ((row_sum - 1.0) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        col_loss = ((col_sum - 1.0) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        doubly_stochastic_loss = row_loss + col_loss

        if torch.isnan(doubly_stochastic_loss) or torch.isinf(doubly_stochastic_loss):
            doubly_stochastic_loss = torch.tensor(0.0, device=pred_matrix.device)

        # 3. MSE损失
        mse_loss = ((pred_prob - gt_perm) ** 2 * mask_matrix).sum() / (mask_matrix.sum() + 1e-8)

        if torch.isnan(mse_loss) or torch.isinf(mse_loss):
            mse_loss = torch.tensor(0.0, device=pred_matrix.device)

        # 组合损失
        total_loss = ce_loss + 0.1 * doubly_stochastic_loss + 0.5 * mse_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'ds_loss': doubly_stochastic_loss.item(),
            'mse_loss': mse_loss.item()
        }

    def predict_x0_from_noise(self, noisy_alignment, noise_pred, timesteps):
        """
        从噪声预测恢复x0（干净的对齐矩阵）

        公式: x0 = (x_t - sqrt(1-α_t) * ε) / sqrt(α_t)
        """
        alphas_cumprod = self.model.module.alphas_cumprod.to(noisy_alignment.device)

        # 获取当前时间步的α_t
        sqrt_alpha_t = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[timesteps]).view(-1, 1, 1)

        # 预测x0
        pred_x0 = (noisy_alignment - sqrt_one_minus_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-8)

        return pred_x0

    def compute_accuracy(self, pred_x0, gt_perm, mask):
        """
        计算预测准确率（使用贪心匹配）

        pred_x0: (B, N, N) 预测的对齐分数矩阵
        gt_perm: (B, N, N) 真实的排列矩阵
        mask: (B, N) 有效原子mask
        """
        B, N, _ = pred_x0.shape

        # 对预测矩阵做softmax得到概率
        pred_masked = torch.clamp(pred_x0, -50, 50)
        mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)
        pred_masked = pred_masked.masked_fill(~mask_matrix.bool(), -1e9)
        pred_prob = F.softmax(pred_masked, dim=-1)

        # 贪心匹配：每行取最大值的列
        pred_perm = torch.zeros_like(pred_prob)
        pred_idx = torch.argmax(pred_prob, dim=-1)  # (B, N)
        pred_perm.scatter_(2, pred_idx.unsqueeze(-1), 1.0)

        # 计算准确率
        correct = (pred_perm * gt_perm * mask_matrix).sum()
        total = mask.sum()
        accuracy = correct / (total + 1e-8)

        return accuracy.item()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        total_loss = 0
        total_noise_loss = 0
        total_align_loss = 0
        total_acc = 0
        num_batches = 0

        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        else:
            pbar = self.train_loader

        for batch in pbar:
            coords_A = batch['coords_A'].to(self.device)
            types_A = batch['types_A'].to(self.device)
            coords_B = batch['coords_B'].to(self.device)
            types_B = batch['types_B'].to(self.device)
            perm_matrix = batch['perm_matrix'].to(self.device)
            mask = batch['mask'].to(self.device)

            B = coords_A.size(0)

            # 随机采样时间步
            timesteps = torch.randint(0, self.config['num_timesteps'], (B,), device=self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    # 添加噪声
                    noisy_alignment, noise = self.model.module.add_noise(perm_matrix, timesteps)

                    # 预测噪声
                    noise_pred = self.model(coords_A, types_A, coords_B, types_B, mask, noisy_alignment, timesteps)

                    # 1. 噪声预测MSE loss
                    mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)
                    noise_loss = F.mse_loss(noise_pred * mask_matrix, noise * mask_matrix)

                    # 2. 从噪声预测恢复x0
                    pred_x0 = self.predict_x0_from_noise(noisy_alignment, noise_pred, timesteps)

                    # 3. 对x0应用对齐损失
                    align_loss, align_details = self.compute_alignment_loss(pred_x0, perm_matrix, mask)

                    # 4. 组合损失
                    loss = noise_loss + self.config['align_loss_weight'] * align_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noisy_alignment, noise = self.model.module.add_noise(perm_matrix, timesteps)
                noise_pred = self.model(coords_A, types_A, coords_B, types_B, mask, noisy_alignment, timesteps)

                mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)
                noise_loss = F.mse_loss(noise_pred * mask_matrix, noise * mask_matrix)

                pred_x0 = self.predict_x0_from_noise(noisy_alignment, noise_pred, timesteps)
                align_loss, align_details = self.compute_alignment_loss(pred_x0, perm_matrix, mask)

                loss = noise_loss + self.config['align_loss_weight'] * align_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()

            if not (torch.isnan(loss) or torch.isinf(loss)):
                # 计算准确率
                with torch.no_grad():
                    acc = self.compute_accuracy(pred_x0, perm_matrix, mask)

                total_loss += loss.item()
                total_noise_loss += noise_loss.item()
                total_align_loss += align_loss.item()
                total_acc += acc
                num_batches += 1

            if self.rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': loss.item(),
                    'noise': noise_loss.item(),
                    'align': align_loss.item(),
                    'acc': acc
                })

        # 同步损失和准确率
        metrics = torch.tensor([total_loss, total_noise_loss, total_align_loss, total_acc, num_batches], device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        avg_loss = metrics[0].item() / max(metrics[4].item(), 1)
        avg_noise_loss = metrics[1].item() / max(metrics[4].item(), 1)
        avg_align_loss = metrics[2].item() / max(metrics[4].item(), 1)
        avg_acc = metrics[3].item() / max(metrics[4].item(), 1)

        return avg_loss, avg_noise_loss, avg_align_loss, avg_acc

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_noise_loss = 0
        total_align_loss = 0
        total_acc = 0
        num_batches = 0

        if self.rank == 0:
            pbar = tqdm(self.val_loader, desc='Validation')
        else:
            pbar = self.val_loader

        for batch in pbar:
            coords_A = batch['coords_A'].to(self.device)
            types_A = batch['types_A'].to(self.device)
            coords_B = batch['coords_B'].to(self.device)
            types_B = batch['types_B'].to(self.device)
            perm_matrix = batch['perm_matrix'].to(self.device)
            mask = batch['mask'].to(self.device)

            B = coords_A.size(0)
            timesteps = torch.randint(0, self.config['num_timesteps'], (B,), device=self.device)

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    noisy_alignment, noise = self.model.module.add_noise(perm_matrix, timesteps)
                    noise_pred = self.model(coords_A, types_A, coords_B, types_B, mask, noisy_alignment, timesteps)

                    mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)
                    noise_loss = F.mse_loss(noise_pred * mask_matrix, noise * mask_matrix)

                    pred_x0 = self.predict_x0_from_noise(noisy_alignment, noise_pred, timesteps)
                    align_loss, _ = self.compute_alignment_loss(pred_x0, perm_matrix, mask)

                    loss = noise_loss + self.config['align_loss_weight'] * align_loss
            else:
                noisy_alignment, noise = self.model.module.add_noise(perm_matrix, timesteps)
                noise_pred = self.model(coords_A, types_A, coords_B, types_B, mask, noisy_alignment, timesteps)

                mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)
                noise_loss = F.mse_loss(noise_pred * mask_matrix, noise * mask_matrix)

                pred_x0 = self.predict_x0_from_noise(noisy_alignment, noise_pred, timesteps)
                align_loss, _ = self.compute_alignment_loss(pred_x0, perm_matrix, mask)

                loss = noise_loss + self.config['align_loss_weight'] * align_loss

            if not (torch.isnan(loss) or torch.isinf(loss)):
                # 计算准确率
                acc = self.compute_accuracy(pred_x0, perm_matrix, mask)

                total_loss += loss.item()
                total_noise_loss += noise_loss.item()
                total_align_loss += align_loss.item()
                total_acc += acc
                num_batches += 1

        # 同步损失和准确率
        metrics = torch.tensor([total_loss, total_noise_loss, total_align_loss, total_acc, num_batches], device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        avg_loss = metrics[0].item() / max(metrics[4].item(), 1)
        avg_noise_loss = metrics[1].item() / max(metrics[4].item(), 1)
        avg_align_loss = metrics[2].item() / max(metrics[4].item(), 1)
        avg_acc = metrics[3].item() / max(metrics[4].item(), 1)

        return avg_loss, avg_noise_loss, avg_align_loss, avg_acc

    def train(self, start_epoch=0):
        """完整训练流程"""
        if self.rank == 0:
            print(f"Starting hybrid diffusion training on {self.world_size} GPUs")
            print(f"Config: {self.config}")
            if start_epoch > 0:
                print(f"Resuming from epoch {start_epoch + 1}")

        for epoch in range(start_epoch, self.config['epochs']):
            if self.rank == 0:
                print(f"\n=== Epoch {epoch + 1}/{self.config['epochs']} ===")

            train_loss, train_noise, train_align, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_noise_loss'].append(train_noise)
            self.history['train_align_loss'].append(train_align)
            self.history['train_acc'].append(train_acc)

            val_loss, val_noise, val_align, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_noise_loss'].append(val_noise)
            self.history['val_align_loss'].append(val_align)
            self.history['val_acc'].append(val_acc)

            self.scheduler.step()

            if self.rank == 0:
                print(f"Train - Loss: {train_loss:.6f} (Noise: {train_noise:.6f}, Align: {train_align:.6f}), Acc: {train_acc:.4f}")
                print(f"Val   - Loss: {val_loss:.6f} (Noise: {val_noise:.6f}, Align: {val_align:.6f}), Acc: {val_acc:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

                # 保存最佳模型（基于验证准确率）
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt', epoch)
                    print(f"✓ Saved best model (val_acc: {val_acc:.4f}, val_loss: {val_loss:.6f})")

                # 定期保存
                if (epoch + 1) % self.config['save_every'] == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', epoch)

        if self.rank == 0:
            self.save_checkpoint('final_model.pt', self.config['epochs'] - 1)
            self.save_history()
            print("\n=== Training completed ===")
            print(f"Best validation accuracy: {self.best_val_acc:.4f}")
            print(f"Best validation loss: {self.best_val_loss:.6f}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点并恢复训练状态"""
        if self.rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'train_noise_loss': [],
            'train_align_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_noise_loss': [],
            'val_align_loss': [],
            'val_acc': []
        })

        epoch = 0
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            try:
                epoch = int(os.path.basename(checkpoint_path).split('_')[-1].split('.')[0])
            except:
                epoch = len(self.history['train_loss'])

        if self.rank == 0:
            print(f"✓ Checkpoint loaded successfully")
            print(f"  Epoch: {epoch}")
            print(f"  Best val_acc: {self.best_val_acc:.4f}")
            print(f"  Best val_loss: {self.best_val_loss:.6f}")
            print(f"  History length: {len(self.history['train_loss'])} epochs")

        return epoch

    def save_checkpoint(self, filename, epoch):
        """保存检查点"""
        if self.rank != 0:
            return

        os.makedirs(self.config['output_dir'], exist_ok=True)
        path = os.path.join(self.config['output_dir'], filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history
        }, path)

    def save_history(self):
        """保存训练历史"""
        if self.rank != 0:
            return

        path = os.path.join(self.config['output_dir'], 'hybrid_diffusion_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def find_latest_checkpoint(output_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for fname in os.listdir(output_dir):
        if fname.startswith('checkpoint_epoch_') and fname.endswith('.pt'):
            try:
                epoch = int(fname.split('_')[-1].split('.')[0])
                checkpoints.append((epoch, os.path.join(output_dir, fname)))
            except:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def main_worker(rank, world_size, config):
    """每个进程的主函数"""
    setup_ddp(rank, world_size)

    if rank == 0:
        print("Creating dataloaders...")
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders_ddp(
        config['train_paths'],
        config['val_paths'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        print("Creating hybrid diffusion model...")
    model = DiffusionAlignmentModel(
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        num_cross_layers=config['num_cross_layers'],
        num_heads=config['num_heads'],
        num_timesteps=config['num_timesteps']
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")

    # 创建trainer
    trainer = HybridDiffusionTrainerDDP(model, train_loader, val_loader, train_sampler, val_sampler, config, rank, world_size)

    # 检查是否有checkpoint需要恢复
    start_epoch = 0
    if config.get('resume', True):
        checkpoint_path = find_latest_checkpoint(config['output_dir'])
        if checkpoint_path and rank == 0:
            print(f"\n{'='*60}")
            print(f"Found checkpoint: {checkpoint_path}")
            print(f"Resuming training...")
            print(f"{'='*60}\n")

        if checkpoint_path:
            start_epoch = trainer.load_checkpoint(checkpoint_path)

    # 训练
    trainer.train(start_epoch=start_epoch)

    cleanup_ddp()


def main():
    """主函数"""
    data_dir = os.path.join(PROJECT_ROOT, "data", "ready")
    output_dir = os.path.join(PROJECT_ROOT, "outputs_diffusion_hybrid")
    config = {
        # 数据
        'train_paths': [
            os.path.join(data_dir, 'rgd1_train.pkl'),
            os.path.join(data_dir, 't1x_train.pkl')
        ],
        'val_paths': [
            os.path.join(data_dir, 'rgd1_val.pkl'),
            os.path.join(data_dir, 't1x_val.pkl')
        ],
        'batch_size': 512,
        'num_workers': 16,

        # 模型 - 与improved相同的架构
        'node_dim': 768,
        'hidden_dim': 2048,
        'num_gnn_layers': 12,
        'num_cross_layers': 8,
        'num_heads': 16,

        # 扩散参数
        'num_timesteps': 1000,

        # 训练
        'epochs': 500,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'use_amp': True,

        # 混合损失权重
        'align_loss_weight': 0.5,  # 对齐损失的权重（可调节）

        # 保存
        'output_dir': output_dir,
        'save_every': 10
    }

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    print(f"Hybrid Loss: Noise MSE + {config['align_loss_weight']} × Alignment Loss")

    if world_size < 2:
        print("Warning: DDP works best with 2+ GPUs")
        world_size = 1

    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
