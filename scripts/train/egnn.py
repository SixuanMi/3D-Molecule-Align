import os
import sys

# 路径修复
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils import MolStructure, AtomMapping
except ImportError:
    pass 

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 新增绘图库
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import knn_graph
    from torch_scatter import scatter 
except ImportError:
    print("错误: 未找到 torch_geometric 或 torch_scatter。")
    sys.exit(1)

# ==========================================
# 1. 基础 EGNN 层
# ==========================================
class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index
        rel_pos = x[row] - x[col]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        
        edge_feat = torch.cat([h[row], h[col], dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        
        m_i = scatter(m_ij, row, dim=0, reduce='sum', dim_size=h.size(0))
        
        node_input = torch.cat([h, m_i], dim=-1)
        h_new = h + self.node_mlp(node_input)
        
        coord_weight = self.coord_mlp(m_ij)
        x_update = scatter(rel_pos * coord_weight, row, dim=0, reduce='mean', dim_size=x.size(0))
        x_new = x + x_update
        
        return h_new, x_new

# ==========================================
# 2. 编码器：Siamese EGNN
# ==========================================
class MoleculeEGNN(nn.Module):
    def __init__(self, num_atom_types=100, hidden_dim=128, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_idx, x, batch=None):
        h = self.embedding(h_idx)
        for layer in self.layers:
            edge_index = knn_graph(x, k=8, batch=batch, loop=False)
            h, x = layer(h, x, edge_index)
        return self.out_norm(h)

# ==========================================
# 3. 完整的模型封装
# ==========================================
class EGNNAlignmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MoleculeEGNN(hidden_dim=128)
        self.temperature = nn.Parameter(torch.tensor(0.07)) 

    def forward(self, batch):
        h_s = self.encoder(batch.x, batch.pos, batch.batch)
        h_t = self.encoder(batch.h_t, batch.pos_t, batch.batch)
        return F.normalize(h_s, dim=1), F.normalize(h_t, dim=1)

# ==========================================
# 4. 辅助函数：绘图与数据处理
# ==========================================
def plot_training_curves(history, save_dir):
    """绘制并保存训练过程曲线"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curve', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy Curve
    # 将小数转换为百分比
    train_acc_pct = [x * 100 for x in history['train_acc']]
    val_acc_pct = [x * 100 for x in history['val_acc']]
    
    ax2.plot(epochs, train_acc_pct, 'b-o', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_acc_pct, 'r-s', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curve', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n📈 训练曲线图已保存至: {save_path}")
    plt.close()

def process_pkl_data(pkl_files):
    data_list = []
    print(f"正在读取 {len(pkl_files)} 个数据文件...")
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path: sys.path.insert(0, root_dir)

    for file_path in pkl_files:
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
                except: continue
        except: continue
    return data_list

def compute_supervised_matching_loss_and_acc(h_s, h_t, mapping_indices, batch_idx, temperature):
    loss = 0
    total_correct = 0
    total_nodes = 0
    
    node_counts = torch.bincount(batch_idx)
    splits = torch.cumsum(node_counts, dim=0)
    start = 0
    total_graphs = batch_idx.max().item() + 1
    
    for i in range(total_graphs):
        end = splits[i].item()
        
        curr_h_s = h_s[start:end] 
        curr_h_t = h_t[start:end] 
        curr_mapping = mapping_indices[start:end] 
        
        logits = torch.matmul(curr_h_s, curr_h_t.T) / temperature
        
        loss += F.cross_entropy(logits, curr_mapping)
        
        preds = logits.argmax(dim=1)
        total_correct += (preds == curr_mapping).sum().item()
        total_nodes += (end - start)
        start = end
        
    avg_loss = loss / total_graphs
    avg_acc = total_correct / total_nodes if total_nodes > 0 else 0
    return avg_loss, avg_acc

def train_main():
    DATA_DIR = "../../data/ready"
    MODEL_SAVE_PATH = "models/best_gnn_model.pth"
    BATCH_SIZE = 128  
    EPOCHS = 20
    NUM_WORKERS = 4
    
    if not os.path.exists(DATA_DIR) and os.path.exists(f"../{DATA_DIR}"):
        DATA_DIR = f"../{DATA_DIR}"
        MODEL_SAVE_PATH = f"../{MODEL_SAVE_PATH}"
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    pkl_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
    all_data = process_pkl_data(pkl_files)
    if not all_data: return
    
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用 EGNN 监督匹配网络 | 设备: {device}")
    
    model = EGNNAlignmentModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    # 记录训练历史
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            h_s, h_t = model(batch)
            loss, acc = compute_supervised_matching_loss_and_acc(
                h_s, h_t, batch.mapping, batch.batch, model.temperature
            )
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            train_acc += acc * batch.num_graphs
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc*100:.2f}%"})
            
        avg_train_loss = train_loss / len(train_data)
        avg_train_acc = train_acc / len(train_data)
        
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                h_s, h_t = model(batch)
                loss, acc = compute_supervised_matching_loss_and_acc(
                    h_s, h_t, batch.mapping, batch.batch, model.temperature
                )
                val_loss += loss.item() * batch.num_graphs
                val_acc += acc * batch.num_graphs
        
        avg_val_loss = val_loss / len(val_data)
        avg_val_acc = val_acc / len(val_data)
        
        # 记录数据
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch+1} | Acc: {avg_train_acc*100:.2f}% | Val Acc: {avg_val_acc*100:.2f}%")
        
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> 最佳模型已保存 (Val Acc: {best_acc*100:.2f}%)")

    # 训练结束后绘制曲线
    plot_training_curves(history, os.path.dirname(MODEL_SAVE_PATH))

# ==========================================
# 5. 对外推理接口
# ==========================================
def load_trained_model(model_path, device='cpu'):
    if not os.path.exists(model_path): raise FileNotFoundError(f"找不到模型文件: {model_path}")
    model = EGNNAlignmentModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def reorder_gnn(ref_struct, cand_struct, model=None):
    if model is None: raise ValueError("Need loaded model")
    device = next(model.parameters()).device
    
    atom_types_s = torch.tensor(ref_struct.atoms, dtype=torch.long).to(device)
    coords_s = torch.tensor(ref_struct.coordinates, dtype=torch.float).to(device)
    atom_types_t = torch.tensor(cand_struct.atoms, dtype=torch.long).to(device)
    coords_t = torch.tensor(cand_struct.coordinates, dtype=torch.float).to(device)
    
    batch_idx = torch.zeros(len(atom_types_s), dtype=torch.long, device=device)
    class MiniBatch: pass
    batch = MiniBatch()
    batch.x = atom_types_s
    batch.pos = coords_s
    batch.h_t = atom_types_t
    batch.pos_t = coords_t
    batch.batch = batch_idx
    
    with torch.no_grad():
        h_s, h_t = model(batch)
        
    similarity = torch.matmul(h_s, h_t.T)
    ref_atoms = np.array(ref_struct.atoms)
    cand_atoms = np.array(cand_struct.atoms)
    type_mismatch = ref_atoms[:, None] != cand_atoms[None, :]
    sim_np = similarity.cpu().numpy()
    sim_np[type_mismatch] = -1e9
    
    row_ind, col_ind = linear_sum_assignment(sim_np, maximize=True)
    return col_ind

if __name__ == "__main__":
    train_main()