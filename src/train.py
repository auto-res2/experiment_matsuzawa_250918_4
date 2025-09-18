import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.utils import degree, add_self_loops
import math

# Note: Using local mock implementations of 'gact' and 'fastfood_fft'
try:
    from . import gact
    from .fastfood_fft import Fastfood
except ImportError:
    # Fallback for direct script execution
    import gact
    from fastfood_fft import Fastfood

# SCoRe-GNN Implementation
class SCoReConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, trace_bits=256, lazy_T=5, conf_tau=0.9, q_bits=3, dropout=0.6, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.trace_bits = trace_bits
        self.lazy_T = lazy_T
        self.conf_tau = conf_tau
        self.q_bits = q_bits
        self.dropout = dropout

        self.lin_q = nn.Linear(in_channels, heads * out_channels)
        self.lin_k = nn.Linear(in_channels, heads * out_channels)
        self.lin_v = nn.Linear(in_channels, heads * out_channels)
        self.lin_out = nn.Linear(heads * out_channels, out_channels)

        self.fastfood = Fastfood(heads * out_channels, 1.0)

        # CALR state
        self.message_buffer = None
        self.confidence_streak_counter = None
        self.hub_message_count = 0

        if self.q_bits > 0:
            gact.set_optimization_level('L2' if self.q_bits <=3 else 'L1') # Simplified mapping

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()
        self.lin_out.reset_parameters()

    def _scrf(self, x, edge_index):
        N, _ = x.shape
        if not hasattr(self, 'deg_for_scrf') or self.deg_for_scrf is None:
            deg = degree(edge_index[0], N, dtype=x.dtype).to(x.device)
            self.deg_for_scrf = deg
        else:
            deg = self.deg_for_scrf

        probe = torch.randn(self.trace_bits, N, device=x.device)
        
        # Compute (D-A)v without materializing L
        # Manual adjacency computation to avoid message passing interference
        row, col = edge_index
        # Create adjacency matrix action: A @ probe.T using scatter
        ap_t = torch.zeros_like(probe.t())  # Shape: (N, trace_bits)
        ap_t.scatter_add_(0, row.unsqueeze(1).expand(-1, self.trace_bits), probe.t()[col])
        
        # D@probe.T is just deg * probe.T  
        dp_t = deg.view(-1, 1) * probe.t()
        lp_t = dp_t - ap_t

        # Estimate trace tr(L) = E[vT L v]
        lap_trace = (probe * lp_t.t()).sum() / self.trace_bits
        sigma = (self.out_channels / lap_trace.abs().clamp(min=1e-6)).sqrt().clamp(min=0.1, max=10.0)
        
        # Apply Fastfood projection with conditioned sigma
        self.fastfood.sigma.data = sigma.detach()
        return self.fastfood(x)

    def forward(self, x, edge_index, epoch=0, return_attention_weights=False):
        H, C = self.heads, self.out_channels
        N = x.size(0)

        # CALR Logic
        use_calr = self.training and self.lazy_T > 0 and epoch > 0 and (epoch % self.lazy_T != 0)
        freeze_mask = torch.zeros(N, dtype=torch.bool, device=x.device)

        if use_calr:
            with torch.no_grad():
                confidences = x.softmax(dim=-1).max(dim=-1)[0]
                is_confident = confidences > self.conf_tau
                if self.confidence_streak_counter is None:
                    self.confidence_streak_counter = torch.zeros(N, device=x.device)
                self.confidence_streak_counter[is_confident] += 1
                self.confidence_streak_counter[~is_confident] = 0
                freeze_mask = self.confidence_streak_counter >= self.lazy_T

        q = self.lin_q(x).view(-1, H, C)
        k = self.lin_k(x).view(-1, H, C)
        v = self.lin_v(x).view(-1, H, C)
        
        if self.q_bits > 0:
             k = gact.func.quantize(k, self.q_bits)
             v = gact.func.quantize(v, self.q_bits)

        q_proj = self._scrf(q.view(N, H*C), edge_index).view(-1, H, C)
        k_proj = self._scrf(k.view(N, H*C), edge_index).view(-1, H, C)

        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        # propagate_type: (q_proj: Tensor, k_proj: Tensor, v: Tensor)
        out = self.propagate(edge_index, q_proj=q_proj, k_proj=k_proj, v=v, size=None)

        if use_calr and freeze_mask.any():
            out[freeze_mask] = self.message_buffer[freeze_mask]
        
        if self.training:
            if self.message_buffer is None:
                self.message_buffer = torch.zeros_like(out)
            self.message_buffer[~freeze_mask] = out[~freeze_mask].detach()

        out = out.view(-1, H * C)
        out = self.lin_out(out)
        
        # Logging hub messages
        if self.training and hasattr(self, 'deg_for_scrf'):
            is_hub = self.deg_for_scrf > 500
            self.hub_message_count = (~freeze_mask[is_hub]).sum().item()
        
        return out

    def message(self, q_proj_i: torch.Tensor, k_proj_j: torch.Tensor, v_j: torch.Tensor, index: torch.Tensor, ptr: torch.Tensor, size_i: int) -> torch.Tensor:
        alpha = (q_proj_i * k_proj_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = torch.exp(alpha - torch.max(alpha)) # Numerically stable softmax
        alpha = alpha / (alpha.sum(dim=0).unsqueeze(0))

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = v_j * alpha.unsqueeze(-1)
        return out

class SCoReGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads=4, dropout=0.5, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SCoReConv(in_dim, hidden_dim, heads=heads, dropout=dropout, **kwargs))
        for _ in range(num_layers - 2):
            self.layers.append(SCoReConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, **kwargs))
        self.layers.append(SCoReConv(hidden_dim, out_dim, heads=1, dropout=dropout, **kwargs))
        self.dropout = dropout

    def forward(self, x, edge_index, epoch=0):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, epoch=epoch)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# Baseline Models
class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads=4, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.layers.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, epoch=0):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x

class AdaFlashGNN(nn.Module):
    # Simplified AdaFlash: uses degree to modulate attention score.
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads=4, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # Using GATv2Conv as a base for more expressive attention
        from torch_geometric.nn import GATv2Conv
        self.layers.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.layers.append(GATv2Conv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, epoch=0):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x

class TransformerGNN(MessagePassing):
    # Base for BigBird and BlockDiag
    def __init__(self, in_channels, out_channels, heads=1, attention_mode='vanilla', block_size=64, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.attention_mode = attention_mode
        self.block_size = block_size
        self.lin_q = nn.Linear(in_channels, heads * out_channels)
        self.lin_k = nn.Linear(in_channels, heads * out_channels)
        self.lin_v = nn.Linear(in_channels, heads * out_channels)
        self.lin_out = nn.Linear(heads * out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)

    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        q = self.lin_q(x).view(-1, H, C)
        k = self.lin_k(x).view(-1, H, C)
        v = self.lin_v(x).view(-1, H, C)
        out = self.propagate(edge_index, q=q, k=k, v=v, size=None)
        out = out.view(-1, H * C)
        return self.lin_out(out)

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        alpha = (q_i * k_j).sum(dim=-1) / math.sqrt(self.out_channels)

        # Apply attention mask based on mode
        if self.attention_mode == 'block_diag':
            row, col = index
            mask = (row // self.block_size) != (col // self.block_size)
            alpha[mask] = -1e9
        elif self.attention_mode == 'big_bird': # Simplified: local + random
            row, col = index
            local_mask = torch.abs(row - col) > self.block_size
            random_mask = torch.rand_like(row, dtype=torch.float) > 0.1 # 10% random attention
            mask = local_mask & random_mask
            alpha[mask] = -1e9
        
        alpha = F.softmax(alpha, dim=-1)
        return v_j * alpha.view(-1, self.heads, 1)

class BigBirdGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads=4, dropout=0.5, block_size=64):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerGNN(in_dim, hidden_dim, heads=heads, attention_mode='big_bird', block_size=block_size))
        for _ in range(num_layers - 2):
            self.layers.append(TransformerGNN(hidden_dim * heads, hidden_dim, heads=heads, attention_mode='big_bird', block_size=block_size))
        self.layers.append(TransformerGNN(hidden_dim * heads, out_dim, heads=1, attention_mode='big_bird', block_size=block_size))
        self.dropout = dropout

    def forward(self, x, edge_index, epoch=0):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                 x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x

class BlockDiagTransGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads=4, dropout=0.5, block_size=128):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerGNN(in_dim, hidden_dim, heads=heads, attention_mode='block_diag', block_size=block_size))
        for _ in range(num_layers - 2):
            self.layers.append(TransformerGNN(hidden_dim*heads, hidden_dim, heads=heads, attention_mode='block_diag', block_size=block_size))
        self.layers.append(TransformerGNN(hidden_dim*heads, out_dim, heads=1, attention_mode='block_diag', block_size=block_size))
        self.dropout = dropout

    def forward(self, x, edge_index, epoch=0):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                 x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x

# Model Factory
def get_model(config, data):
    model_name = config['model']['name']
    model_params = config['model']
    num_layers = model_params.get('layers', 4)
    hidden_dim = model_params.get('hidden_dim', 256)
    heads = model_params.get('heads', 4)
    dropout = model_params.get('dropout', 0.5)

    in_dim = data.num_node_features
    # For multi-label or graph classification, out_dim is num_classes. For multi-class, it's also num_classes.
    if data.y.dim() > 1 and data.y.shape[1] > 1: # Multi-label or graph classification
        out_dim = data.y.shape[1]
    else: # Node classification
        out_dim = data.num_classes

    if model_name == 'SCoRe-GNN' or model_name == 'SCoRe-no-SCRF':
        return SCoReGNN(in_dim, hidden_dim, out_dim, num_layers, heads=heads, dropout=dropout, 
                        lazy_T=model_params.get('lazy_T', 5), 
                        conf_tau=model_params.get('conf_tau', 0.9),
                        q_bits=model_params.get('q_bits', 3))
    elif model_name == 'GAT':
        return GATModel(in_dim, hidden_dim, out_dim, num_layers, heads=heads, dropout=dropout)
    elif model_name == 'AdaFlash':
        return AdaFlashGNN(in_dim, hidden_dim, out_dim, num_layers, heads=heads, dropout=dropout)
    elif model_name == 'BigBird-GNN':
        return BigBirdGNN(in_dim, hidden_dim, out_dim, num_layers, heads=heads, dropout=dropout, block_size=64)
    elif model_name == 'BlockDiag-TransGNN':
        return BlockDiagTransGNN(in_dim, hidden_dim, out_dim, num_layers, heads=heads, dropout=dropout, block_size=128)
    else:
        raise ValueError(f"Unknown model: {model_name}")

