import os
import torch
from torch_geometric.datasets import Flickr, Reddit, TUDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import networkx as nx
from torch_geometric.utils import from_networkx
import numpy as np

# Monkey patch to automatically answer 'y' for OGB dataset downloads
def auto_yes_input(prompt):
    print(prompt)
    print("Automatically answering 'y' for dataset download")
    return 'y'

# Apply monkey patch for OGB downloads
import builtins
original_input = builtins.input

def get_data(name, root='data/'):
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    processed_path = os.path.join(path, 'processed', 'data.pt')

    if os.path.exists(processed_path):
        print(f"Loading cached dataset: {name}")
        try:
            data = torch.load(processed_path, weights_only=False)
        except Exception as e:
            print(f"Failed to load cached data with weights_only=False: {e}")
            # If weights_only=False fails, allow unsafe globals for PyG data structures
            import torch_geometric.data.data
            torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
            data = torch.load(processed_path, weights_only=True)
        return data
    
    print(f"Downloading and processing dataset: {name}")
    try:
        if name.startswith('ogbn'):
            # Temporarily replace input function for OGB datasets
            builtins.input = auto_yes_input
            try:
                dataset = PygNodePropPredDataset(name=name, root=path, transform=T.ToUndirected())
                data = dataset[0]
            finally:
                # Restore original input function
                builtins.input = original_input
        elif name == 'Flickr':
            dataset = Flickr(root=path, transform=T.ToUndirected())
            data = dataset[0]
        elif name == 'Reddit':
            dataset = Reddit(root=path, transform=T.ToUndirected())
            data = dataset[0]
        elif name == 'peptides-functional':
             # For graph classification datasets
            dataset = TUDataset(root=path, name='Peptides-func', use_node_attr=True)
            # This returns a list of graphs, we will handle this in main script
            torch.save(dataset, processed_path)
            return dataset
        else:
            raise ValueError(f"Unknown dataset: {name}")
    except Exception as e:
        print(f"Failed to download or process dataset {name}. Error: {e}")
        raise FileNotFoundError(f"Dataset {name} could not be loaded. Aborting.")

    # Z-score normalization for node features
    if data.x is not None:
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        data.x = (data.x - mean) / std.clamp(min=1e-8)
    
    # Assign num_classes if not present
    if hasattr(data, 'y') and data.y is not None:
        if data.y.dim() == 1 or data.y.shape[1] == 1:
             data.num_classes = len(torch.unique(data.y))
        else: # multi-label
             data.num_classes = data.y.shape[1]

    torch.save(data, processed_path)
    return data

def generate_synthetic_graph(d, h, p_rewire, n=20000, num_classes=10, feature_dim=256):
    print(f"Generating synthetic graph: d={d}, h={h}, p_rewire={p_rewire}")
    # 1. Create random regular graph
    g = nx.random_regular_graph(d, n)

    # 2. Rewire edges
    edges_to_rewire = []
    for u, v in g.edges():
        if np.random.rand() < p_rewire:
            edges_to_rewire.append((u, v))
    
    for u, v in edges_to_rewire:
        g.remove_edge(u, v)
        new_neighbor = np.random.choice([i for i in range(n) if i != u and not g.has_edge(u, i)])
        g.add_edge(u, new_neighbor)

    # 3. Assign labels via stochastic block model
    labels = np.random.randint(0, num_classes, n)
    for u, v in g.edges():
        if labels[u] == labels[v]: # Intra-class edge
            if np.random.rand() > h:
                labels[v] = np.random.randint(0, num_classes)
        else: # Inter-class edge
            if np.random.rand() < h:
                labels[v] = labels[u]
    
    # 4. Generate features
    class_means = np.random.randn(num_classes, feature_dim)
    features = np.zeros((n, feature_dim), dtype=np.float32)
    for i in range(n):
        features[i] = np.random.normal(loc=class_means[labels[i]], scale=1.0)

    data = from_networkx(g)
    data.x = torch.from_numpy(features)
    data.y = torch.from_numpy(labels).long()
    data.num_classes = num_classes

    # 5. Create splits
    perm = torch.randperm(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[perm[:train_end]] = True
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask[perm[train_end:val_end]] = True
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask[perm[val_end:]] = True

    return data
