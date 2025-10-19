import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.utils import to_networkx
import networkx as nx

class WrappedModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x, edge_index, batch=None):
        if x is None:
            num_nodes = edge_index.max().item() + 1
            x = torch.ones((num_nodes, self.original_model.convs[0].in_channels)).to(edge_index.device)

        edge_attr = torch.ones(
            (edge_index.size(1), self.original_model.convs[0].nn[0].in_features)
        ).to(x.device)

        num_graphs = batch.max().item() + 1 if batch is not None else 1
        global_in_channels = (
            self.original_model.post_pool[0].in_features - self.original_model.convs[0].out_channels
        )
        u = torch.zeros((num_graphs, global_in_channels)).to(x.device)

        return self.original_model(x, edge_index, edge_attr, batch, u)

def gnnexplainer(test_dataset, model, output_dir, device):
    print("Running GNNExplainer on all test graphs...")

    wrapped_model = WrappedModel(model).to(device)
    wrapped_model.eval()

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='phenomenon', 
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw'
        )
    )

    all_node_masks = []
    all_edge_masks = []

    for i, sample_graph in enumerate(test_dataset):
        try:
            sample_graph = sample_graph.to(device)
            batch = sample_graph.batch if hasattr(sample_graph, "batch") else torch.zeros(sample_graph.x.size(0), dtype=torch.long).to(device)
            target = sample_graph.y.to(device)  

            explanation = explainer(
                x=sample_graph.x,
                edge_index=sample_graph.edge_index,
                batch=batch,
                target=target
            )

            node_feat_mask = explanation.node_mask
            edge_mask = explanation.edge_mask

            node_mask_np = node_feat_mask.detach().cpu().numpy()
            edge_mask_np = edge_mask.detach().cpu().numpy()
            all_node_masks.append(node_mask_np)
            all_edge_masks.append(edge_mask_np)

            # --- Visualization ---
            G = to_networkx(sample_graph, to_undirected=True)
            pos = nx.spring_layout(G, seed=42)

            # Reduce node mask to 1 value per node
            node_colors = node_feat_mask.mean(dim=1).detach().cpu().numpy()
            node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min() + 1e-8)
            node_colors = node_colors[:len(G.nodes())]

            # Match edge mask to G.edges()
            edge_index_tuples = list(map(tuple, sample_graph.edge_index.cpu().numpy().T))
            G_edges = list(G.edges())
            edge_mask_dict = {tuple(sorted(edge)): mask for edge, mask in zip(edge_index_tuples, edge_mask_np)}
            edge_weights = [edge_mask_dict.get(tuple(sorted(e)), 0.0) for e in G_edges]
            edge_weights = np.array(edge_weights)
            edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)

            plt.close('all')

            # --- Edge Visualization ---
            plt.figure(figsize=(8, 6))
            nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color='white', node_size=400)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, edgelist=G_edges, edge_color=edge_weights, edge_cmap=plt.cm.Reds, width=2.0)
            plt.title(f"GNNExplainer Node and Edge Importance - Graph {i}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"gnn_explainer_graph_{i}.png"), dpi=300)
            plt.close()

            # --- Node Importance Visualization ---
            plt.figure(figsize=(8, 6))
            nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=node_colors, cmap=plt.cm.Blues, node_size=400)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, edge_color= None, alpha=0.3)
            plt.title(f"GNNExplainer Node and Edge Importance - Graph {i}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"gnnexplainer_node_importance_graph_{i}.png"), dpi=300, transparent= True)
            plt.close()

            # --- Node Mask Only Visualization ---
            plt.figure(figsize=(8, 6))
            nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=node_colors, cmap=plt.cm.YlGnBu, node_size=500)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, alpha=0.3)
            plt.title(f"GNNExplainer Node Mask Only - Graph {i}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"gnnexplainer_node_mask_only_graph_{i}.png"), dpi=300)
            plt.close()

        except Exception as e:
            print(f"Failed GNNExplainer on sample {i}: {e}")
            all_node_masks.append(None)
            all_edge_masks.append(None)

    print("GNNExplainer analysis completed.")

    # --- Summary Statistics ---
    node_mask_means = []
    edge_mask_means = []

    for n_mask, e_mask in zip(all_node_masks, all_edge_masks):
        node_mean = np.mean(n_mask) if n_mask is not None else np.nan
        edge_mean = np.mean(e_mask) if e_mask is not None else np.nan
        node_mask_means.append(node_mean)
        edge_mask_means.append(edge_mean)

    valid_indices = [i for i, (n, e) in enumerate(zip(node_mask_means, edge_mask_means)) if not (np.isnan(n) or np.isnan(e))]
    df = pd.DataFrame({
        "Graph Index": valid_indices,
        "Mean Node Mask": [node_mask_means[i] for i in valid_indices],
        "Mean Edge Mask": [edge_mask_means[i] for i in valid_indices]
    })
    df.to_excel(os.path.join(output_dir, "gnnexplainer_results.xlsx"), index=False)

    final_node_mean = np.nanmean(node_mask_means)
    final_edge_mean = np.nanmean(edge_mask_means)

    print(f"Average Node Mask Importance: {final_node_mean:.4f}")
    print(f"Average Edge Mask Importance: {final_edge_mean:.4f}")

    plt.figure(figsize=(6, 4))
    plt.bar(['Node Mask Mean', 'Edge Mask Mean'], [final_node_mean, final_edge_mean], color=['skyblue', 'salmon'])
    plt.ylabel('Mean Importance')
    plt.title('Average Mask Importance (GNNExplainer)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gnnexplainer_summary_mean_importance.png"), dpi=300)
    plt.close()    
