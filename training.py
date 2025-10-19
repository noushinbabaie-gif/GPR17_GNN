import os
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from GPR17_GNN.config import file_path
from GPR17_GNN.data_utils import SMILESDataset
from GPR17_GNN.model import EdgeNet
from GPR17_GNN.evaluations import train_one_epoch, evaluate
from GPR17_GNN.plots import plot_residuals_vs_predicted, normal_probability_plot, plot_true_vs_predicted, williams_plot
from GPR17_GNN.evaluate import evaluate_statistics_for_datasets
from GPR17_GNN.GNNexplainer import gnnexplainer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_file_path = file_path
    output_dir = "D:/code/new run asli/smiles for prediction/redata/newrun2"
    
    # Check if file exists
    if not os.path.exists(input_file_path):
        print(f"Dataset file not found in {input_file_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load data from Excel
    # Adjust your path/filename accordingly
    df = pd.read_excel(input_file_path)  # expects columns named "SMILES" and "pAffinity"

    # Shuffle data
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_excel(os.path.join(output_dir,"shuffled_df2.xlsx"))
    # Example split into train, validation, test
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    n_samples = len(df)
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n_samples))
    
    train_df = df.iloc[:train_end]

    #number_of_train_rows=train_df
    val_df = df.iloc[train_end:val_end]
    #number_of_val_rows=val_df
    test_df = df.iloc[val_end:]
    test_smiles = test_df["SMILES"].to_numpy()

    # Create datasets
    train_dataset = SMILESDataset(train_df)
    val_dataset = SMILESDataset(val_df)
    test_dataset = SMILESDataset(test_df)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Get sample data to see input feature dimensions
    sample_data = train_dataset[0]
    num_node_features = sample_data.x.shape[1]  # e.g., 5 from featurize_atom
    num_edge_features = sample_data.edge_attr.shape[1] if sample_data.edge_attr.shape[0] > 0 else 6
    num_global_features = sample_data.u.shape[1]  # e.g., 2 for (mw, tpsa)
    #print(num_node_features+num_edge_features+num_global_features)
    # Define model
    model = EdgeNet(in_channels=num_node_features,
                    edge_in_channels=num_edge_features,
                    global_in_channels=num_global_features,
                    hidden_channels=64,
                    num_layers=4).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    epochs = 500
    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_r2, _, _, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val R2: {val_r2:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    # Final evaluation on test set
    test_loss, test_r2, test_slope, test_intercept, preds, trues = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_loss:.4f}, Test R2: {test_r2:.4f}, "
          f"Slope: {test_slope:.4f}, Intercept: {test_intercept:.4f}")

    # 7. Residual Analysis
    plot_residuals_vs_predicted(preds, trues, test_idx, output_dir)
    normal_probability_plot(preds, trues, output_dir)
    
    def extract_graph_features(loader):
        model.eval()
        outputs = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = data.x
                for i, (conv, bn) in enumerate(zip(model.convs, model.bns)):
                    x = conv(x, data.edge_index, data.edge_attr)
                    x = bn(x)
                    x = F.relu(x)
                # Global pooling
                x = global_max_pool(x, data.batch)
                # Concat global features
                x = torch.cat([x, data.u], dim=-1)
                outputs.append(x.cpu().numpy())
        return np.vstack(outputs)

    gnnexplainer(test_dataset, model, output_dir, device)

    # Extract approximate "design matrix" from test loader
    avg_loss, r2, slope, intercept, test_pred, test_true = evaluate(model, test_loader, device)
    test_features = extract_graph_features(test_loader)
    avg_loss, r2, slope, intercept, validation_pred, validation_true = evaluate(model, val_loader, device)
    val_features = extract_graph_features(val_loader)
    avg_loss, r2, slope, intercept, training_pred, training_true = evaluate(model, train_loader, device)
    train_features = extract_graph_features(train_loader)

    williams_plot(train_features,
                  val_features, 
                  test_features, 
                  training_pred, 
                  validation_pred, 
                  test_pred, 
                  training_true, 
                  validation_true,
                  test_true, 
                  train_idx, 
                  val_idx, 
                  test_idx,
                  output_dir)

    fig_train, ax_train = plot_true_vs_predicted(
        training_true,
        validation_true,
        test_true,
        training_pred,
        validation_pred,
        test_pred,
        train_idx,
        val_idx,
        test_idx,
        output_dir)


    # Show all plots
    plt.show()
    # Evaluate statistics for train, val, and test
    results = evaluate_statistics_for_datasets(model, train_loader, val_loader, test_loader, device)
    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(output_dir, "Results.xlsx"))
    print("The code has completed successfully!")      
