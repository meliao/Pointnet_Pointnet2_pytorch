import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
# from MoleculeDataSet import PointCloudMoleculeDataSet

def make_training_progress_plot(training_results_fp: str,
                                out_fp: str,
                                title: str=None) -> None:

    df = pd.read_table(training_results_fp)
    fig, ax = plt.subplots()

    fig.patch.set_facecolor('white')
    fig.set_size_inches(10, 10)

    ax.plot(df['epoch'], df['train_MSE'], '-', label='Train MSE')
    ax.plot(df['epoch'], df['test_MSE'], '-', label='Test MSE')

    ax.set_yscale('log')
    ax.set_xlabel('Epoch', size=20)
    ax.set_ylabel('Train/Test MSE', size=20)
    if title is not None:
        ax.set_title(title, size=20)
        fig.tight_layout()
    ax.legend(prop={'size': 15})
    plt.savefig(out_fp)


def make_predictions_plot(model, data_loader, out_fp, device, title=None) -> None:
    model = model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for points_and_features, _, target in data_loader:
            points_and_features = points_and_features.to(device)
            # U_matrices = U_matrices.to(device)
            
            preds = model(points_and_features)
            
            all_preds.append(preds.cpu())
            all_labels.append(target)

    all_preds = torch.cat(all_preds).flatten().numpy()
    all_labels = torch.cat(all_labels).flatten().numpy()

    fig, ax = plt.subplots()

    fig.patch.set_facecolor('white')
    fig.set_size_inches(10, 10)

    ax.plot(all_preds, all_labels, '.')
    ax.plot(ax.get_ybound(), ax.get_ybound(), '--')
    ax.set_xlabel('Predicted Energies', size=20)
    ax.set_ylabel('True Energies', size=20)
    # ax.set_xbound(*ax.get_ybound())
    # plt.show()

    if title is not None:
        ax.set_title(title, size=20)
        fig.tight_layout()

    plt.savefig(out_fp)