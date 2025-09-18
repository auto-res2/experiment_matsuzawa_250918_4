import json
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.stats import wilcoxon
import torch.nn.functional as F

def calculate_ece(logits, labels, n_bins=15):
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def evaluate_model(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index)
        out = out[mask]
        y_true = data.y[mask]
        
        metrics = {}
        # Multi-label vs Multi-class
        if y_true.dim() > 1 and y_true.shape[1] > 1:
            y_prob = torch.sigmoid(out)
            y_pred = (y_prob > 0.5).long()
            metrics['roc_auc'] = roc_auc_score(y_true.cpu().numpy(), y_prob.cpu().numpy(), average='macro')
            metrics['f1_macro'] = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            metrics['accuracy'] = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) # For graph classification
            metrics['ece'] = 0 # ECE not well-defined for multi-label
        else:
            y_prob = F.softmax(out, dim=1)
            y_pred = y_prob.argmax(dim=1)
            y_true_flat = y_true.squeeze()
            metrics['accuracy'] = accuracy_score(y_true_flat.cpu().numpy(), y_pred.cpu().numpy())
            metrics['roc_auc'] = roc_auc_score(y_true_flat.cpu().numpy(), y_prob.cpu().numpy(), multi_class='ovr', average='macro')
            metrics['f1_macro'] = f1_score(y_true_flat.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            metrics['ece'] = calculate_ece(out, y_true_flat)
            
    return metrics

def plot_pareto_front(results_df, output_dir):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_df, x='energy_joules_mean', y='accuracy_mean', hue='model', style='dataset', s=200)
    plt.title('Experiment 1: Pareto Front (Accuracy vs. Energy)')
    plt.xlabel('Cumulative Energy (Joules)')
    plt.ylabel('Test Accuracy / ROC-AUC')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'images', 'exp1_pareto_front.pdf'))
    plt.close()

def plot_synthetic_results(results_df, output_dir):
    for h in results_df['h'].unique():
        plt.figure(figsize=(10, 6))
        subset = results_df[results_df['h'] == h]
        sns.lineplot(data=subset, x='p_rewire', y='accuracy_mean', hue='model', style='d', marker='o')
        plt.title(f'Experiment 2: Accuracy vs. Rewiring (Homophily h={h})')
        plt.xlabel('Rewiring Probability (p_rewire)')
        plt.ylabel('Test Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'images', f'exp2_synthetic_h_{h}.pdf'))
        plt.close()

def plot_ablation_bars(results_df, output_dir):
    metrics_to_plot = ['accuracy_mean', 'ece_mean', 'peak_mem_mb_mean', 'hub_msgs_mean']
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        sns.barplot(data=results_df, x='dataset', y=metric, hue='variant')
        plt.title(f'Experiment 3: Ablation Study - {metric}')
        plt.ylabel(metric)
        plt.xlabel('Dataset')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'images', f'exp3_ablation_{metric}.pdf'))
        plt.close()

def run_evaluation(exp_name, results_log, output_dir):
    print(f"--- Running Evaluation for {exp_name} ---")
    df = pd.DataFrame(results_log)

    # Aggregate results by grouping by all config params except seed
    group_cols = [c for c in df.columns if c not in ['seed', 'final_accuracy', 'final_roc_auc', 'final_f1_macro', 'final_ece', 'total_time', 'total_energy', 'peak_memory_mb', 'final_hub_msgs']]
    if not group_cols:
        agg_df = df
    else:
        agg_funcs = {
            'final_accuracy': ['mean', 'std'],
            'final_roc_auc': ['mean', 'std'],
            'final_f1_macro': ['mean', 'std'],
            'final_ece': ['mean', 'std'],
            'total_time': ['mean', 'std'],
            'total_energy': ['mean', 'std'],
            'peak_memory_mb': ['mean', 'std'],
            'final_hub_msgs': ['mean', 'std'],
        }
        # Only aggregate columns that exist
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
        if not agg_funcs:
             agg_df = df.groupby(group_cols).size().reset_index(name='counts')
        else:
            agg_df = df.groupby(group_cols).agg(agg_funcs).reset_index()
            agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
            for col in group_cols:
                agg_df.rename(columns={f'{col}_': col}, inplace=True)

    # Statistical Tests for Exp 1
    if exp_name == 'experiment_1':
        stats = {}
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            score_ref = dataset_df[dataset_df['model'] == 'SCoRe-GNN']['final_accuracy']
            if len(score_ref) == 0: continue
            stats[dataset] = {}
            for model in dataset_df['model'].unique():
                if model == 'SCoRe-GNN': continue
                score_comp = dataset_df[dataset_df['model'] == model]['final_accuracy']
                if len(score_comp) > 0:
                    stat, p = wilcoxon(score_ref, score_comp)
                    stats[dataset][model] = {'statistic': stat, 'p_value': p}
        agg_df['wilcoxon_stats'] = pd.Series([stats])

    # Generate Plots
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    if exp_name == 'experiment_1':
        plot_pareto_front(agg_df, output_dir)
    elif exp_name == 'experiment_2':
        plot_synthetic_results(agg_df, output_dir)
    elif exp_name == 'experiment_3':
        plot_ablation_bars(agg_df, output_dir)

    # Save and Print Results
    results_dict = agg_df.to_dict(orient='records')
    results_json = json.dumps(results_dict, indent=4)
    
    json_path = os.path.join(output_dir, f'{exp_name}_results.json')
    with open(json_path, 'w') as f:
        f.write(results_json)

    print(f"Evaluation results for {exp_name} saved to {json_path}")
    print("--- Results JSON --- G B
" + results_json)
    print("--- End Evaluation --- G B
")