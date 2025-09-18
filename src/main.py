import argparse
import yaml
import os
import torch
import numpy as np
import random
import time
import wandb
import threading

# Configure wandb for offline mode to avoid authentication issues
os.environ['WANDB_MODE'] = 'offline'

from ptflops import get_model_complexity_info

# Assuming pynvml is installed for energy monitoring
try:
    import pynvml
except ImportError:
    pynvml = None

# Use relative imports for project modules
from .preprocess import get_data, generate_synthetic_graph
from .train import get_model
from .evaluate import evaluate_model, run_evaluation

# --- Utility Functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(config):
    device_str = config.get('globals', {}).get('device', 'cpu')
    if 'cuda' in device_str and not torch.cuda.is_available():
        print(f"Warning: CUDA specified ('{device_str}') but not available. Falling back to CPU.")
        return torch.device('cpu')
    return torch.device(device_str)

class PowerLogger(threading.Thread):
    def __init__(self, interval=1):
        super().__init__()
        self._stop_event = threading.Event()
        self.interval = interval
        self.power_readings = []
        if pynvml:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

    def run(self):
        if not self.handle: return
        while not self._stop_event.is_set():
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # In Watts
            self.power_readings.append(power)
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        if pynvml:
            pynvml.nvmlShutdown()
        total_time = len(self.power_readings) * self.interval
        if not self.power_readings: return 0
        avg_power = sum(self.power_readings) / len(self.power_readings)
        return avg_power * total_time # Total Joules

# --- Training and Validation ---
def run_single_experiment(config, device, run_id):
    set_seed(config['seed'])
    
    # --- Data Loading ---
    if config.get('synthetic_params'):
        data = generate_synthetic_graph(**config['synthetic_params'])
    else:
        data = get_data(config['dataset']['name'])

    # --- Model, Optimizer, Scheduler ---
    model = get_model(config, data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    criterion = torch.nn.CrossEntropyLoss() if data.y.dim() == 1 else torch.nn.BCEWithLogitsLoss()

    # --- Logging Setup ---
    wandb.init(project=config['globals']['wandb_project'], config=config, name=run_id, reinit=True)
    
    best_val_metric = -1
    patience_counter = 0
    total_start_time = time.time()
    power_logger = PowerLogger()
    power_logger.start()

    # --- FLOPs Calculation ---
    try:
        macs, params = get_model_complexity_info(model, (data.x, data.edge_index), as_strings=False, print_per_layer_stat=False, verbose=False)
        flops = 2 * macs # approximation
        wandb.log({'params': params, 'flops_per_forward': flops})
    except Exception as e:
        print(f"Could not compute FLOPs: {e}")
        flops = 0

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device), epoch=epoch)
        
        train_mask = data.train_mask.to(device)
        loss = criterion(out[train_mask], data.y[train_mask].to(device).float() if isinstance(criterion, torch.nn.BCEWithLogitsLoss) else data.y[train_mask].to(device))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # --- Validation ---
        val_metrics = evaluate_model(model, data, data.val_mask, device)
        train_metrics = evaluate_model(model, data, data.train_mask, device)
        metric_key = 'roc_auc' if 'roc_auc' in val_metrics and val_metrics['roc_auc'] > 0 else 'accuracy'
        val_metric = val_metrics[metric_key]
        
        # --- Checkpointing ---
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['globals']['log_dir'], f'{run_id}_best_model.pt'))
        else:
            patience_counter += 1
        
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch}")
            break

        log_dict = {
            'epoch': epoch,
            'train_loss': loss.item(),
            f'train_{metric_key}': train_metrics[metric_key],
            f'val_{metric_key}': val_metric,
        }
        if hasattr(model, 'layers') and hasattr(model.layers[0], 'hub_message_count'):
            log_dict['hub_msgs'] = sum(l.hub_message_count for l in model.layers if hasattr(l, 'hub_message_count'))
        wandb.log(log_dict)

    # --- Final Evaluation ---
    model.load_state_dict(torch.load(os.path.join(config['globals']['log_dir'], f'{run_id}_best_model.pt')))
    test_metrics = evaluate_model(model, data, data.test_mask, device)
    total_time_taken = time.time() - total_start_time
    total_energy_joules = power_logger.stop()

    final_results = {
        'run_id': run_id,
        'final_accuracy': test_metrics.get('accuracy', 0),
        'final_roc_auc': test_metrics.get('roc_auc', 0),
        'final_f1_macro': test_metrics.get('f1_macro', 0),
        'final_ece': test_metrics.get('ece', 0),
        'best_val_metric': best_val_metric,
        'total_time': total_time_taken,
        'total_energy': total_energy_joules,
        'peak_memory_mb': torch.cuda.max_memory_allocated(device) / (1024 * 1024),
    }
    if hasattr(model, 'layers') and hasattr(model.layers[0], 'hub_message_count'):
        final_results['final_hub_msgs'] = sum(l.hub_message_count for l in model.layers if hasattr(l, 'hub_message_count'))

    wandb.log(final_results)
    wandb.finish()
    torch.cuda.empty_cache()
    return final_results

# --- Main Orchestrator ---
def main():
    parser = argparse.ArgumentParser(description="Run SCoRe-GNN Experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run with smoke_test.yaml config.')
    group.add_argument('--full-experiment', action='store_true', help='Run with full_experiment.yaml config.')
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = get_device(config)
    output_dir = ".research/iteration1"
    config['globals']['log_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- EXPERIMENT 1: End-to-End Benchmark ---
    if 'experiment_1' in config:
        print("\n===== Starting Experiment 1: End-to-End Benchmark =====")
        exp1_results = []
        exp1_config = config['experiment_1']
        for dataset_name in exp1_config['datasets']:
            for model_name in exp1_config['models']:
                for seed in config['globals']['seeds']:
                    run_config = config.copy()
                    run_config['dataset'] = {'name': dataset_name}
                    run_config['model'] = {'name': model_name, **exp1_config.get('model_params', {})}
                    run_config['training'] = exp1_config['training']
                    run_config['seed'] = seed
                    run_id = f"exp1_{dataset_name}_{model_name}_seed{seed}"
                    print(f"\n--- Running: {run_id} ---")
                    
                    result = run_single_experiment(run_config, device, run_id)
                    exp1_results.append({**result, 'dataset': dataset_name, 'model': model_name, 'seed': seed})
        run_evaluation('experiment_1', exp1_results, output_dir)

    # --- EXPERIMENT 2: Controlled Study ---
    if 'experiment_2' in config:
        print("\n===== Starting Experiment 2: Controlled Study =====")
        exp2_results = []
        exp2_config = config['experiment_2']
        for d in exp2_config['d']:
            for h in exp2_config['h']:
                for p_rewire in exp2_config['p_rewire']:
                    for model_name in exp2_config['models']:
                        for seed in config['globals']['seeds']:
                            run_config = config.copy()
                            run_config['synthetic_params'] = {'d': d, 'h': h, 'p_rewire': p_rewire}
                            run_config['dataset'] = {'name': f'synth_d{d}_h{h}_p{p_rewire}'}
                            run_config['model'] = {'name': model_name, **exp2_config.get('model_params', {})}
                            run_config['training'] = exp2_config['training']
                            run_config['seed'] = seed
                            run_id = f"exp2_d{d}_h{h}_p{p_rewire}_{model_name}_seed{seed}"
                            print(f"\n--- Running: {run_id} ---")
                            result = run_single_experiment(run_config, device, run_id)
                            exp2_results.append({**result, **run_config['synthetic_params'], 'model': model_name, 'seed': seed})
        run_evaluation('experiment_2', exp2_results, output_dir)

    # --- EXPERIMENT 3: Ablation Study ---
    if 'experiment_3' in config:
        print("\n===== Starting Experiment 3: Ablation Study =====")
        exp3_results = []
        exp3_config = config['experiment_3']
        for dataset_name in exp3_config['datasets']:
            for variant in exp3_config['variants']:
                for lazy_T in exp3_config.get('sweep_lazy_T', [5]):
                    for conf_tau in exp3_config.get('sweep_conf_tau', [0.9]):
                        for seed in config['globals']['seeds']:
                            run_config = config.copy()
                            run_config['dataset'] = {'name': dataset_name}
                            model_params = {
                                'name': 'SCoRe-GNN',
                                'lazy_T': 0 if variant == 'no_calr' else lazy_T,
                                'q_bits': 0 if variant == 'no_qbala' else 3,
                                'conf_tau': conf_tau
                            }
                            if variant == 'no_calr_qbala':
                                model_params['lazy_T'] = 0
                                model_params['q_bits'] = 0
                            
                            run_config['model'] = model_params
                            run_config['training'] = exp3_config['training']
                            run_config['seed'] = seed
                            variant_name = f"{variant}_T{lazy_T}_tau{conf_tau}"
                            run_id = f"exp3_{dataset_name}_{variant_name}_seed{seed}"
                            print(f"\n--- Running: {run_id} ---")
                            result = run_single_experiment(run_config, device, run_id)
                            exp3_results.append({**result, 'dataset': dataset_name, 'variant': variant_name, 'seed': seed})
        run_evaluation('experiment_3', exp3_results, output_dir)

if __name__ == '__main__':
    main()