import argparse
import yaml
import os
import json
import torch
import random
import numpy as np
from . import train
from . import preprocess
from . import evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Run FLASH-TT experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config['run']['seed'])

    # --- 1. Prepare Source Model ---
    model_config = config['model']
    model_save_dir = os.path.join(config['run']['output_dir'], 'models')
    model_path = train.prepare_source_model(
        model_name=model_config['name'], 
        lora_rank=model_config.get('lora_rank'),
        save_dir=model_save_dir
    )

    # --- 2. Load Model ---
    model = train.timm.create_model(train.model_map.get(model_config['name'], model_config['name']), pretrained=False)
    if 'deit' in model_config['name'] and model_config.get('lora_rank', 0) > 0:
        model = train.add_lora_to_deit(model, rank=model_config['lora_rank'])
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- 3. Get Data Stream ---
    data_loader = preprocess.get_data_stream(config)

    # --- 4. Run Experiment ---
    results = evaluate.run_experiment(config, model, data_loader)
    
    # --- 5. Save Results ---
    results_dir = os.path.join(config['run']['output_dir'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    exp_name = f"{config['model']['name']}_{config['data']['name']}_{config['method']['name']}"
    if 'corruption' in config['data']:
        exp_name += f"_{config['data']['corruption']}_{config['data']['severity']}"
    exp_name += f"_seed{config['run']['seed']}"

    results_path = os.path.join(results_dir, f"{exp_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
        
    print(f"\nFull results saved to: {results_path}")

if __name__ == '__main__':
    main()
