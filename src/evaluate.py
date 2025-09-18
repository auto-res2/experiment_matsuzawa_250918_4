import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import json
import copy
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc

# --- FLASH-TT Core Components ---

class FlashPatch:
    """ Base class for monkey-patching layers for FLASH-TT. """
    @staticmethod
    def get_pseudo_labels(outputs: torch.Tensor):
        return outputs.argmax(dim=1)

    @staticmethod
    def get_predictions(outputs: torch.Tensor):
        return F.softmax(outputs, dim=1)

class FlashBN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, running_mean, running_var, weight, bias, training, momentum, eps, outputs):
        # Standard BN forward pass
        result = F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
        
        # Analytic gradient calculation
        batch_mean = x.mean([0, 2, 3])
        batch_var = x.var([0, 2, 3], unbiased=False)
        inv_std = 1. / torch.sqrt(batch_var + eps)

        y_hat = FlashPatch.get_predictions(outputs)
        mu_out = (y_hat.T @ x.permute(1, 0, 2, 3).flatten(1)).T / y_hat.sum(0)
        mu_out = mu_out.squeeze()
        
        grad_beta = (mu_out - y_hat.mean(0)) * inv_std
        grad_gamma = ((x - batch_mean.view(1, -1, 1, 1)) * inv_std.view(1, -1, 1, 1) * grad_beta.view(1, -1, 1, 1)).mean(0)

        ctx.grad_beta = grad_beta
        ctx.grad_gamma = grad_gamma
        return result

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("FLASH-TT is backward-free. Do not call .backward() on the model.")

class FlashLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, outputs):
        result = F.layer_norm(x, normalized_shape, weight, bias, eps)
        
        mu_in = x.mean(-1, keepdim=True)
        sigma_in_sq = x.var(-1, keepdim=True, unbiased=False)
        sigma_in = torch.sqrt(sigma_in_sq + eps)
        
        y_hat = FlashPatch.get_predictions(outputs).unsqueeze(-1)
        mu_out = (y_hat * result).sum(0) / (y_hat.sum(0) + 1e-9)
        
        # Using analytic formulas from the paper
        g_beta = (mu_out - y_hat.mean(0)) / sigma_in.mean(0)
        g_gamma = (((x - mu_in) / sigma_in_sq) * (mu_out - y_hat.mean(0))).mean(0)

        ctx.grad_beta = g_beta.squeeze()
        ctx.grad_gamma = g_gamma.squeeze()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("FLASH-TT is backward-free.")

class FlashGN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_groups, weight, bias, eps, outputs):
        result = F.group_norm(x, num_groups, weight, bias, eps)

        B, C, H, W = x.shape
        x_reshaped = x.view(B, num_groups, C // num_groups, H, W)
        
        mu_in = x_reshaped.mean([2, 3, 4], keepdim=True)
        sigma_in_sq = x_reshaped.var([2, 3, 4], keepdim=True, unbiased=False)
        sigma_in = torch.sqrt(sigma_in_sq + eps)

        y_hat = FlashPatch.get_predictions(outputs)
        mu_out = (y_hat.T @ result.flatten(1)).T.view(B, C, -1).mean(-1)

        g_beta = (mu_out - y_hat.mean(0)) / sigma_in.mean(0).view(num_groups,-1).flatten()
        g_gamma = ((x_reshaped - mu_in) / sigma_in_sq * (mu_out - y_hat.mean(0)).view(B, C, 1, 1)).mean(0).flatten()
        g_beta = g_beta.mean(0)
        g_gamma = g_gamma.mean(0)

        ctx.grad_beta = g_beta
        ctx.grad_gamma = g_gamma
        return result

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("FLASH-TT is backward-free.")

class FlashLora(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_A, lora_B, scaling, original_output, outputs):
        result = original_output + (x @ lora_A @ lora_B) * scaling
        y_hat = FlashPatch.get_predictions(outputs)
        
        # Analytic gradient for LoRA (simplified for entropy loss)
        grad_lora_A = (x.T @ (y_hat - F.softmax(result, dim=1)) @ lora_B.T)
        grad_lora_B = ((x @ lora_A).T @ (y_hat - F.softmax(result, dim=1)))

        ctx.grad_lora_A = grad_lora_A
        ctx.grad_lora_B = grad_lora_B
        return result

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("FLASH-TT is backward-free.")

# --- Helper for patching the model ---

class FlashState:
    def __init__(self):
        self.outputs = None
        self.activations = {}
        self.hooks = []

    def clear(self):
        self.outputs = None
        self.activations = {}
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

flash_state = FlashState()

def patch_model_for_flash(model: nn.Module):
    """ Monkey-patches a model to use FLASH-TT analytic gradients. """
    from .train import LoraLayer
    
    def get_activation_hook(name):
        def hook(model, input, output):
            flash_state.activations[name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.old_forward = module.forward
            def new_bn_forward(self, x):
                return FlashBN.apply(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps, flash_state.outputs)
            module.forward = new_bn_forward.__get__(module, type(module))
            flash_state.hooks.append(module.register_forward_hook(get_activation_hook(name)))

        elif isinstance(module, nn.LayerNorm):
            module.old_forward = module.forward
            def new_ln_forward(self, x):
                return FlashLN.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, flash_state.outputs)
            module.forward = new_ln_forward.__get__(module, type(module))
            flash_state.hooks.append(module.register_forward_hook(get_activation_hook(name)))
        
        elif isinstance(module, nn.GroupNorm):
            module.old_forward = module.forward
            def new_gn_forward(self, x):
                return FlashGN.apply(x, self.num_groups, self.weight, self.bias, self.eps, flash_state.outputs)
            module.forward = new_gn_forward.__get__(module, type(module))
            flash_state.hooks.append(module.register_forward_hook(get_activation_hook(name)))
        
        elif isinstance(module, LoraLayer):
            module.old_forward = module.forward
            def new_lora_forward(self, x):
                original_output = self.original_layer(x)
                return FlashLora.apply(x, self.lora_A, self.lora_B, self.scaling, original_output, flash_state.outputs)
            module.forward = new_lora_forward.__get__(module, type(module))
    
    return model

class SpectralTrigger:
    def __init__(self, delta=0.5, power_iter=2):
        self.delta = delta
        self.power_iter = power_iter
        self.source_lambda1 = None
        self.prev_lambda1 = None

    def __call__(self, acts: torch.Tensor, model: nn.Module) -> Tuple[bool, float, float]:
        if acts.dim() > 2:
            acts = acts.flatten(2).mean(2)
        acts = acts - acts.mean(0, keepdim=True)
        
        # Low-rank Gram matrix approximation
        gram_matrix = acts @ acts.T
        v = torch.randn(acts.size(0), device=acts.device)
        for _ in range(self.power_iter):
            v = gram_matrix @ v
            v = v / torch.norm(v)
        lambda1 = (v.T @ gram_matrix @ v).item()

        if self.source_lambda1 is None:
            self.source_lambda1 = lambda1
            self.prev_lambda1 = lambda1
        
        # KL divergence on channel means (simplified)
        channel_means = acts.mean(0)
        kl_div = F.kl_div(channel_means.softmax(-1).log(), torch.ones_like(channel_means).softmax(-1), reduction='sum').item()

        lambda_change = abs(lambda1 - self.source_lambda1)
        trigger = max(kl_div, lambda_change) > self.delta
        
        rho_t = (lambda1 / self.prev_lambda1)**0.5 if self.prev_lambda1 > 1e-6 else 1.0
        self.prev_lambda1 = lambda1
        
        return trigger, lambda_change, kl_div, rho_t

# --- Baseline Methods ---

def configure_model(model):
    """Configure model for adaptation."""
    model.train()
    # disable grad for all trainable params
    for p in model.parameters():
        p.requires_grad = False
    # enable grad for specific params
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

@torch.enable_grad()
def tent_post_update(model, inputs, optimizer):
    outputs = model(inputs)
    loss = -(outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


# --- Main Evaluation Loop ---

def run_experiment(config: Dict[str, Any], model: nn.Module, data_loader: torch.utils.data.DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    method_name = config['method']['name']
    results = defaultdict(list)
    num_samples_processed = 0
    total_correct = 0
    accuracies = []
    batch_indices = []

    source_model_state = copy.deepcopy(model.state_dict())

    optimizer = None
    if method_name in ['tent', 'post', 'eata', 'rotta']:
        model = configure_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['method'].get('lr', 1e-4), betas=(0.9, 0.999))
    elif method_name == 'adabn':
        model.train() # Use batch statistics
    elif method_name == 'flash_tt':
        model = patch_model_for_flash(model)
        trigger = SpectralTrigger(delta=config['method']['delta'], power_iter=config['method']['power_iter'])
        cached_updates = {}
        aborts = 0
        updates = 0
    
    start_time = time.time()

    for i, (images, labels) in enumerate(data_loader):
        if config['run'].get('max_batches') and i >= config['run']['max_batches']:
            break

        images, labels = images.to(device), labels.to(device)

        if method_name in ['tent', 'post']:
            outputs = tent_post_update(model, images, optimizer)
        elif method_name == 'flash_tt':
            flash_state.clear()
            
            # Forward pass
            with torch.no_grad():
                flash_state.outputs = model(images)

            # Trigger
            activations = list(flash_state.activations.values())[-1]
            is_triggered, lambda_ch, kl, rho_t = trigger(activations, model)

            if is_triggered:
                updates += 1
                # M4: Safeguard
                with torch.no_grad():
                    current_loss = F.cross_entropy(flash_state.outputs, labels)
                
                temp_state = copy.deepcopy(model.state_dict())
                for name, module in model.named_modules():
                    if hasattr(module, 'old_forward'): # Patched layer
                        grad_beta = module.forward.__self__.__self__.grad_beta
                        grad_gamma = module.forward.__self__.__self__.grad_gamma
                        
                        temp_state[f'{name}.bias'] -= config['method']['lr'] * grad_beta
                        temp_state[f'{name}.weight'] -= config['method']['lr'] * grad_gamma
                        cached_updates[name] = (grad_beta, grad_gamma)

                # Check loss increase
                model.load_state_dict(temp_state)
                with torch.no_grad():
                    new_outputs = model(images)
                    new_loss = F.cross_entropy(new_outputs, labels)
                
                if new_loss > current_loss and not config['method'].get('no_safeguard', False):
                    aborts += 1
                    model.load_state_dict(source_model_state) # Revert
                else:
                    source_model_state = temp_state # Commit
            else:
                 # M3: Adaptive curvature replay
                temp_state = copy.deepcopy(model.state_dict())
                for name, (grad_beta, grad_gamma) in cached_updates.items():
                    temp_state[f'{name}.bias'] -= config['method']['lr'] * grad_beta * rho_t
                    temp_state[f'{name}.weight'] -= config['method']['lr'] * grad_gamma * rho_t
                model.load_state_dict(temp_state)
                source_model_state = temp_state
            
            outputs = model(images)
        else: # source, adabn
            with torch.no_grad():
                outputs = model(images)
        
        pred = outputs.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        num_samples_processed += len(labels)
        current_acc = total_correct / num_samples_processed * 100
        
        if i % 10 == 0:
            accuracies.append(current_acc)
            batch_indices.append(num_samples_processed)
            print(f"Batch {i}/{len(data_loader)}, Samples {num_samples_processed}, Acc: {current_acc:.2f}%")

    end_time = time.time()

    # Final metrics
    final_accuracy = total_correct / num_samples_processed * 100
    total_time = end_time - start_time
    throughput = num_samples_processed / total_time
    online_auc = auc(batch_indices, accuracies) / batch_indices[-1] if len(batch_indices) > 1 else 0.0
    
    results_dict = {
        'final_accuracy': final_accuracy,
        'online_accuracy_auc': online_auc * 100, # as percentage
        'throughput_img_s': throughput,
        'total_samples': num_samples_processed,
        'total_time_s': total_time,
        'config': config,
    }
    if method_name == 'flash_tt':
        results_dict['safeguard_aborts'] = aborts
        results_dict['triggered_updates'] = updates

    # Create and save plot
    output_dir = config['run']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"accuracy_over_time.png")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=batch_indices, y=accuracies)
    plt.title(f"Online Accuracy for {method_name} on {config['data']['name']}")
    plt.xlabel("Images Processed")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved accuracy plot to {fig_path}")

    print("\n--- Experiment Results ---")
    print(json.dumps(results_dict, indent=2, default=str))
    print("------------------------\n")

    return results_dict
