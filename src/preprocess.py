import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
import timm
import numpy as np
from PIL import Image
import math

def get_transforms(model_name: str):
    model_map = {
        'resnet50': 'resnet50.a1_in1k',
        'convnext_tiny': 'convnext_tiny.in12k_ft_in1k',
        'deit_small': 'deit_small_patch16_224.fb_in1k',
        'mobilevit_xs': 'mobilevit_xs.cvdf_in1k',
    }
    timm_model_name = model_map.get(model_name, model_name)
    
    # Create a dummy model to get its data config
    model = timm.create_model(timm_model_name, pretrained=False)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    if 'cifar' in model_name.lower(): # Special handling for CIFAR
        # Enlarge CIFAR images to match ImageNet resolution
        transform.transforms.insert(0, transforms.Resize(data_config['input_size'][1:], interpolation=transforms.InterpolationMode.BICUBIC))
        transform.transforms.insert(0, transforms.Pad(4))

    return transform

class ContrastShiftTransform:
    def __init__(self, alpha_schedule):
        self.alpha_schedule = alpha_schedule
        self.step = 0

    def __call__(self, img_tensor):
        if self.step >= len(self.alpha_schedule):
            alpha = self.alpha_schedule[-1]
        else:
            alpha = self.alpha_schedule[self.step]
        
        mean = img_tensor.mean(dim=[1, 2], keepdim=True)
        shifted = (1 + alpha) * (img_tensor - mean) + mean
        self.step += 1
        return torch.clamp(shifted, 0, 1)

class StyleTransferTransform:
    """ Simulates texture swap by swapping FFT phase components. """
    def __init__(self, style_source_dataset):
        self.style_source = style_source_dataset
        self.transform = get_transforms('resnet50') # A generic transform

    def __call__(self, img_tensor):
        style_idx = np.random.randint(0, len(self.style_source))
        style_img, _ = self.style_source[style_idx]
        style_tensor = self.transform(style_img)

        # FFT-based style transfer
        content_fft = torch.fft.fft2(img_tensor)
        style_fft = torch.fft.fft2(style_tensor)

        content_amp, content_phase = torch.abs(content_fft), torch.angle(content_fft)
        style_amp, style_phase = torch.abs(style_fft), torch.angle(style_fft)

        # Swap phase
        stylized_fft = content_amp * torch.exp(1j * style_phase)
        stylized_img = torch.fft.ifft2(stylized_fft).real
        
        # Normalize to original range
        stylized_img = (stylized_img - stylized_img.min()) / (stylized_img.max() - stylized_img.min())
        return stylized_img

class SyntheticStreamDataset(Dataset):
    def __init__(self, datasets, schedule, length):
        self.datasets = datasets
        self.schedule = schedule
        self.length = length
        self.dataset_lengths = [len(d) for d in datasets]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        progress = idx / self.length
        # Find the current active dataset based on schedule
        source_idx = np.random.choice(len(self.datasets), p=self.schedule(progress))
        inner_idx = idx % self.dataset_lengths[source_idx]
        return self.datasets[source_idx][inner_idx]

class LabelPermutationDataset(Dataset):
    def __init__(self, base_dataset, permute_count):
        self.base_dataset = base_dataset
        self.permute_count = permute_count
        num_classes = max(y for _, y in base_dataset) + 1
        self.permutation = torch.randperm(num_classes)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if idx < self.permute_count:
            label = self.permutation[label].item()
        return img, label

def get_data_stream(config):
    dataset_name = config['data']['name']
    model_name = config['model']['name']
    batch_size = config['run']['batch_size']
    
    transform = get_transforms(model_name)

    def apply_transform(batch):
        batch['image'] = [transform(img.convert('RGB')) for img in batch['image']]
        return batch

    if 'cifar10-c' in dataset_name:
        corruption = config['data']['corruption']
        severity = config['data']['severity']
        dataset = load_dataset("randall-lab/cifar10-c", corruption, split=f"severity_{severity}", trust_remote_code=True)
    elif 'imagenet-c' in dataset_name:
        corruption = config['data']['corruption']
        severity = config['data']['severity']
        # HF doesn't have a clean ImageNet-C, so we simulate with an existing corruption dataset
        dataset = load_dataset("imagenet_c", corruption, split=f"validation_severity_{severity}", trust_remote_code=True)
    elif 'imagenet-v2' in dataset_name:
        dataset = load_dataset("vaishaal/ImageNetV2-MatchedFrequency", split="test")
        if config['data'].get('permute_labels', False):
            dataset.set_transform(apply_transform)
            permute_count = config['data']['permute_count']
            # Convert to list for LabelPermutationDataset
            mapped_dataset = [(item['image'], item['label']) for item in dataset]
            dataset = LabelPermutationDataset(mapped_dataset, permute_count)
            # Re-wrap to use collate_fn
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif 'synthetic' in dataset_name:
        # Load base dataset (e.g., ImageNet clean val set)
        base_dataset = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
        base_dataset.set_transform(lambda x: {'image': x['image'], 'label': x['label']})
        
        if 'contrast' in config['data']['shift_type']:
            total_steps = config['run']['max_batches']
            alpha_schedule = np.linspace(0, 0.4, total_steps)
            contrast_transform = ContrastShiftTransform(alpha_schedule)
            final_transform = transforms.Compose([transform, contrast_transform])
        elif 'style' in config['data']['shift_type']:
            style_source = load_dataset("cifar10", split="test")
            style_transform = StyleTransferTransform(style_source)
            final_transform = transforms.Compose([transform, style_transform])
        else: # spectral drift
            fog_ds = load_dataset("imagenet_c", 'fog', split='validation_severity_5', trust_remote_code=True)
            noise_ds = load_dataset("imagenet_c", 'gaussian_noise', split='validation_severity_5', trust_remote_code=True)
            three_d_ds = load_dataset("imagenet_c", 'jpeg_compression', split='validation_severity_5', trust_remote_code=True)
            datasets = [fog_ds, noise_ds, three_d_ds]
            for ds in datasets:
                ds.set_transform(apply_transform)
            
            def schedule(p):
                if p < 0.33:
                    return [1.0, 0.0, 0.0]
                elif p < 0.66:
                    return [0.0, 1.0, 0.0]
                else:
                    return [0.0, 0.0, 1.0]
            
            dataset = SyntheticStreamDataset(datasets, schedule, length=50000)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        def apply_final_transform(batch):
            batch['image'] = [final_transform(img.convert('RGB')) for img in batch['image']]
            return batch
        base_dataset.set_transform(apply_final_transform)
        dataset = base_dataset

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    dataset.set_transform(apply_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
