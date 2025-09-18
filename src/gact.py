"""
Mock implementation of GACT (Gradient Compressed Training) library
This is a stub implementation for testing purposes.
"""
import torch
import logging

# Set up logging to suppress unnecessary output
logging.basicConfig(level=logging.WARNING)

def set_optimization_level(level='L2'):
    """Mock function to set GACT optimization level"""
    pass

class GACTFunc:
    """Mock GACT function utilities"""
    
    @staticmethod
    def quantize(tensor, bits=3):
        """
        Mock quantization function.
        For now, returns the original tensor without quantization.
        """
        if bits <= 0:
            return tensor
        
        # Simple quantization mock - clamp values and add some noise to simulate compression
        # In a real implementation, this would perform proper bit quantization
        with torch.no_grad():
            # Simulate quantization by discretizing values
            max_val = tensor.abs().max()
            if max_val == 0:
                return tensor
            
            # Normalize to [-1, 1]
            normalized = tensor / max_val
            
            # Quantize to specified bits (simplified)
            levels = 2 ** bits - 1
            quantized = torch.round(normalized * levels) / levels
            
            # Scale back
            result = quantized * max_val
            
            # Add small amount of noise to simulate compression artifacts
            noise_scale = max_val * 0.001  # 0.1% noise
            noise = torch.randn_like(result) * noise_scale
            result = result + noise
            
            return result.to(tensor.dtype)

# Create the func attribute
func = GACTFunc()