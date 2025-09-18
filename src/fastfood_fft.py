"""
Mock implementation of Fastfood FFT library
This is a stub implementation for testing purposes.
"""
import torch
import torch.nn as nn
import math


class Fastfood(nn.Module):
    """
    Mock implementation of Fastfood random projection.
    
    Fastfood is a method for fast random projections that uses structured
    matrices (Hadamard, diagonal, permutation) to achieve O(d log d) complexity
    instead of O(d^2) for dense random projections.
    
    This is a simplified mock that performs a structured transformation.
    """
    
    def __init__(self, input_dim, sigma=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        
        # Initialize structured matrices (simplified version)
        # In real Fastfood, this would be Hadamard matrices, permutations, etc.
        self.register_buffer('permutation', torch.randperm(input_dim))
        self.register_buffer('diagonal', torch.randn(input_dim))
        
        # For numerical stability
        self.eps = 1e-8
        
    def forward(self, x):
        """
        Apply Fastfood projection to input tensor x.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Transformed tensor of same shape
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")
        
        # Apply sigma scaling
        sigma_val = self.sigma.clamp(min=self.eps)
        x_scaled = x / sigma_val
        
        # Apply permutation
        x_perm = x_scaled[..., self.permutation]
        
        # Apply diagonal transformation
        x_diag = x_perm * self.diagonal
        
        # Apply Walsh-Hadamard-like transformation (simplified)
        # Real Fastfood would use proper Walsh-Hadamard transform
        x_transformed = self._walsh_hadamard_transform(x_diag)
        
        # Apply another permutation and diagonal
        x_perm2 = x_transformed[..., torch.flip(self.permutation, [0])]
        x_final = x_perm2 * torch.flip(self.diagonal, [0])
        
        return x_final
    
    def _walsh_hadamard_transform(self, x):
        """
        Simplified Walsh-Hadamard transform approximation.
        This is a mock - real implementation would use proper WHT.
        """
        # For simplicity, we'll use a DCT-like transform as approximation
        if x.dim() == 1:
            return torch.fft.dct(x, norm='ortho')
        else:
            # Apply along last dimension
            return torch.fft.dct(x, norm='ortho', dim=-1)
    
    def extra_repr(self):
        return f'input_dim={self.input_dim}, sigma={self.sigma.item():.4f}'