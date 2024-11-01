# Scripts/models.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

def hyper_initial(size: Tuple[int, int]) -> nn.Parameter:
    """
    Xavier/ Glorot Normal Initialization for network weights
    """
    in_dim, out_dim = size
    std = torch.sqrt(torch.tensor(2.0/(in_dim + out_dim)))
    return nn.Parameter(torch.randn(size) * std)

class BurgersPINN(nn.Module):
    def __init__(self, layers: List[int], nu: float = 0.01/np.pi):
        """
        Physics-Informed Neural Network for Burgers' equation
        
        Args:
            layers: List of integers defining the network architecture
            nu: Viscosity parameter
        """
        super(BurgersPINN, self).__init__()
        self.L = len(layers)
        self.W = nn.ParameterList([hyper_initial([layers[l-1], layers[l]]) 
                                  for l in range(1, self.L)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros([1, layers[l]])) 
                                  for l in range(1, self.L)])
        self.nu = nu
    
    def DNN(self, X: torch.Tensor) -> torch.Tensor:
        """Deep Neural Network forward pass"""
        A = X
        for i in range(self.L - 2):
            A = torch.tanh(torch.matmul(A, self.W[i]) + self.b[i])
        Y = torch.matmul(A, self.W[-1]) + self.b[-1]
        return Y
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute u(x,t)"""
        input_tensor = torch.cat([x, t], dim=1)
        u = self.DNN(input_tensor)
        return u
    
    def compute_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the physics-informed residual (f) for Burgers' equation
        
        Args:
            x: Spatial coordinates tensor
            t: Time coordinates tensor
        
        Returns:
            f: Residual of the PDE
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        
        # First-order derivatives
        u_t = torch.autograd.grad(u, t, 
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
        u_x = torch.autograd.grad(u, x,
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
        
        # Second-order derivative
        u_xx = torch.autograd.grad(u_x, x,
                                  grad_outputs=torch.ones_like(u_x),
                                  create_graph=True)[0]
        
        # Burgers' equation residual
        f = u_t + u * u_x - self.nu * u_xx
        
        return f

def initialize_network(layers: List[int], nu: float = 0.01/np.pi) -> Tuple[BurgersPINN, torch.optim.Optimizer]:
    """Initialize the PINN and its optimizer"""
    model = BurgersPINN(layers, nu)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer

# Example usage
if __name__ == "__main__":
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model, optimizer = initialize_network(layers)
    
    # Example forward pass and residual computation
    batch_size = 100
    x = torch.randn(batch_size, 1)
    t = torch.randn(batch_size, 1)
    
    u = model(x, t)
    f = model.compute_residual(x, t)
    
    print(f"Solution u shape: {u.shape}")
    print(f"Residual f shape: {f.shape}")