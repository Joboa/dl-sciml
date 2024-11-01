# Scripts/train.py

import torch
import numpy as np
from typing import Dict, List
import time
import pandas as pd
from models import initialize_network, BurgersPINN
from data_utils import load_burger_data
from config import CONFIG
import os

def train_model(model: BurgersPINN, 
                optimizer: torch.optim.Optimizer,
                data: Dict[str, torch.Tensor],
                config: Dict) -> List[float]:
    """
    Train the PINN model
    
    Args:
        model: The PINN model
        optimizer: The optimizer
        data: Dictionary containing training data
        config: Configuration dictionary
    
    Returns:
        List of loss values during training
    """
    losses = []
    start_time = time.time()
    
    for epoch in range(config['training']['max_epochs']):
        def closure():
            optimizer.zero_grad()
            u_pred = model(data['X_u_train'][:, 0:1], data['X_u_train'][:, 1:2])
            f_pred = model.compute_residual(data['X_f_train'][:, 0:1], 
                                           data['X_f_train'][:, 1:2])
            
            mse_u = torch.mean((data['u_train'] - u_pred) ** 2)
            mse_f = torch.mean(f_pred ** 2)
            
            loss = mse_u + mse_f
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.6f}')
    
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.4f} seconds')
    
    return losses

def save_results(model: BurgersPINN, losses: List[float], config: Dict):
    """Save model and training history"""
    # Save loss values
    loss_df = pd.DataFrame(losses, columns=['Loss'])
    loss_df.to_csv(config['paths']['loss_save'], index=False)
    print(f"Training loss saved to {config['paths']['loss_save']}")
    
    # Save model
    os.makedirs(config['paths']['model_save'], exist_ok=True)
    model_path = os.path.join(config['paths']['model_save'], 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_history': losses
    }, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Load data
    data = load_burger_data(
        CONFIG['data']['file_path'],
        CONFIG['data']['N_u'],
        CONFIG['data']['N_f']
    )
    
    # Initialize model
    model, optimizer = initialize_network(
        CONFIG['model']['layers'],
        CONFIG['training']['nu']
    )
    
    # Train model
    losses = train_model(model, optimizer, data, CONFIG)
    
    # Save results
    save_results(model, losses, CONFIG)

if __name__ == "__main__":
    main()