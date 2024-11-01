# Scripts/inference.py
# Inference is the process of running a training model

import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models import BurgersPINN, initialize_network
from data_utils import load_burger_data
from config import CONFIG
import os

def predict(model: BurgersPINN, X_star: torch.Tensor) -> np.ndarray:
    """Make predictions using the trained model"""
    model.eval()
    with torch.no_grad():
        x_star = X_star[:, 0:1]
        t_star = X_star[:, 1:2]
        u_pred = model(x_star, t_star)
    return u_pred.numpy()

def compute_error(u_pred: np.ndarray, u_star: np.ndarray) -> float:
    """Compute relative L2 error"""
    return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

def plot_results(x: np.ndarray, t: np.ndarray, 
                U_pred: np.ndarray, Exact: np.ndarray, 
                X_u_train: np.ndarray, error: float,
                save_path: str):
    """Plot and save comparison between true and predicted solutions"""
    
    fig = plt.figure(figsize=(12, 8))
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', 
            label=f'Data ({X_u_train.shape[0]} points)', 
            markersize=4, clip_on=False)
    
    for t_i in [0.25, 0.5, 0.75]:
        line = np.linspace(x.min(), x.max(), 2)[:,None]
        ax.plot(t_i*np.ones((2,1)), line, 'w-', linewidth=1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title(f'$u(t,x)$ - Relative L2 Error: {error:.2e}', fontsize=10)
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    for i, t_i in enumerate([0.25, 0.5, 0.75]):
        ax = plt.subplot(gs1[0, i])
        idx = int(t_i * 100)
        ax.plot(x, Exact[idx,:], 'b-', linewidth=2, label='Exact')       
        ax.plot(x, U_pred[idx,:], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_title(f'$t = {t_i}$', fontsize=10)
        ax.axis('square')
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])
        if i == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), 
                     ncol=5, frameon=False)
    
    # fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_results.png'))
    plt.close()

def main():
    # Load data
    data = load_burger_data(
        CONFIG['data']['file_path'],
        CONFIG['data']['N_u'],
        CONFIG['data']['N_f']
    )

    # print(data.keys())
    
    # Load trained model
    model, _ = initialize_network(CONFIG['model']['layers'])
    checkpoint = torch.load(os.path.join(CONFIG['paths']['model_save'], 'model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Make predictions
    u_pred = predict(model, data['X_star'])
    
    # Compute error
    error = compute_error(u_pred, data['u_star'].numpy())
    print(f'Relative L2 Error: {error:.2e}')
    
    # Reshape predictions for plotting
    X, T = np.meshgrid(data['x'].flatten(), data['t'].flatten())
    U_pred = griddata(data['X_star'].numpy(), u_pred.flatten(), 
                      (X, T), method='cubic')
    
    # Plot results
    plot_results(data['x'].numpy(), data['t'].numpy(),
                U_pred, data['Exact'].numpy(),
                data['X_u_train'].numpy(), error,
                CONFIG['paths']['results'])

if __name__ == "__main__":
    main()