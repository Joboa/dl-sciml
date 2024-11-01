# Scripts/data_utils.py

import scipy.io
import numpy as np
import torch
from pyDOE import lhs
from typing import Dict, Tuple

def load_burger_data(file_path: str, N_u: int, N_f: int) -> Dict[str, torch.Tensor]:
    """
    Load and preprocess data for Burger's equation
    
    Args:
        file_path: Path to the .mat file containing the data
        N_u: Number of training points for u
        N_f: Number of collocation points for f
    
    Returns:
        Dictionary containing preprocessed data tensors
    """
    data = scipy.io.loadmat(file_path)
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    # Initial Condition
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T

    # Boundary condition -1
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]

    # Boundary condition 1
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    # Random sampling
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # Convert to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    Exact = torch.tensor(Exact, dtype=torch.float32)
    X_u_train = torch.tensor(X_u_train, dtype=torch.float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    X_f_train = torch.tensor(X_f_train, dtype=torch.float32)
    
    # For testing/validation
    X_star = torch.tensor(X_star, dtype=torch.float32)
    u_star = torch.tensor(u_star, dtype=torch.float32)

    return {
        'X_u_train': X_u_train,
        'u_train': u_train,
        'X_f_train': X_f_train,
        'X_star': X_star,
        'u_star': u_star,
        'lb': lb,
        'ub': ub,
        'x': x,
        't': t,
        'Exact': Exact
    }