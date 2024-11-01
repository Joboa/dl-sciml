# Scripts/config.py

import numpy as np

CONFIG = {
    'data': {
        'file_path': './Input/burgers_shock.mat',
        'N_u': 100,  # Number of Initial and Boundary data points
        'N_f': 10000,  # Number of residual points
    },
    'training': {
        'max_epochs': 20000,
        'learning_rate': 1e-3,
        'nu': 0.01/np.pi,  # Viscosity
    },
    'model': {
        'layers': [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    },
    'paths': {
        'model_save': './Models/model_neuralnet1',
        'loss_save': './Models/training_loss.csv',
        'results': './Results'
    }
}