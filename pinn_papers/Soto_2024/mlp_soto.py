# Soto et al (2024): Pytorch version
# Credits to the original author: The original code is in Tensorflow
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, TensorDataset

# GPU Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using PyTorch version: {torch.__version__}, Device: {device}')

# Read data (same as original)
data = scipy.io.loadmat('data/Case 1/Pinball_PIV.mat')

# Data loading and preprocessing (similar to original)
T_PIV = data['T_PIV']
noise_level = 0
np.random.seed(0)
U_PIV = data['U_PIV'] + np.random.normal(0, noise_level, size=data['U_PIV'].shape)
np.random.seed(1)
V_PIV = data['V_PIV'] + np.random.normal(0, noise_level, size=data['V_PIV'].shape)

# Load other data similarly...
np.random.seed(2)
U_probe = data['U_probe'] + np.random.normal(0, noise_level, size=data['U_probe'].shape)
np.random.seed(3)
V_probe = data['V_probe'] + np.random.normal(0, noise_level, size=data['V_probe'].shape)
np.random.seed(4)
MLP_PIV = data['MLP_PIV'] + np.random.normal(0, noise_level, size=data['MLP_PIV'].shape)
np.random.seed(5)
MLP_PIV_test = data['MLP_PIV_test'] + np.random.normal(0, noise_level, size=data['MLP_PIV_test'].shape)

# Test data loading
U_test_probe = data['U_probe_test']
V_test_probe = data['V_probe_test']
U_test_PIV = data['U_PIV_test']
V_test_PIV = data['V_PIV_test']
T_test_PIV = data['T_PIV_test']
X_test_PIV = data['X_PIV_test']
Y_test_PIV = data['Y_PIV_test']

del data

# NaN handling (same as original)
u_mean_PIV = np.mean(U_PIV, axis=1)[:, None]
U_nan_index = np.argwhere(np.isnan(u_mean_PIV))
U_PIV = np.delete(U_PIV, U_nan_index[:, 0], axis=0)
V_PIV = np.delete(V_PIV, U_nan_index[:, 0], axis=0)
U_test_PIV = np.delete(U_test_PIV, U_nan_index[:, 0], axis = 0)
V_test_PIV = np.delete(V_test_PIV, U_nan_index[:, 0], axis = 0)
T_test_PIV = np.delete(T_test_PIV, U_nan_index[:, 0], axis = 0)
X_test_PIV = np.delete(X_test_PIV, U_nan_index[:, 0], axis = 0)
Y_test_PIV = np.delete(Y_test_PIV, U_nan_index[:, 0], axis = 0)

# Dimensions
dim_T_PIV = U_PIV.shape[1]
dim_N_PIV = U_PIV.shape[0]
dim_N_probe = U_probe.shape[0]
dim_N_probe = U_probe.shape[0]
dim_T_test_PIV = U_test_PIV.shape[1]
dim_N_test_PIV = U_test_PIV.shape[0]

# Remove NaN from PIV data
for loop_nan in range(0, 2, 1):
    U_nan_index = np.argwhere(np.isnan(U_PIV))
    U_PIV[U_nan_index[:, 0],  U_nan_index[:, 1]] = U_PIV[U_nan_index[:, 0], U_nan_index[:, 1] - 1]
    V_nan_index = np.argwhere(np.isnan(V_PIV))
    V_PIV[V_nan_index[:, 0],  V_nan_index[:, 1]] = V_PIV[V_nan_index[:, 0], V_nan_index[:, 1] - 1]
    
    U_nan_index = np.argwhere(np.isnan(U_test_PIV))
    U_test_PIV[U_nan_index[:, 0],  U_nan_index[:, 1]] = U_test_PIV[U_nan_index[:, 0], U_nan_index[:, 1] - 1]
    V_nan_index = np.argwhere(np.isnan(V_test_PIV))
    V_test_PIV[V_nan_index[:, 0],  V_nan_index[:, 1]] = V_test_PIV[V_nan_index[:, 0], V_nan_index[:, 1] - 1]

print('Double-check for NaN in PIV data', np.sum(np.isnan(U_PIV)))
print('Double-check for NaN in test data', np.sum(np.isnan(U_test_PIV)))

# POD most energetic modes extraction
def energy_modes(sig_PIV, threshold):
    energy = np.zeros((sig_PIV.shape[0], 1))
    for index in range(sig_PIV.shape[0]):
        energy[index, 0] = np.sum(sig_PIV[:index] ** 2) / np.sum(sig_PIV ** 2)  
    return np.argwhere(energy > threshold)[0, 0]

# POD of PIV data
u_mean_PIV = torch.mean(torch.tensor(U_PIV), dim=1, keepdim=True)
v_mean_PIV = torch.mean(torch.tensor(V_PIV), dim=1, keepdim=True)

# Concatenate the fluctuating components
uvw_PIV = torch.cat([
    (torch.tensor(U_PIV) - u_mean_PIV).T,
    (torch.tensor(V_PIV) - v_mean_PIV).T
], dim=1)

# Perform SVD using PyTorch
psi_PIV, sig_PIV, phiT_PIV = torch.linalg.svd(uvw_PIV, full_matrices=False)

# Convert to numpy for compatibility with existing energy_modes function
psi_PIV = psi_PIV.numpy()
sig_PIV = sig_PIV.numpy()
phiT_PIV = phiT_PIV.numpy()

# Eliminate higher energy modes
print('Original modes = ', len(sig_PIV))
N_modes = energy_modes(sig_PIV, 0.9)
print('Elbow method = ', N_modes)

# Reduce modes
psi_PIV_red = psi_PIV[:, :N_modes]
sig_PIV_red = sig_PIV[:N_modes]
phiT_PIV_red = phiT_PIV[:N_modes, :]

# Calculate reduced amplitude matrix
A_PIV_red = psi_PIV_red @ np.diag(sig_PIV_red)

# POD test data
uvw_test_PIV = torch.cat([
    (torch.tensor(U_test_PIV) - u_mean_PIV).T,
    (torch.tensor(V_test_PIV) - v_mean_PIV).T
], dim=1)

# Convert to numpy for matrix operations
uvw_test_PIV = uvw_test_PIV.numpy()
psi_test_PIV = np.array(uvw_test_PIV @ phiT_PIV.T @ np.diag(sig_PIV ** (-1)))
A_test_PIV = psi_test_PIV[:, :N_modes] @ np.diag(sig_PIV_red)

# Cleanup
del T_PIV, U_PIV, V_PIV

# Define the PyTorch model
class MLPModel(nn.Module):
    def __init__(self, n_features, n_steps, N_modes):
        super(MLPModel, self).__init__()
        self.neurons = int(np.ceil(np.sqrt(n_steps * n_features * N_modes)))
        
        self.layers = nn.Sequential(
            nn.Linear(n_steps * n_features, self.neurons),
            nn.Tanh(),
            nn.Linear(self.neurons, self.neurons),
            nn.Sigmoid(),
            nn.Linear(self.neurons, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, N_modes)
        )

    def forward(self, x):
        return self.layers(x)

# Custom loss functions
class TemporalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Calculate MSE
        mse_loss = self.mse(pred, target)
        
        # Calculate correlation for each batch item
        corr = torch.zeros(1, pred.shape[0], device=pred.device)
        for idx in range(pred.shape[0]):
            # Stack the predictions and targets for correlation calculation
            stacked = torch.stack([pred[idx], target[idx]])
            corr[0, idx] = torch.corrcoef(stacked)[0, 1]
        
        return mse_loss / torch.mean(torch.abs(corr))

# Initialize model, optimizer, and loss function
n_features = 2
n_steps = dim_N_probe * 150
N_modes = N_modes
model = MLPModel(n_features, n_steps, N_modes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = TemporalLoss()

# Training tracking
train_loss_MLP = []
num_epochs = 1000
snap_batch = int(np.floor(MLP_PIV.shape[0] / 10))  # Batch size

epoch_test = torch.zeros(1, num_epochs)
temporal = torch.zeros(1, num_epochs)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    
    # Randomize data
    idx_epoch = torch.randperm(MLP_PIV.shape[0])
    MLP_PIV_epoch = MLP_PIV[idx_epoch]
    A_PIV_epoch = A_PIV_red[idx_epoch]
    
    # Batch training
    for snap in range(0, MLP_PIV.shape[0], snap_batch):
        # Prepare batch data
        MLP_PIV_batch = torch.FloatTensor(
            MLP_PIV_epoch[snap:snap + snap_batch]).to(device)
        A_PIV_red_batch = torch.FloatTensor(
            A_PIV_epoch[snap:snap + snap_batch]).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(MLP_PIV_batch)
        loss = criterion(outputs, A_PIV_red_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    # Calculate average epoch loss
    avg_loss = np.mean(epoch_losses)
    train_loss_MLP.append(avg_loss)
    
    # Adaptive learning rate
    if avg_loss > 5e-2:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    elif avg_loss > 5e-4:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    elif avg_loss > 5e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-6
    elif avg_loss > 5e-7:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-7
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-8
    
    # Testing phase
    model.eval()
    with torch.no_grad():
        A_test = torch.zeros(dim_T_test_PIV, N_modes)
        corr = torch.zeros(1, dim_T_test_PIV)
        
        for snap in range(dim_T_test_PIV):
            MLP_out = torch.FloatTensor(
                MLP_PIV_test[snap:snap + 1]).to(device)
            Y_out = model(MLP_out)
            A_test[snap:snap + 1] = Y_out.cpu()
        
        # Calculate correlations
        for idx in range(A_test.shape[0]):
            stacked = torch.stack([
                torch.tensor(A_test_PIV[idx]),
                A_test[idx]
            ])
            corr[0, idx] = torch.corrcoef(stacked)[0, 1]
        
        epoch_test[0, epoch] = torch.mean(corr)
    
    print(f"Epoch: {epoch} Loss_training: {avg_loss:.3e} Test: {epoch_test[0, epoch]:.3e}")
    
    # Save results at the end of training
    if (epoch + 1) % num_epochs == 0:
        model.eval()
        with torch.no_grad():
            A_TR = torch.zeros(dim_T_test_PIV, N_modes)
            
            for snap in range(0, dim_T_test_PIV):
                MLP_out = torch.FloatTensor(
                    MLP_PIV_test[snap:snap + 1]).to(device)
                Y_out = model(MLP_out)
                A_TR[snap:snap + 1] = Y_out.cpu()
            
            # Field reconstruction
            uvw_pred = torch.mm(A_TR.float(), torch.tensor(phiT_PIV_red).float())
            
            # Split predictions into U and V components
            mid_point = uvw_pred.shape[1] // 2
            U_MLP = uvw_pred[:, :mid_point].T + torch.tensor(u_mean_PIV)
            V_MLP = uvw_pred[:, mid_point:].T + torch.tensor(v_mean_PIV)
            
            # Save results
            scipy.io.savemat('Pinball_MLP_PyTorch.mat',
                           {'U_MLP': U_MLP.numpy(),
                            'V_MLP': V_MLP.numpy(),
                            'T_MLP': T_test_PIV,
                            'X_MLP': X_test_PIV,
                            'Y_MLP': Y_test_PIV,
                            'Train_loss_MLP': train_loss_MLP,
                            'Test': epoch_test.numpy()})

print('Training completed')