# Updated boundary conditions [Final Final]
import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the data (for comparison later)
data = loadmat('/content/velocity.mat')
x = data['x']
y = data['y']
u_actual = data['u']
v_actual = data['v']

pressure_data = loadmat('/content/pressure.mat')
p_actual = pressure_data['p']

# Normalize the data
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# Convert to PyTorch tensors and move to device
data_x = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True).view(-1, 1).to(device)
data_y = torch.tensor(y_norm, dtype=torch.float32, requires_grad=True).view(-1, 1).to(device)
data_points = torch.cat((data_x, data_y), dim=1).to(device)

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

# Physics-Informed Loss Function
def pinn_loss(pred, data_points, re=100):
    u_pred, v_pred, p_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:]

    # Define tolerance for boundary detection
    tolerance = 1e-5
    # Lid boundary (y = 1)
    lid_indices = torch.isclose(data_points[:, 1], torch.tensor(1.0, device=device), atol=tolerance)
    # Walls (left: x = 0, right: x = 1, bottom: y = 0)
    left_wall_indices = torch.isclose(data_points[:, 0], torch.tensor(0.0, device=device), atol=tolerance)
    right_wall_indices = torch.isclose(data_points[:, 0], torch.tensor(1.0, device=device), atol=tolerance)
    bottom_wall_indices = torch.isclose(data_points[:, 1], torch.tensor(0.0, device=device), atol=tolerance)
    # Combine wall indices
    wall_indices = left_wall_indices | right_wall_indices | bottom_wall_indices

    # Compute derivatives
    u_x = torch.autograd.grad(u_pred, data_points, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0:1]
    u_y = torch.autograd.grad(u_pred, data_points, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 1:]
    v_x = torch.autograd.grad(v_pred, data_points, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 0:1]
    v_y = torch.autograd.grad(v_pred, data_points, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1:]
    p_x = torch.autograd.grad(p_pred, data_points, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0][:, 0:1]
    p_y = torch.autograd.grad(p_pred, data_points, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0][:, 1:]

    # Second-order derivatives
    u_xx = torch.autograd.grad(u_x, data_points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, data_points, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:]
    v_xx = torch.autograd.grad(v_x, data_points, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, data_points, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:]

    # Continuity equation
    continuity = u_x + v_y

    # Navier-Stokes equations
    momentum_u = u_pred * u_x + v_pred * u_y + p_x - (1 / re) * (u_xx + u_yy)
    momentum_v = u_pred * v_x + v_pred * v_y + p_y - (1 / re) * (v_xx + v_yy)

    # Compute boundary loss only at boundary points
    boundary_loss = torch.mean((u_pred[lid_indices] - 1.0) ** 2) + \
                    torch.mean(v_pred[lid_indices] ** 2) + \
                    torch.mean(u_pred[wall_indices] ** 2) + \
                    torch.mean(v_pred[wall_indices] ** 2)

    # Physics-informed loss terms
    continuity_loss = torch.mean(continuity ** 2)
    momentum_u_loss = torch.mean(momentum_u ** 2)
    momentum_v_loss = torch.mean(momentum_v ** 2)

    # Total physics-informed loss
    return continuity_loss + momentum_u_loss + momentum_v_loss + boundary_loss

# Grid search for optimal network configuration
param_grid = {
    'layers': [[2, 50, 50, 50, 3], [2, 100, 100, 100, 3], [2, 50, 100, 50, 3]]
}
best_loss = float('inf')
best_model = None
best_loss_history = None

for params in ParameterGrid(param_grid):
    model = PINN(params['layers']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loss tracking
    loss_history = []

    for epoch in range(10000):
        optimizer.zero_grad()

        # Predict on entire domain
        predictions = model(data_points)

        # Compute physics-informed loss
        loss = pinn_loss(predictions, data_points)

        # Store loss
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Layers {params['layers']}, Epoch {epoch}: Loss = {loss.item()}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model = model
        best_loss_history = loss_history

# Visualize loss history
plt.figure(figsize=(10, 5))
plt.plot(best_loss_history)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Physics-Informed Loss')
plt.yscale('log')  # Using log scale to better visualize loss reduction
plt.tight_layout()
plt.show()

# Save the best model
torch.save(best_model.state_dict(), 'pinn_lid_driven_cavity.pth')

# Generate predictions
predictions = best_model(data_points).detach().cpu().numpy()
pred_u, pred_v, pred_p = predictions[:, 0], predictions[:, 1], predictions[:, 2]

# Visualization of predictions
plt.figure(figsize=(15, 5))

# Actual u
plt.subplot(1, 3, 1)
plt.scatter(x, y, c=u_actual, cmap='jet', s=5)
plt.title("Actual u-velocity")
plt.colorbar()

# Predicted u
plt.subplot(1, 3, 2)
plt.scatter(x, y, c=pred_u, cmap='jet', s=5)
plt.title("Predicted u-velocity")
plt.colorbar()

# Difference
plt.subplot(1, 3, 3)
plt.scatter(x, y, c=np.abs(u_actual - pred_u), cmap='viridis', s=5)
plt.title("Absolute Error u-velocity")
plt.colorbar()

plt.tight_layout()
plt.show()

# L2 relative error
l2_error_u = np.linalg.norm(u_actual - pred_u) / np.linalg.norm(u_actual)
l2_error_v = np.linalg.norm(v_actual - pred_v) / np.linalg.norm(v_actual)
l2_error_p = np.linalg.norm(p_actual - pred_p) / np.linalg.norm(p_actual)
print(f"L2 Relative Error - u: {l2_error_u}, v: {l2_error_v}, p: {l2_error_p}")