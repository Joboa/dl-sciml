import numpy as np
import os
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

# Function to load data from a folder
def load_data_from_folder(path_to_folder):
    files = [f for f in os.listdir(path_to_folder) if f.endswith('.dat')]
    n_files = len(files)
    
    u_sum = 0.0
    v_sum = 0.0
    
    snapshots_data = []
    
    print('Reading files...')
    for i, file in enumerate(files):
        filepath = os.path.join(path_to_folder, file)
        data = np.loadtxt(filepath, delimiter=' ', skiprows=6)
        
        x = data[:, 0]
        y = data[:, 1]
        u_inst = data[:, 2]
        v_inst = data[:, 3]
        
        x_unique = np.unique(x)
        y_unique = np.unique(y)
    
        rows = len(x_unique)
        cols = len(y_unique)
        
        u_sum += u_inst
        v_sum += v_inst
        
        snapshots_data.append(np.column_stack((u_inst, v_inst)))
        print(f'Time instance read: {i + 1}')
    
    Umean = u_sum / n_files
    Vmean = v_sum / n_files
    print('MEAN COMPUTED!')
    
    U = np.zeros((rows, cols, n_files))
    V = np.zeros((rows, cols, n_files))
    
    k = 0
    for j in range(n_files):
        ufluc = snapshots_data[k][:, 0] - Umean
        vfluc = snapshots_data[k][:, 1] - Vmean
        
        U[:, :, k] = ufluc.reshape((rows, cols))
        V[:, :, k] = vfluc.reshape((rows, cols))
        k += 1
    
    return U, V, x_unique, y_unique

# Define the path to your folder
path_to_folder = os.getenv("DATA_PATH")

# Load and organize data
U, V, x_unique, y_unique = load_data_from_folder(path_to_folder)

# Compute the mean
U_mean = np.mean(U, axis=2)
V_mean = np.mean(V, axis=2)

# Remove the mean
U_fluctuations = U - U_mean[:, :, np.newaxis]
V_fluctuations = V - V_mean[:, :, np.newaxis]

# Reshape data for SVD
rows, cols, time_steps = U.shape
# print(rows, cols, time_steps)
U_R = U_fluctuations.reshape(rows * cols, time_steps)
V_R = V_fluctuations.reshape(rows * cols, time_steps)
vel_data = np.vstack((U_R, V_R))

# Perform SVD
psi, sigma, phi_T = np.linalg.svd(vel_data, full_matrices=False)
mode_energy = (sigma ** 2)

# Energy threshold for reconstruction (50% energy)
cumulative_energy = np.cumsum(mode_energy) / np.sum(mode_energy)
num_modes_to_retain = np.searchsorted(cumulative_energy, 0.50)

# POD modes and temporal coefficients
POD_modes = psi[:, :num_modes_to_retain]
print(POD_modes.shape)
temporal_coeffs = phi_T[:num_modes_to_retain, :]

# Reconstruct using retained modes
U_red = np.dot(POD_modes[:rows*cols, :], temporal_coeffs)
V_red = np.dot(POD_modes[rows*cols:, :], temporal_coeffs)

# Reshape reconstructed data
U_reduced = U_red.reshape(rows, cols, time_steps)
V_reduced = V_red.reshape(rows, cols, time_steps)


# Function to visualize a mode
def visualize_modes(mode, x_unique, y_unique, title, component="U"):
    rows, cols = len(y_unique), len(x_unique)
    mode_reshaped = mode.reshape(rows, cols)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(x_unique, y_unique, mode_reshaped, levels=50, cmap="jet")
    plt.colorbar(label=f"{component} Component")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()

# Visualize the first three modes for U and V components
num_modes_to_visualize = 3
for i in range(num_modes_to_visualize):
    U_mode = POD_modes[:rows*cols, i]  # U component of the mode
    V_mode = POD_modes[rows*cols:, i]  # V component of the mode
    
    visualize_modes(U_mode, x_unique, y_unique, f"Mode {i+1} - U Component", component="U")
    visualize_modes(V_mode, x_unique, y_unique, f"Mode {i+1} - V Component", component="V")

# Plot POD modes
num_modes_to_plot = 3
fig, axes = plt.subplots(1, num_modes_to_plot, figsize=(15, 5))

for i in range(num_modes_to_plot):
    mode = POD_modes[:, i].reshape(rows, cols)
    ax = axes[i]
    cax = ax.imshow(mode, cmap='viridis')
    ax.set_title(f'POD Mode {i+1}')
    fig.colorbar(cax, ax=ax)

plt.tight_layout()
plt.show()



# Visualize reconstructed fields
def visualize_field(field, x_unique, y_unique, title, component="U"):
    rows, cols = len(y_unique), len(x_unique)
    field_reshaped = field.reshape(rows, cols)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(x_unique, y_unique, field_reshaped, levels=50, cmap="jet")
    plt.colorbar(label=f"{component} Component")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()

# Time step to visualize
time_steps_to_visualize = [0, 2, 4, 6, 8, 10]

for t in time_steps_to_visualize:
    U_field = U_reduced[:, :, t]  # Reconstructed U field at time step t
    V_field = V_reduced[:, :, t]  # Reconstructed V field at time step t
    
    visualize_field(U_field, x_unique, y_unique, f"Reconstructed U Field at Time Step {t}", component="U")
    visualize_field(V_field, x_unique, y_unique, f"Reconstructed V Field at Time Step {t}", component="V")


# Plot the cumulative energy
plt.figure(figsize=(8, 6))
plt.plot(cumulative_energy * 100, 'ko--')
plt.xlabel('POD Modes')
plt.ylabel('Cumulative Energy (%)')
plt.title('Cumulative Energy Distribution')
plt.grid(True)
plt.show()

# Plot the percent energy of the first 10 modes
percent_energy = (mode_energy[:10] / np.sum(mode_energy)) * 100
plt.figure(figsize=(8, 6))
plt.plot(percent_energy, 's-')
plt.xlabel('POD Modes')
plt.ylabel('POD Energy (%)')
plt.title('Energy Distribution Among First 10 POD Modes')
plt.grid(True)
plt.show()

# Calculate and plot a1/sqrt(2*lambda1) and a2/sqrt(lambda2)
lambda1 = mode_energy[0] / np.sum(mode_energy)
lambda2 = mode_energy[1] / np.sum(mode_energy)
a1 = temporal_coeffs[0, :]
a2 = temporal_coeffs[1, :]
a1_normalized = a1 / np.sqrt(2 * lambda1)
a2_normalized = a2 / np.sqrt(2* lambda2)

plt.figure(figsize=(8, 6))
plt.plot(a1_normalized, a2_normalized, 'ro')
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel(r'$a_1(t)/\sqrt{2 \lambda_1}$')
plt.ylabel(r'$a_2(t)/\sqrt{\lambda_2}$')
plt.title('Normalized Temporal Coefficients')
plt.axis('equal')
plt.grid(True)
plt.show()
